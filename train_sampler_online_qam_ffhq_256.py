import torch
import numpy as np
import copy
import time
import os
from torch.nn import Module
from models.binaryae import BinaryAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.data_utils import get_data_loaders
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts,\
    get_sampler, get_online_samples,  get_online_samples_with_noisy_code, get_online_samples_denoise_code, get_online_decode_code_into_images
from utils.train_utils import EMA, NativeScalerWithGradNormCount
from utils.log_utils import log, log_stats, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images, \
    MovingAverage
import misc
import torch.distributed as dist
from utils.lr_sched import adjust_lr, lr_scheduler
from omegaconf import OmegaConf
import glob
from ldm.ldm.util import instantiate_from_config
from PIL import Image
from utils.qam_utils import *
from utils.m_qam_awgn_util import *

def make_code_noise(clean_code, eb_n0_db, qam_order=16, device=None):
    received_code = make_code_noise_single(clean_code, eb_n0_db=eb_n0_db, qam_order=qam_order, device=device)
    return received_code

def main(H, vis):

    misc.init_distributed_mode(H)

    Eb_N0_db_range = torch.tensor([H.snr_range[0], H.snr_range[1]])

    Eb_N0_dB_value = torch.linspace(Eb_N0_db_range[1], Eb_N0_db_range[0], 64)   

    #ldm ae
    ldm_ae_dir = "./ldm/models/ldm/ffhq256"
    # ckpt = "./ldm/models/ldm/ffhq256/model.ckpt"
    base_configs = sorted(glob.glob(os.path.join(ldm_ae_dir, "config.yaml")))

    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = configs[0]
 
    # model

    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['encoder', 'quantize', 'generator'],
        remove_component_from_key=False
    )

    bergan = BinaryAutoEncoder(H)
    bergan.load_state_dict(ae_state_dict, strict=True)
    bergan = bergan.cuda()
    del ae_state_dict


    sampler = get_sampler(H, None).cuda()

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    if H.distributed:
        find_unused = H.guidance
        sampler = torch.nn.parallel.DistributedDataParallel(sampler, device_ids=[H.gpu], find_unused_parameters=find_unused)
        sampler_without_ddp = sampler.module

    optim_eps = H.optim_eps
    optim = torch.optim.AdamW(sampler_without_ddp.parameters(), lr=H.lr, weight_decay=H.weight_decay, betas=(0.9, 0.95), eps=optim_eps)

    losses = np.array([])
    val_losses = np.array([])
    elbo = np.array([])
    val_elbos = np.array([])
    mean_losses = np.array([])
    start_step = 0
    log_start_step = 0

    loss_ma = MovingAverage(100)

    if H.load_model_step > 0:
        device = sampler.device
        sampler = load_model(sampler, H.sampler, H.load_model_step, H.load_model_dir, device=device).cuda()

        
    scaler = NativeScalerWithGradNormCount(H.amp, H.init_scale)

    if H.load_step > 0:
        start_step = H.load_step + 1

        device = sampler.device

        allow_mismatch = H.allow_mismatch
        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch)
            except Exception:
                ema_sampler = copy.deepcopy(sampler_without_ddp)
        
        if not allow_mismatch:
            if H.load_optim:
                optim = load_model(
                    optim, f'{H.sampler}_optim', H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch)
                for param_group in optim.param_groups:
                    param_group['lr'] = H.lr
        try:
            train_stats = load_stats(H, H.load_step)
        except Exception:
            train_stats = None

        if not H.reset_step:
            if not H.reset_scaler:
                try:
                    scaler.load_state_dict(torch.load(os.path.join(H.load_dir, 'saved_models', f'absorbingbnl_scaler_{H.load_step}.th')))
                except Exception:
                    print('Failing to load scaler.')
        else:
            H.load_step = 0

        
        if train_stats is not None:
            losses, mean_losses, val_losses, elbo, H.steps_per_log

            losses = train_stats["losses"],
            mean_losses = train_stats["mean_losses"],
            val_losses = train_stats["val_losses"],
            val_elbos = train_stats["val_elbos"]
            log_start_step = 0

            losses = losses[0]
            mean_losses = mean_losses[0]
            val_losses = val_losses[0]
            val_elbos = torch.Tensor([0])

        else:
            log('No stats file found for loaded model, displaying stats from load step only.')
            log_start_step = start_step

        if H.reset_step:
            start_step = 0
    
  
    
    train_dataset = instantiate_from_config(config.data.params.train)
    val_dataset = instantiate_from_config(config.data.params.validation)

    if H.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_train = %s" % str(sampler_train))
            
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=4,
            sampler=sampler_train,
            batch_size=H.batch_size,
            pin_memory=True,
            drop_last=True,
        )
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=4,
            sampler=sampler_val,
            batch_size=H.batch_size,
            pin_memory=True,
            drop_last=False,
        )

    # for step in range(start_step, H.train_steps):
    H.train_steps = H.train_steps * H.update_freq
    H.warmup_iters = H.warmup_iters * H.update_freq
    H.steps_per_log = H.steps_per_log * H.update_freq
    lr_sched = lr_scheduler(base_value=H.lr, final_value=1e-6, iters=H.train_steps+1, warmup_steps=H.warmup_iters,
                     start_warmup_value=1e-6, lr_type='constant')
    print(lr_sched)
    step = start_step - 1
    epoch = -1

    optim.zero_grad()
    label = None
    while True:
        epoch += 1
        train_loader.sampler.set_epoch(epoch)
        for data in train_loader:
            step += 1

            adjust_lr(optim, lr_sched, step)
            step_start_time = time.time()
            #[b,64,64,3]
            img = data["image"].cuda() 
            #TODO 归一化到0-1
            img = img / 255.0
            img = img.permute(0, 3, 1, 2)

            with torch.no_grad():
                code = bergan(img, code_only=True).detach()
                b,c,h,w = code.shape
                x = code.view(b,c,-1).permute(0,2,1).contiguous()

            with torch.cuda.amp.autocast(enabled=H.amp):
             
                stats = sampler(x)
                loss = stats['loss']
                loss = loss / H.update_freq

            if step == 0 and dist.get_rank() == 0:
                images = get_online_samples(H, bergan, ema_sampler if H.ema else sampler, x=x)
                save_images(images, 'samples', 999999999, H.log_dir, H.save_individually)
                # save to test the reconstruction quality

            grad_norm = scaler(loss, optim, clip_grad=H.grad_norm,
                                parameters=sampler_without_ddp.parameters(), create_graph=False,
                                update_grad=(step + 1) % H.update_freq == 0)

            if (step + 1) % H.update_freq == 0:
                optim.zero_grad()
            loss_ma.update(loss.item())
            if H.ema and step % (H.steps_per_update_ema * H.update_freq) == 0 and step > 0:
                ema.update_model_average(ema_sampler, sampler)

            torch.cuda.synchronize()

            if dist.get_rank() == 0:
                if step % H.steps_per_log == 0:

                    stats['lr'] = optim.param_groups[0]['lr']
                    step_time_taken = time.time() - step_start_time
                    stats['step_time'] = step_time_taken
                    mean_loss = np.mean(losses)
                    stats['mean_loss'] = loss_ma.avg()

                    if "scale" in scaler.state_dict().keys():
                        stats['loss scale'] = scaler.state_dict()["scale"]
                    mean_losses = np.append(mean_losses, mean_loss)
                    losses = np.array([])

                    log_stats(step, stats)

                if step % H.steps_per_save_output == 0:
                 
                    # TODO add noise to validate image encoded latent code x_0 into x_t, pass in start t
                    # 取val_loader的第一个batch
                    val_data = next(iter(val_loader))
                    val_img = val_data["image"].cuda()
                    val_img = val_img / 255.0
                    val_img = val_img.permute(0, 3, 1, 2)

                    with torch.no_grad():
                        code_val = bergan(val_img, code_only=True).detach()
                        b,c,h,w = code_val.shape
                        x_val = code_val.view(b,c,-1).permute(0,2,1).contiguous()
                    
                    noise_t = 63
                    
                    #gray code qam noise has been added inside make_code_noise
                    eb_n0_db = Eb_N0_dB_value[noise_t]
                    x_val_noisy_code_t = make_code_noise(x_val, eb_n0_db=eb_n0_db, qam_order=H.qam_order, device=x_val.device)
                    x_val_denoised = get_online_samples_denoise_code(ema_sampler if H.ema else sampler, x_val_noisy_code_t, noise_t)
                    images = get_online_decode_code_into_images(x_val_denoised, bergan, H)

                    save_images(images, 'samples', step, H.log_dir, H.save_individually)
                    # save_images(val_img, 'gt', step, H.log_dir, H.save_individually)


                if step % H.steps_per_checkpoint == 0 and step > H.load_step:
                    save_model(sampler, H.sampler, step, H.log_dir)
                    save_model(optim, f'{H.sampler}_optim', step, H.log_dir)
                    save_model(scaler, f'{H.sampler}_scaler', step, H.log_dir)

                    if H.ema:
                        save_model(ema_sampler, f'{H.sampler}_ema', step, H.log_dir)

                    train_stats = {
                        'losses': losses,
                        'mean_losses': mean_losses,
                        'val_losses': val_losses,
                        'elbo': elbo,
                        'val_elbos': val_elbos,
                        'steps_per_log': H.steps_per_log,
                        'steps_per_eval': H.steps_per_eval,
                    }
                    save_stats(H, train_stats, step)
            
            if step == H.train_steps:
                if dist.get_rank() == 0:
                    print(f"Training complete at step {step}.")
                return


if __name__ == '__main__':
    H = get_sampler_hparams()
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)
    main(H, None)
