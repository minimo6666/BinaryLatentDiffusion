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
import torchvision

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


        # 获取前1000张验证图像
    val_images = []
    num_images_needed = 100
    count = 0

    with torch.no_grad():
        for val_batch in val_loader:
            if count >= num_images_needed:
                break
                
            batch_size = val_batch["image"].size(0)
            remaining = num_images_needed - count
            take = min(batch_size, remaining)
            
            val_img_batch = val_batch["image"][:take].cuda()
            val_img_batch = val_img_batch / 255.0
            val_img_batch = val_img_batch.permute(0, 3, 1, 2)
            
            val_images.append(val_img_batch)
            count += take

    # 合并所有批次并保存原始图像
    val_img = torch.cat(val_images, dim=0)
    # save_images(val_img, 'samples', step, H.log_dir, H.save_individually, name='samples_for_psnr/gt_100')

    # 编码得到1000个x_val
    with torch.no_grad():
        code_val = bergan(val_img, code_only=True).detach()
        b, c, h, w = code_val.shape
        x_val = code_val.view(b, c, -1).permute(0, 2, 1).contiguous()

    # 对每个噪声水平进行处理
    for noise_t in range(63, -1, -1):
        print(f'Processing noise level {noise_t}')
        eb_n0_db = Eb_N0_dB_value[noise_t]
        eb_n0_db_str = f"{eb_n0_db:.2f}"  # 格式化为两位小数
        
        # 创建目录保存当前噪声水平的图像
        os.makedirs(f'{H.log_dir}/samples_for_psnr/{eb_n0_db_str}_db', exist_ok=True)
        
        # 分批处理
        num_batches = (num_images_needed + H.batch_size - 1) // H.batch_size
        
  
        global_idx = 0  # 用于全局图像编号
        for batch_idx in range(num_batches):
            start_idx = batch_idx * H.batch_size
            end_idx = min((batch_idx + 1) * H.batch_size, num_images_needed)
            
            # 获取当前批次的编码
            x_val_batch = x_val[start_idx:end_idx]
            
            # 添加噪声
            x_val_noisy_code_t_batch = make_code_noise(
                x_val_batch, eb_n0_db=eb_n0_db, qam_order=H.qam_order, device=x_val_batch.device
            )

            #解码噪声图像
            noisy_images_batch = get_online_decode_code_into_images(x_val_noisy_code_t_batch, bergan, H)
            
            # # 去噪
            # x_val_denoised_batch = get_online_samples_denoise_code(
            #     ema_sampler if H.ema else sampler, x_val_noisy_code_t_batch, noise_t
            # )
            
            # # 解码为图像
            # images_batch = get_online_decode_code_into_images(x_val_denoised_batch, bergan, H)

            # denoised_name=f'/samples_for_psnr/{H.qam_order}QAM/denoise/{eb_n0_db_str}_db'

            # for idx in range(len(images_batch)):
            #     torchvision.utils.save_image(torch.clamp(images_batch[idx], 0, 1), f"{H.log_dir}/{denoised_name}/{step}_{global_idx}.png")
            #     global_idx += 1
            
            noise_prefix = f"{H.log_dir}/samples_for_psnr/{H.qam_order}QAM/noise/{eb_n0_db_str}_db"
            os.makedirs(noise_prefix, exist_ok=True)

            for idx in range(len(noisy_images_batch)):
                torchvision.utils.save_image(
                    torch.clamp(noisy_images_batch[idx], 0, 1), 
                    f"{noise_prefix}/{step}_{global_idx}.png"
                )


if __name__ == '__main__':
    H = get_sampler_hparams()
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)
    main(H, None)
