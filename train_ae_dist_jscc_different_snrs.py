

# file for running the training of the binaryae
import torch
import numpy as np
import copy
import time
import random
from torchvision.transforms.functional import hflip
# from models.binaryae import BinaryGAN
from models.binaryae_jscc_different_snrs import BinaryGAN
from hparams import get_vqgan_hparams
from utils.data_utils import get_data_loaders, cycle
from utils.train_utils import EMA
from utils.log_utils import log, log_stats, save_model, save_stats, save_images, \
                             config_log, start_training_log
from utils.vqgan_utils import load_binaryae_from_checkpoint
from ldm.ldm.util import instantiate_from_config

import misc
import torch.distributed as dist
from omegaconf import OmegaConf
import glob
import os

def main(args):
    misc.init_distributed_mode(args)
    binaryae = BinaryGAN(args).cuda()

    #ldm ae
    ldm_ae_dir = "./ldm/models/ldm/ffhq256"
    # ckpt = "./ldm/models/ldm/ffhq256/model.ckpt"
    base_configs = sorted(glob.glob(os.path.join(ldm_ae_dir, "config.yaml")))

    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = configs[0]
 
    train_dataset = instantiate_from_config(config.data.params.train)
    val_dataset = instantiate_from_config(config.data.params.validation)

    if args.distributed:
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
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=True,
        )
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=4,
            sampler=sampler_val,
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=False,
        )


    # train_loader, val_loader = get_data_loaders(
    #     args.dataset,
    #     args.img_size,
    #     args.batch_size,
    #     get_val_dataloader=(args.steps_per_eval != 0),
    #     custom_dataset_path=args.path_to_data,
    #     num_workers=4,
    #     distributed=args.distributed,
    #     random=True,
    #     args=args, 
    # )

    # if val_loader is not None:
    #     val_iterator = cycle(val_loader)

    # if args.ema:
    #     ema = EMA(args.ema_beta)
    #     ema_binaryae = copy.deepcopy(binaryae)
    # else:
    #     ema_binaryae = None

    if args.ema:
        ema = EMA(args.ema_beta)
        # 1. 重新实例化一个模型，避免使用 deepcopy 复制 LPIPS
        ema_binaryae = BinaryGAN(args).cuda()
        # 2. 加载当前 binaryae 的权重
        ema_binaryae.load_state_dict(binaryae.state_dict())
    else:
        ema_binaryae = None
    
    if args.distributed:
        binaryae = torch.nn.parallel.DistributedDataParallel(binaryae, device_ids=[args.gpu], find_unused_parameters=True)
        binaryae_without_ddp = binaryae.module

    optim = torch.optim.Adam(binaryae_without_ddp.ae.parameters(), lr=args.lr)
    d_optim = torch.optim.Adam(binaryae_without_ddp.disc.parameters(), lr=args.lr)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

    losses = np.array([])
    mean_losses = np.array([])
    val_losses = np.array([])
    recon_losses = np.array([])
    latent_ids = []
    fids = np.array([])
    best_fid = float('inf')

    start_step = 0
    log_start_step = 0
    eval_start_step = args.steps_per_eval
    if args.load_step > 0:
        start_step = args.load_step + 1  # don't repeat the checkpointed step
        binaryae, optim, d_optim, ema_binaryae, train_stats = load_binaryae_from_checkpoint(args, binaryae, optim, d_optim, ema_binaryae)

        # stats won't load for old models with no associated stats file
        if train_stats is not None:
            losses = train_stats["losses"]
            mean_losses = train_stats["mean_losses"]
            val_losses = train_stats["val_losses"]
            # latent_ids = train_stats["latent_ids"]
            fids = train_stats["fids"]
            best_fid = train_stats["best_fid"]
            args.steps_per_log = train_stats["steps_per_log"]
            args.steps_per_eval = train_stats["steps_per_eval"]

            log_start_step = 0
            eval_start_step = args.steps_per_eval
            log('Loaded stats')
        else:
            log_start_step = start_step
            if args.steps_per_eval:
                if args.steps_per_eval == 1:
                    eval_start_step = start_step
                else:
                    eval_start_step = start_step + args.steps_per_eval - start_step % args.steps_per_eval



    log(f"ae params: {(sum(p.numel() for p in binaryae_without_ddp.ae.parameters())/1e6)}M")
    log(f"disc params:{(sum(p.numel() for p in binaryae_without_ddp.disc.parameters()))/1e6}M")
    log(f"total params:{(sum(p.numel() for p in binaryae_without_ddp.ae.parameters())/1e6) + (sum(p.numel() for p in binaryae_without_ddp.disc.parameters()))/1e6}M")

    # for step in range(start_step, args.train_steps):
    step = start_step - 1
    epoch = -1
    while True:
        epoch += 1

        train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            step += 1
            step_start_time = time.time()

            # if isinstance(batch, list):
            #     x = batch[0]
            # elif isinstance(batch, dict):
            #     x = batch['image_tensor']
            # else:
            #     x = batch

            #[b,64,64,3]
            x = batch["image"].cuda() 
            #TODO 归一化到0-1
            x = x / 255.0
            x = x.permute(0, 3, 1, 2)

            if args.horizontal_flip:
                if random.random() <= 0.5:
                    x = hflip(x)

            x = x.cuda()

            if args.amp:
                optim.zero_grad()
                with torch.cuda.amp.autocast():
                    x_hat, stats = binaryae(x, step)
                scaler.scale(stats['loss']).backward()
                scaler.step(optim)
                scaler.update()
            else:
                x_hat, stats = binaryae(x, step)
                optim.zero_grad()
                stats['loss'].backward()
                optim.step()


            if step > args.disc_start_step:
                if args.amp:
                    d_optim.zero_grad()
                    with torch.cuda.amp.autocast():
                        stats = binaryae.module.disc_iter(x_hat.detach(), x, stats)
                    d_scaler.scale(stats['d_loss']).backward()
                    d_scaler.step(d_optim)
                    d_scaler.update()
                else:
                    d_optim.zero_grad()
                    stats = binaryae.module.disc_iter(x_hat, x, stats)
                    stats['d_loss'].backward()
                    d_optim.step()

            torch.cuda.synchronize()

            if step == 1500000:
                for param_group in optim.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                print('lr decay')
            
            # collect latent ids
            # log codebook usage
            if dist.get_rank() == 0:

                code = stats['latent_ids'].cpu().contiguous()
                # code = code.to(torch.bool)
                code = code > 0.5
                latent_ids.append(code)
                # pdb.set_trace()
                if step % 200 == 0 and step > 0:
                    # pdb.set_trace()
                    latent_ids = torch.cat(latent_ids, dim=0).permute(1,0,2,3).reshape(args.codebook_size, -1)
                    codesample_size = latent_ids.shape[1]
                    latent_ids = latent_ids * 1.0

                    latent_ids = latent_ids.sum(-1)

                    odd_idx = ((latent_ids == 0) * 1.0).sum() + ((latent_ids == codesample_size) * 1.0).sum()
                    
                    if int(args.codebook_size - odd_idx) != args.codebook_size:
                        log(f'Codebook size: {args.codebook_size}   Unique Codes Used in Epoch: {args.codebook_size - odd_idx}')
                    latent_ids = []
                
            if step % args.steps_per_log == 0 and dist.get_rank() == 0:
                losses = np.append(losses, stats['loss'].item())
                mean_loss = np.mean(losses)
                stats['loss'] = mean_loss
                step_time = time.time() - step_start_time
                stats['step_time'] = step_time
                mean_losses = np.append(mean_losses, mean_loss)
                recon_losses = np.append(recon_losses, stats['l1'])
                losses = np.array([])

                log_stats(step, stats)

            if args.ema and step % args.steps_per_update_ema == 0 and step > 0:
                ema.update_model_average(ema_binaryae, binaryae)

            if step % args.steps_per_save_output == 0 and dist.get_rank() == 0:
                if args.ema:
                    x_hat, _ = ema_binaryae(x, step)
                save_images(x_hat, 'recons', step, args.log_dir, args.save_individually)

            if step % args.steps_per_checkpoint == 0 and step > args.load_step and dist.get_rank() == 0:

                save_model(binaryae_without_ddp, 'binaryae', step, args.log_dir)
                save_model(optim, 'ae_optim', step, args.log_dir)
                save_model(d_optim, 'disc_optim', step, args.log_dir)
                if args.ema:
                    save_model(ema_binaryae, 'binaryae_ema', step, args.log_dir)

                train_stats = {
                    'losses': losses,
                    'mean_losses': mean_losses,
                    'val_losses': val_losses,
                    'fids': fids,
                    'best_fid': best_fid,
                    'steps_per_log': args.steps_per_log,
                    'steps_per_eval': args.steps_per_eval,
                }
                save_stats(args, train_stats, step)
            
            if step == args.train_steps:
                exit()

if __name__ == '__main__':
    args = get_vqgan_hparams()
    config_log(args.log_dir)
    log('---------------------------------')
    log(f'Setting up training for binaryae on {args.dataset}')
    start_training_log(args)
    main(args)
