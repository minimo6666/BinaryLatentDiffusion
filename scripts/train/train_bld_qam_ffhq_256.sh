#!/bin/bash

# 分布式训练启动命令 #gray coding support
# 消融实验: AdaLN time conditioning (sole change vs. baseline)
CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12350 \
    --use_env \
    train_sampler_online_qam_ffhq_256.py \
    --sampler bld_dsc \
    --dataset ffhq \
    --ema \
    --steps_per_checkpoint 50000 \
    --codebook_size 64 \
    --img_size 256 \
    --steps_per_display_output 10000 \
    --steps_per_save_output 10000 \
    --steps_per_log 100 \
    --total_steps 64 \
    --sample_steps 64 \
    --beta_type linear \
    --amp \
    --train_steps 200000 \
    --ae_load_dir "/home/minimo/Project/BinaryLatentDiffusion/logs/BAE_C64" \
    --ae_load_step 8100000 \
    --batch_size 32 \
    --latent_shape 1 16 16 \
    --log_dir "/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15" \
    --loss_final mean \
    --p_flip \
    --norm_first \
    --qam_order 16 \
    --snr_range 0 15 \
    --time_cond_mode adaln \
    # --load_step 20000 \
    # --load_dir /mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15