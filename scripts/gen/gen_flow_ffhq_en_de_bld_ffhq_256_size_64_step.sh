#!/bin/bash

CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 12340 \
    --use_env \
    ./evaluation/vip/ffhq/gen_imgs_ffhq_en_de_bld_ffhq_256_size_64_steps.py \
    --sampler bld \
    --dataset ffhq \
    --ema \
    --steps_per_checkpoint 10000 \
    --codebook_size 64 \
    --img_size 256 \
    --steps_per_display_output 5000 \
    --steps_per_save_output 5000 \
    --steps_per_log 100 \
    --total_steps 64 \
    --sample_steps 64 \
    --beta_type linear \
    --amp \
    --train_steps 2000000 \
    --ae_load_dir logs/BAE_C64 \
    --ae_load_step 8100000 \
    --batch_size 64 \
    --latent_shape 1 16 16 \
    --log_dir /mnt/data/0/mohao/Project/BinaryLatentDiffusion/logs/original_bld_en_de_ffhq_256_9_1 \
    --loss_final mean \
    --p_flip \
    --norm_first \
    --load_step 200000 \
    --load_dir /mnt/data/0/mohao/Project/BinaryLatentDiffusion/logs/original_bld_en_de_ffhq_256_9_1