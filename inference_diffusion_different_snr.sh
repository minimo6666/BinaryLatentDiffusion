#!/bin/bash

# 指定显卡
export CUDA_VISIBLE_DEVICES="2"

echo "🚀 开始基于 Diffusion 去噪器的信噪比交叉评估..."
echo "📂 模型读取目录: /mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15"
echo "💾 结果保存目录: /mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15/eval_snr_0_to_15"

# 启动分布式验证 (完全对齐你的 sample launch.json)
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12366 \
    --use_env \
    gen_imgs_under_diff_snr_with_our_diffusion_sampler.py \
    --sampler bld_dsc \
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
    --ae_load_dir "/home/minimo/Project/BinaryLatentDiffusion/logs/BAE_C64" \
    --ae_load_step 8100000 \
    --batch_size 10 \
    --latent_shape 1 16 16 \
    --log_dir "/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15/eval_snr_0_to_15" \
    --loss_final mean \
    --p_flip \
    --norm_first \
    --load_step 100000 \
    --load_dir "/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15" \
    --save_individually \
    --qam_order 16 \
    --snr_range 0 15

echo "🎉 Diffusion 去噪评估测试完毕！"