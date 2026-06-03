#!/bin/bash
set -e

# ==============================================================================
# Memory-efficient Sampling + Metrics + Plot (one-by-one BAE encoding)
# Usage: bash scripts/eval/run_sample_and_eval.sh
# ==============================================================================

GPU_ID=1
AE_DIR="/home/minimo/Project/BinaryLatentDiffusion/logs/BAE_C64"
AE_STEP=8100000
LOAD_STEP=50000
WEIGHT_DIR="/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15_adaln"
RESULT_DIR="/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15_adaln"
JSCC_DIR="/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15/experiments/jscc_different_snr_performance"

echo "============================================"
echo "BLD_DSC Sampling + Metrics + Plot (AdaLN)"
echo "Weights: ${WEIGHT_DIR}"
echo "Load step: ${LOAD_STEP}"
echo "Results: ${RESULT_DIR}"
echo "GPU:     ${GPU_ID}"
echo "============================================"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12366 \
    --use_env \
    scripts/eval/sample_and_eval.py \
    --sampler bld_dsc \
    --dataset ffhq \
    --ema \
    --codebook_size 64 \
    --img_size 256 \
    --total_steps 64 \
    --sample_steps 64 \
    --beta_type linear \
    --amp \
    --ae_load_dir "${AE_DIR}" \
    --ae_load_step ${AE_STEP} \
    --batch_size 16 \
    --latent_shape 1 16 16 \
    --log_dir "${RESULT_DIR}" \
    --loss_final mean \
    --p_flip \
    --norm_first \
    --load_step ${LOAD_STEP} \
    --load_dir "${WEIGHT_DIR}" \
    --qam_order 16 \
    --snr_range 0 15 \
    --time_cond_mode adaln \
    --result_dir "${RESULT_DIR}" \
    --num_images 100 \
    --eval_snrs "0,2,4,6,8,10,12,14" \
    --jscc_dir "${JSCC_DIR}"

echo ""
echo "============================================"
echo "Done!"
echo "Images:  ${RESULT_DIR}/samples_for_psnr/16QAM/"
echo "Metrics: ${RESULT_DIR}/metrics_results/16QAM/"
echo "Plots:   ${RESULT_DIR}/pdf/"
echo "  → psnr_comparison.pdf"
echo "  → ssim_comparison.pdf"
echo "  → denoise_metrics.pdf"
echo "============================================"
