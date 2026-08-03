#!/bin/bash
set -e

# ==============================================================================
# BLD_DSC Full Pipeline: Train → Sample → Metrics → Plot
# Ablation: AdaLN + Semantic-Aware Soft Loss
#
# Weights & Large Data → /mnt/data (large storage)
# Final PDF Plots → local scripts/eval/results/pdf
# ==============================================================================

GPU_ID=5
AE_DIR="/home/minimo/Project/BinaryLatentDiffusion/logs/BAE_C64"
AE_STEP=8100000
TRAIN_STEPS=50000
BATCH_SIZE=16
SAMPLE_BATCH=10

# 🌟 新的实验名称 (加入了语义感知损失)
EXP_NAME="binary_diffusion_ffhq_256_qam16_snr_0_15_adaln_semantic_loss"

# 🌟 所有的权重、采样图片、中间评估结果都存到大容量的 /mnt 下
BASE_DIR="/mnt/data/0/mohao/Project/BLD_DSC/logs/${EXP_NAME}"
WEIGHT_DIR="${BASE_DIR}"
RESULT_DIR="${BASE_DIR}/eval_results"

# 🌟 最终生成的 PDF 保存到你指定的本地目录
PDF_DIR="/home/minimo/Project/BLD_DSC/BinaryLatentDiffusion/scripts/eval/results/pdf"

# 确保 PDF 输出目录存在
mkdir -p "${PDF_DIR}"

echo "============================================"
echo "BLD_DSC Full Pipeline (AdaLN + Semantic Loss)"
echo "Weights & Data: ${BASE_DIR}"
echo "Final PDFs:     ${PDF_DIR}"
echo "GPU:            ${GPU_ID}"
echo "============================================"

# ---- Step 1: Training ----
echo ""
echo "[1/4] Training..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m torch.distributed.launch \
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
    --train_steps ${TRAIN_STEPS} \
    --ae_load_dir "${AE_DIR}" \
    --ae_load_step ${AE_STEP} \
    --batch_size ${BATCH_SIZE} \
    --latent_shape 1 16 16 \
    --log_dir "${WEIGHT_DIR}" \
    --loss_final mean \
    --p_flip \
    --norm_first \
    --qam_order 16 \
    --snr_range 0 15 \
    --time_cond_mode adaln

echo "[1/4] Training complete."

# ---- Step 2: Sampling ----
echo ""
echo "[2/4] Sampling at all SNR levels..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12366 \
    --use_env \
    gen_qam_ffhq_256.py \
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
    --batch_size ${SAMPLE_BATCH} \
    --latent_shape 1 16 16 \
    --log_dir "${RESULT_DIR}" \
    --loss_final mean \
    --p_flip \
    --norm_first \
    --load_step ${TRAIN_STEPS} \
    --load_dir "${WEIGHT_DIR}" \
    --qam_order 16 \
    --snr_range 0 15 \
    --time_cond_mode adaln

echo "[2/4] Sampling complete."

# ---- Step 3: Compute Metrics ----
echo ""
echo "[3/4] Computing PSNR & SSIM..."
# 指标计算目录现在也指向 /mnt 下的 eval_results
SAMPLE_DIR="${RESULT_DIR}/samples_for_psnr/16QAM"
GT_DIR="${SAMPLE_DIR}/gt_${TRAIN_STEPS}"
METRICS_DIR="${RESULT_DIR}/metrics_results/16QAM"

python scripts/eval/compute_metrics.py \
    --data_dir "${SAMPLE_DIR}" \
    --gt_dir "${GT_DIR}" \
    --result_dir "${METRICS_DIR}"

echo "[3/4] Metrics complete."

# ---- Step 4: Plot ----
echo ""
echo "[4/4] Generating plots..."

python scripts/eval/plot_metrics.py \
    --result_dir "${METRICS_DIR}" \
    --save_dir "${PDF_DIR}" \
    --label "BLD_DSC (Semantic Loss)"

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "Weights:  ${WEIGHT_DIR}/saved_models/"
echo "Metrics:  ${METRICS_DIR}/"
echo "Plots:    ${PDF_DIR}/"
echo "  → psnr_comparison.pdf"
echo "  → ssim_comparison.pdf"
echo "  → denoise_metrics.pdf"
echo "============================================"