#!/bin/bash
set -e

# ==============================================================================
# BLD_DSC Full Pipeline: Train → Sample 1000 imgs → Plot vs JSCC
# Variant: Codebook-Norm Weighted BCE Loss (码本范数加权)
#
# Weights & Large Data → /mnt/data (large storage)
# Final PDF Plots → local scripts/eval/results/pdf
# ==============================================================================

GPU_ID=6
AE_DIR="/home/minimo/Project/BinaryLatentDiffusion/logs/BAE_C64"
AE_STEP=8100000
TRAIN_STEPS=50000
BATCH_SIZE=16

# 🌟 新实验名称
EXP_NAME="binary_diffusion_ffhq_256_qam16_snr_0_15_adaln_weighted_bce"

# 🌟 权重存 /mnt
BASE_DIR="/mnt/data/0/mohao/Project/BLD_DSC/logs/${EXP_NAME}"
WEIGHT_DIR="${BASE_DIR}"
RESULT_DIR="${BASE_DIR}/eval_results"

# 🌟 JSCC 1000 张对比数据
JSCC_BASE="/mnt/data/0/mohao/Project/BLD_DSC/logs/different_snrs_jscc_en_de_ffhq"

# 🌟 最终 PDF 输出目录
PDF_DIR="/home/minimo/Project/BLD_DSC/BinaryLatentDiffusion/scripts/eval/results/pdf"
mkdir -p "${PDF_DIR}"

echo "============================================"
echo "BLD_DSC Full Pipeline (Weighted BCE Loss)"
echo "Experiment: ${EXP_NAME}"
echo "Weights:    ${BASE_DIR}"
echo "GPU:        ${GPU_ID}"
echo "Sampling:   1000 images"
echo "============================================"

# ---- Step 1: Training ----
echo ""
echo "[1/3] Training with Codebook-Norm Weighted BCE..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12360 \
    --use_env \
    train_sampler_online_qam_ffhq_256.py \
    --sampler bld_dsc_weighted \
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

echo "[1/3] Training complete."

# ---- Step 2: Sampling 1000 images + Compute Metrics ----
echo ""
echo "[2/3] Sampling 1000 images at 9 SNR levels + computing metrics..."

# 🌟 取 9 个关键 SNR 点，跟 JSCC 的 test SNRs 对齐用于画图
EVAL_SNRS="0,2,4,6,8,10,12,14,15"

PORT=$((26000 + RANDOM % 5000))
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port ${PORT} \
    --use_env \
    scripts/eval/sample_and_eval.py \
    --sampler bld_dsc_weighted \
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
    --load_step ${TRAIN_STEPS} \
    --load_dir "${WEIGHT_DIR}" \
    --qam_order 16 \
    --snr_range 0 15 \
    --time_cond_mode adaln \
    --result_dir "${RESULT_DIR}" \
    --num_images 1000 \
    --eval_snrs "${EVAL_SNRS}"

echo "[2/3] Sampling & metrics complete."

# ---- Step 3: Plot JSCC vs BLD_DSC (Weighted BCE) ----
echo ""
echo "[3/3] Plotting JSCC (1000 imgs) vs BLD_DSC Weighted BCE (1000 imgs)..."

BLD_METRICS="${RESULT_DIR}/metrics_results/16QAM"

python scripts/eval/plot_jscc_vs_bld.py \
    --jscc_base "${JSCC_BASE}" \
    --bld_metrics "${BLD_METRICS}" \
    --bld_label "BLD_DSC (Weighted BCE)" \
    --save_path "${PDF_DIR}/JSCC_vs_Ours_WeightedBCE_1000_PSNR.pdf"

# 复制 sample_and_eval 自己生成的图也拉过来 (BER 等)
cp ${RESULT_DIR}/pdf/*.pdf ${PDF_DIR}/ 2>/dev/null || true

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "Weights:  ${WEIGHT_DIR}/saved_models/"
echo "Metrics:  ${BLD_METRICS}/"
echo "Plots:    ${PDF_DIR}/"
echo "  → JSCC_vs_Ours_WeightedBCE_1000_PSNR.pdf"
echo "  → BER_Improvement.pdf"
echo "============================================"
