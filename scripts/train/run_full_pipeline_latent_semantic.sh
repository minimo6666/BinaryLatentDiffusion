#!/bin/bash
set -e

# ==============================================================================
# BLD_DSC Full Pipeline: Train → Sample → Metrics → Plot
# Ablation: AdaLN + Latent Semantic Distance Loss (潜空间语义距离损失)
#
# 与 pixel-level MSE (semantic_loss) 的区别：
#   - 不穿过 BAE Decoder 回传梯度
#   - 直接在 Codebook 张成的连续潜空间中计算语义距离
#   - x_0 @ Q = Ground Truth, sigmoid(logits) @ Q = Prediction
#
# Results → experiments/<EXP_NAME>/ (weights, eval, pdfs)
# ==============================================================================

GPU_ID=2
AE_DIR="/home/minimo/Project/BinaryLatentDiffusion/logs/BAE_C64"
AE_STEP=8100000
TRAIN_STEPS=50000
BATCH_SIZE=16
SAMPLE_BATCH=10

# 🌟 新实验名称 (Latent Semantic Distance Loss)
EXP_NAME="binary_diffusion_ffhq_256_qam16_snr_0_15_adaln_latent_semantic"

# 🌟 所有产物存到 experiments/<EXP_NAME>/
BASE_DIR="/home/minimo/Project/BLD_DSC/experiments/${EXP_NAME}"
WEIGHT_DIR="${BASE_DIR}"
RESULT_DIR="${BASE_DIR}/eval_results"
PDF_DIR="${BASE_DIR}/pdf"

mkdir -p "${PDF_DIR}"

echo "============================================"
echo "BLD_DSC Full Pipeline"
echo "Method:       AdaLN + Latent Semantic Distance Loss"
echo "Experiment:   ${EXP_NAME}"
echo "Weights:      ${WEIGHT_DIR}"
echo "Eval Results: ${RESULT_DIR}"
echo "Final PDFs:   ${PDF_DIR}"
echo "GPU:          ${GPU_ID}"
echo "============================================"

# ---- Step 1: Training ----
echo ""
echo "[1/4] Training with Latent Semantic Distance Loss..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12350 \
    --use_env \
    train_sampler_online_qam_ffhq_256.py \
    --sampler bld_dsc_latent \
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
    --sampler bld_dsc_latent \
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
    --label "BLD_DSC (Latent Semantic)"

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "Weights:  ${WEIGHT_DIR}/saved_models/"
echo "Metrics:  ${METRICS_DIR}/"
echo "Plots:    ${PDF_DIR}/"
echo "============================================"
