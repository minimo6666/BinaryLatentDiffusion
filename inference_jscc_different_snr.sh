#!/bin/bash

# 指定使用的显卡
export CUDA_VISIBLE_DEVICES=6

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR="/mnt/data/0/mohao/Project/BLD_DSC/logs/different_snrs_jscc_en_de_ffhq"

# 定义需要加载的训练模型 SNR 列表
TRAIN_SNRS=(-15 0 4 8 25)

# 定义需要测试的信道环境 SNR 列表
TEST_SNRS=(-15 0 4 8 12 15 25)

# 先清理可能残留的僵尸进程
pkill -f "gen_imgs_same_jscc_under_diff_snr" 2>/dev/null || true
sleep 1

echo "🚀 开始全量交叉评估 (Train SNR × Test SNR)..."
echo "📂 模型读取目录: ${BASE_DIR}/"
echo "💾 结果保存目录: ${BASE_DIR}/"
echo "🖼️  每对 (train_snr, test_snr) 采样图片数: 1000"

# 外层循环：遍历不同训练条件下得到的模型
for train_snr in "${TRAIN_SNRS[@]}"; do
    echo " "
    echo "=========================================================================="
    echo "📂 [MODEL LOAD] 正在加载在 ${train_snr}dB 下训练的模型权重..."
    echo "=========================================================================="

    # 内层循环：测试当前模型在不同信道干扰下的鲁棒性
    for test_snr in "${TEST_SNRS[@]}"; do
        echo "  --> 🧪 [TEST] 当前测试信道 SNR: ${test_snr}dB"

        # 每次都生成随机端口，避免端口残留冲突
        PORT=$((26000 + RANDOM % 5000))

        python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT} --use_env \
        gen_imgs_same_jscc_under_diff_snr.py \
        --dataset custom \
        --codebook_size 96 \
        --img_size 256 \
        --batch_size 10 \
        --latent_shape 1 32 32 \
        --qam_order 64 \
        --norm_first \
        --deterministic \
        --ch_mult 1 1 2 4 \
        --load_dir "${BASE_DIR}/snr_${train_snr}" \
        --load_step 100000 \
        --log_dir "${BASE_DIR}/en_de_train_under_snr_${train_snr}/snr_${test_snr}" \
        --snr ${test_snr} \
        --amp \
        --ema

        sleep 2  # 确保端口释放干净再进下一轮
    done
done

echo "🎉 所有模型组合的交叉测试已全部完成！"

# =============================================================================
# 画图：JSCC vs BLD_DSC (Semantic Loss)
# =============================================================================
BLD_METRICS="/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15_adaln_semantic_loss/eval_results/metrics_results/16QAM"
PDF_DIR="/home/minimo/Project/BLD_DSC/BinaryLatentDiffusion/scripts/eval/results/pdf"
mkdir -p "${PDF_DIR}"

echo ""
echo "📊 正在生成 JSCC vs BLD_DSC 对比图..."
python scripts/eval/plot_jscc_vs_bld.py \
    --jscc_base "${BASE_DIR}" \
    --bld_metrics "${BLD_METRICS}" \
    --save_path "${PDF_DIR}/JSCC_vs_Ours_Semantic_1000_PSNR.pdf"

echo ""
echo "============================================"
echo "✅ 画图完成！"
echo "📄 ${PDF_DIR}/JSCC_vs_Ours_Semantic_1000_PSNR.pdf"
echo "============================================"
