#!/bin/bash

# 指定使用的显卡
export CUDA_VISIBLE_DEVICES=6

# 定义需要加载的训练模型 SNR 列表
TRAIN_SNRS=(-15 0 4 8 25)

# 定义需要测试的信道环境 SNR 列表
TEST_SNRS=(-15 0 4 8 12 15 25)

echo "🚀 开始全量交叉评估 (Train SNR × Test SNR)..."
echo "📂 模型读取目录: /mnt/data/0/mohao/Project/BLD/train_logs/AWGN/different_snrs_jscc_en_de_ffhq/"
echo "💾 结果保存目录: /mnt/data/0/mohao/Project/BLD/train_logs/AWGN/different_snrs_jscc_en_de_ffhq/"

# 外层循环：遍历不同训练条件下得到的模型
for train_snr in "${TRAIN_SNRS[@]}"; do
    echo " "
    echo "=========================================================================="
    echo "📂 [MODEL LOAD] 正在加载在 ${train_snr}dB 下训练的模型权重..."
    echo "=========================================================================="

    # 内层循环：测试当前模型在不同信道干扰下的鲁棒性
    for test_snr in "${TEST_SNRS[@]}"; do
        echo "  --> 🧪 [TEST] 当前测试信道 SNR: ${test_snr}dB"

        python -m torch.distributed.launch --nproc_per_node=1 --master_port=25680 --use_env \
        gen_imgs_same_jscc_under_diff_snr.py \
        --dataset custom \
        --path_to_data ./data/dataset/DIV2K_256/ \
        --codebook_size 96 \
        --img_size 256 \
        --batch_size 10 \
        --latent_shape 1 32 32 \
        --qam_order 64 \
        --norm_first \
        --deterministic \
        --ch_mult 1 1 2 4 \
        --load_dir /mnt/data/0/mohao/Project/BLD/train_logs/AWGN/different_snrs_jscc_en_de_ffhq/snr_${train_snr} \
        --load_step 100000 \
        --log_dir /mnt/data/0/mohao/Project/BLD/train_logs/AWGN/different_snrs_jscc_en_de_ffhq/en_de_train_under_snr_${train_snr}/snr_${test_snr} \
        --snr ${test_snr} \
        --amp \
        --ema

    done
done

echo "🎉 所有模型组合的交叉测试已全部完成！"