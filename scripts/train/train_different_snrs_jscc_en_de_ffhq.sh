
export CUDA_VISIBLE_DEVICES=1

for snr in -15 0 4 8 25; do
# for snr in -15 25; do

python -m torch.distributed.launch --nproc_per_node=1 --master_port=25680 --use_env  /home/minimo/Project/BLD_DSC/BinaryLatentDiffusion/train_ae_dist_jscc_different_snrs.py \
--dataset custom --path_to_data ./data/dataset/DIV2K_256/ --amp --ema --steps_per_save_output 5000 --codebook_size 96 --steps_per_log 200 \
--steps_per_checkpoint 100000 --img_size 256 --batch_size 6 --latent_shape 1 32 32 --log_dir /mnt/data/0/mohao/Project/BLD/train_logs/AWGN/different_snrs_jscc_en_de_ffhq/snr_$snr \
--disc_start_step 40001 --norm_first --deterministic --ch_mult 1 1 2 4 --train_steps 100000 --snr $snr --qam_order 16 \

done

