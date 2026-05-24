import torch
import numpy as np
import copy
import time
import os
import math
from torch.nn import Module
from models.binaryae import BinaryAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.data_utils import get_data_loaders
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts,\
    get_sampler, get_online_samples,  get_online_samples_with_noisy_code, get_online_samples_denoise_code, get_online_decode_code_into_images
from utils.train_utils import EMA, NativeScalerWithGradNormCount
from utils.log_utils import log, log_stats, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images, \
    MovingAverage
import misc
import torch.distributed as dist
from utils.lr_sched import adjust_lr, lr_scheduler
from omegaconf import OmegaConf
import glob
from ldm.ldm.util import instantiate_from_config
from PIL import Image
from utils.qam_utils import *
from utils.m_qam_awgn_util import *
import torchvision

def make_code_noise(clean_code, eb_n0_db, qam_order=16, device=None):
    received_code = make_code_noise_single(clean_code, eb_n0_db=eb_n0_db, qam_order=qam_order, device=device)
    return received_code

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def main(H, vis):
    misc.init_distributed_mode(H)

    Eb_N0_db_range = torch.tensor([H.snr_range[0], H.snr_range[1]])
    Eb_N0_dB_value = torch.linspace(Eb_N0_db_range[1], Eb_N0_db_range[0], 64)   

    # ldm ae
    ldm_ae_dir = "./ldm/models/ldm/ffhq256"
    base_configs = sorted(glob.glob(os.path.join(ldm_ae_dir, "config.yaml")))
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = configs[0]
 
    # ==========================
    # 1. 加载 BAE
    # ==========================
    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H, ['encoder', 'quantize', 'generator'], remove_component_from_key=False
    )
    bergan = BinaryAutoEncoder(H)
    bergan.load_state_dict(ae_state_dict, strict=True)
    bergan = bergan.cuda()
    bergan.eval()
    del ae_state_dict

    # ==========================
    # 2. 100% 还原你的 Sampler 加载逻辑 (杜绝乱码核心！)
    # ==========================
    sampler = get_sampler(H, None).cuda()

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    if H.distributed:
        find_unused = H.guidance
        sampler = torch.nn.parallel.DistributedDataParallel(sampler, device_ids=[H.gpu], find_unused_parameters=find_unused)
        sampler_without_ddp = sampler.module
    else:
        sampler_without_ddp = sampler

    # 安全获取 device
    device = next(sampler.parameters()).device

    if H.load_model_step > 0:
        sampler = load_model(sampler, H.sampler, H.load_model_step, H.load_model_dir, device=device).cuda()

    if H.load_step > 0:
        allow_mismatch = H.allow_mismatch
        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch).cuda()
        if H.ema:
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch)
                if misc.get_rank() == 0: print("✅ EMA 权重加载成功！")
            except Exception:
                ema_sampler = copy.deepcopy(sampler_without_ddp)
                if misc.get_rank() == 0: print("⚠️ EMA 权重加载失败，使用普通权重！")
    
    sampler.eval()
    if H.ema: ema_sampler.eval()

    # ==========================
    # 3. 数据集准备
    # ==========================
    val_dataset = instantiate_from_config(config.data.params.validation)

    if H.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_val = torch.utils.data.DistributedSampler(val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, num_workers=4, sampler=sampler_val, batch_size=H.batch_size, pin_memory=True, drop_last=False
    )

    # 提取你要测试的 50 张图
    val_images = []
    num_images_needed = 50
    count = 0

    with torch.no_grad():
        for val_batch in val_loader:
            if count >= num_images_needed:
                break
            batch_size = val_batch["image"].size(0)
            remaining = num_images_needed - count
            take = min(batch_size, remaining)
            
            val_img_batch = val_batch["image"][:take].cuda()
            val_img_batch = val_img_batch / 255.0
            val_img_batch = val_img_batch.permute(0, 3, 1, 2)
            
            val_images.append(val_img_batch)
            count += take

    val_img = torch.cat(val_images, dim=0)

    with torch.no_grad():
        code_val = bergan(val_img, code_only=True).detach()
        b, c, h, w = code_val.shape
        x_val = code_val.view(b, c, -1).permute(0, 2, 1).contiguous()

    # ==========================
    # 4. 评测环境准备
    # ==========================
    eval_snrs = [0, 4, 8, 12, 15]
    base_save_dir = "/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15/eval_snr_0_to_15"
    orig_img_dir = f"{base_save_dir}/original_images_val"
    results_txt_path = f"{base_save_dir}/diffusion_psnr_summary.txt"

    if misc.get_rank() == 0:
        os.makedirs(base_save_dir, exist_ok=True)
        os.makedirs(orig_img_dir, exist_ok=True)
        
        # 保存原图
        for idx in range(num_images_needed):
            torchvision.utils.save_image(torch.clamp(val_img[idx], 0, 1), f"{orig_img_dir}/orig_{idx:04d}.png")
            
        if not os.path.exists(results_txt_path):
            with open(results_txt_path, 'w') as f:
                f.write(f"=== Diffusion Denoising PSNR Summary ===\n")
                f.write("Target_SNR(dB), Denoise_Step_T, Average_PSNR(dB)\n")
                f.write("-" * 55 + "\n")

    # ==========================
    # 5. 特定 SNR 点评估
    # ==========================
    for target_snr in eval_snrs:
        # 计算对应步数
        diff = torch.abs(Eb_N0_dB_value - target_snr)
        noise_t = torch.argmin(diff).item()
        eb_n0_db = Eb_N0_dB_value[noise_t].item()
        eb_n0_db_str = f"{eb_n0_db:.2f}"
        
        if misc.get_rank() == 0:
            print(f"\n[{'='*10} Testing SNR: {target_snr}dB (Mapped Step: {noise_t}, Exact SNR: {eb_n0_db_str}dB) {'='*10}]")
            
        save_img_dir = f"{base_save_dir}/snr_{target_snr}_recon"
        if misc.get_rank() == 0:
            os.makedirs(save_img_dir, exist_ok=True)

        num_batches = (num_images_needed + H.batch_size - 1) // H.batch_size
        global_idx = 0
        total_psnr = 0.0

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * H.batch_size
                end_idx = min((batch_idx + 1) * H.batch_size, num_images_needed)
                
                x_val_batch = x_val[start_idx:end_idx]
                gt_img_batch = val_img[start_idx:end_idx]
                
                # 注入精准噪声
                x_val_noisy_code_t_batch = make_code_noise(
                    x_val_batch, eb_n0_db=eb_n0_db, qam_order=H.qam_order, device=x_val_batch.device
                )

                # 原汁原味的去噪代码调用
                x_val_denoised_batch = get_online_samples_denoise_code(
                    ema_sampler if H.ema else sampler, x_val_noisy_code_t_batch, noise_t
                )
                
                # 解码
                images_batch = get_online_decode_code_into_images(x_val_denoised_batch, bergan, H)

                for idx in range(len(images_batch)):
                    img_gt = torch.clamp(gt_img_batch[idx], 0, 1)
                    img_recon = torch.clamp(images_batch[idx], 0, 1)
                    
                    current_psnr = calculate_psnr(img_gt, img_recon)
                    total_psnr += current_psnr
                    
                    if misc.get_rank() == 0:
                        torchvision.utils.save_image(img_recon, f"{save_img_dir}/recon_{global_idx:04d}.png")
                    global_idx += 1

        processed_tensor = torch.tensor([global_idx], dtype=torch.float32, device='cuda')
        psnr_tensor = torch.tensor([total_psnr], dtype=torch.float32, device='cuda')
        
        if H.distributed:
            dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)
            
        global_processed = int(processed_tensor.item())
        global_psnr = psnr_tensor.item()
        avg_psnr = global_psnr / global_processed if global_processed > 0 else 0.0

        if misc.get_rank() == 0:
            print(f"-> ⭐ Diffusion Denoised PSNR for Target SNR {target_snr}dB: {avg_psnr:.4f} dB")
            with open(results_txt_path, 'a') as f:
                f.write(f"{target_snr:>14}, {noise_t:>14}, {avg_psnr:>16.4f}\n")
                f.flush()

if __name__ == '__main__':
    H = get_sampler_hparams()
    config_log(H.log_dir)
    main(H, None)



# import torch
# import numpy as np
# import copy
# import time
# import os
# import math
# import glob
# from omegaconf import OmegaConf
# from PIL import Image
# import torchvision

# from torch.nn import Module
# from models.binaryae import BinaryAutoEncoder, Generator
# from hparams import get_sampler_hparams
# from utils.data_utils import get_data_loaders
# from utils.sampler_utils import retrieve_autoencoder_components_state_dicts,\
#     get_sampler, get_online_samples,  get_online_samples_with_noisy_code, get_online_samples_denoise_code, get_online_decode_code_into_images
# from utils.train_utils import EMA, NativeScalerWithGradNormCount
# from utils.log_utils import log, log_stats, config_log, start_training_log, \
#     save_stats, load_stats, save_model, load_model, save_images, \
#     MovingAverage
# import misc
# import torch.distributed as dist
# from utils.lr_sched import adjust_lr, lr_scheduler
# from omegaconf import OmegaConf
# from ldm.ldm.util import instantiate_from_config
# from utils.qam_utils import *
# from utils.m_qam_awgn_util import *

# def make_code_noise(clean_code, eb_n0_db, qam_order=16, device=None):
#     received_code = make_code_noise_single(clean_code, eb_n0_db=eb_n0_db, qam_order=qam_order, device=device)
#     return received_code

# def calculate_psnr(img1, img2):
#     mse = torch.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(1.0 / math.sqrt(mse.item()))

# def main(H):
#     misc.init_distributed_mode(H)

#     # 训练时 SNR 对应的采样步映射
#     # Linspace: [15, ..., 0] 对应 64 个步
#     Eb_N0_db_range = torch.tensor([H.snr_range[0], H.snr_range[1]])
#     Eb_N0_dB_value = torch.linspace(Eb_N0_db_range[1], Eb_N0_db_range[0], H.total_steps)   

#     # LDM AE 配置加载
#     ldm_ae_dir = "./ldm/models/ldm/ffhq256"
#     base_configs = sorted(glob.glob(os.path.join(ldm_ae_dir, "config.yaml")))
#     config = OmegaConf.load(base_configs[0])
 
#     # ==========================
#     # 1. 加载 BAE (Encoder & Decoder)
#     # ==========================
#     ae_state_dict = retrieve_autoencoder_components_state_dicts(
#         H, ['encoder', 'quantize', 'generator'], remove_component_from_key=False
#     )
#     bergan = BinaryAutoEncoder(H)
#     bergan.load_state_dict(ae_state_dict, strict=True)
#     bergan = bergan.cuda()
#     bergan.eval()
#     del ae_state_dict

#     # ==========================
#     # 2. 加载 Diffusion Sampler
#     # ==========================
#     sampler = get_sampler(H, None).cuda()

#     if H.ema:
#         ema_sampler = copy.deepcopy(sampler)

#     if H.distributed:
#         sampler = torch.nn.parallel.DistributedDataParallel(sampler, device_ids=[H.gpu], find_unused_parameters=H.guidance)
#         sampler_without_ddp = sampler.module
#     else:
#         sampler_without_ddp = sampler

#     device = next(sampler.parameters()).device

#     if H.load_step > 0:
#         sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir, device=device, allow_mismatch=H.allow_mismatch).cuda()
#         if H.ema:
#             try:
#                 ema_sampler = load_model(ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device, allow_mismatch=H.allow_mismatch)
#                 if misc.get_rank() == 0: print("✅ EMA 权重加载成功！")
#             except Exception:
#                 ema_sampler = copy.deepcopy(sampler_without_ddp)
#                 if misc.get_rank() == 0: print("⚠️ EMA 加载失败，回退到普通权重。")
    
#     sampler.eval()
#     if H.ema: ema_sampler.eval()
#     active_sampler = ema_sampler if H.ema else sampler

#     # ==========================
#     # 3. 数据集准备
#     # ==========================
#     val_dataset = instantiate_from_config(config.data.params.validation)
#     if H.distributed:
#         sampler_val = torch.utils.data.DistributedSampler(val_dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=False)
#     else:
#         sampler_val = torch.utils.data.SequentialSampler(val_dataset)

#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, num_workers=4, sampler=sampler_val, batch_size=H.batch_size, pin_memory=True, drop_last=False
#     )

#     # 提取测试图片 (50张)
#     val_images = []
#     num_images_needed = 50
#     count = 0
#     with torch.no_grad():
#         for val_batch in val_loader:
#             if count >= num_images_needed: break
#             take = min(val_batch["image"].size(0), num_images_needed - count)
#             batch = val_batch["image"][:take].cuda() / 255.0
#             val_images.append(batch.permute(0, 3, 1, 2))
#             count += take

#     val_img = torch.cat(val_images, dim=0)

#     # 统一编码
#     with torch.no_grad():
#         code_val = bergan(val_img, code_only=True).detach()
#         b, c, h, w = code_val.shape
#         x_val = code_val.view(b, c, -1).permute(0, 2, 1).contiguous()

#     # ==========================
#     # 4. 边界点评估环境准备
#     # ==========================
#     # 🌟 修改点：只跑极限点 [-15, 25]
#     eval_snrs = [-15, 25]
#     base_save_dir = "/mnt/data/0/mohao/Project/BLD_DSC/logs/binary_diffusion_ffhq_256_qam16_snr_0_15/eval_snr_0_to_15"
#     orig_img_dir = f"{base_save_dir}/original_images_val"
#     results_txt_path = f"{base_save_dir}/diffusion_psnr_summary.txt"

#     if misc.get_rank() == 0:
#         os.makedirs(base_save_dir, exist_ok=True)
#         os.makedirs(orig_img_dir, exist_ok=True)
#         # 保存原图
#         for idx in range(num_images_needed):
#             torchvision.utils.save_image(torch.clamp(val_img[idx], 0, 1), f"{orig_img_dir}/orig_{idx:04d}.png")
        
#         # 写入汇总表头 (如果是新跑的，追加模式)
#         with open(results_txt_path, 'a') as f:
#             f.write(f"\n--- Extreme SNR Testing (Ours) ---\n")
#             f.write("Target_SNR(dB), Denoise_Step_T, Average_PSNR(dB)\n")

#     # ==========================
#     # 5. 执行推理
#     # ==========================
#     for target_snr in eval_snrs:
#         # 🌟 映射逻辑：
#         # -15dB -> 强制 63 次 (T=63)
#         #  25dB -> 强制 1 次 (T=0)
#         if target_snr == -15:
#             noise_t = 63
#         elif target_snr == 25:
#             noise_t = 0
#         else:
#             # 兼容逻辑
#             diff = torch.abs(Eb_N0_dB_value - target_snr)
#             noise_t = torch.argmin(diff).item()
            
#         if misc.get_rank() == 0:
#             print(f"\n[🚀 Extreme Test] Target SNR: {target_snr}dB -> Steps T: {noise_t}")
            
#         save_img_dir = f"{base_save_dir}/snr_{target_snr}_recon"
#         if misc.get_rank() == 0: os.makedirs(save_img_dir, exist_ok=True)

#         num_batches = (num_images_needed + H.batch_size - 1) // H.batch_size
#         global_idx, total_psnr = 0, 0.0

#         with torch.no_grad():
#             for batch_idx in range(num_batches):
#                 start_idx, end_idx = batch_idx * H.batch_size, min((batch_idx + 1) * H.batch_size, num_images_needed)
#                 x_batch, gt_batch = x_val[start_idx:end_idx], val_img[start_idx:end_idx]
                
#                 # 直接注入当前测试目标的 SNR 噪声 (-15 或 25)
#                 x_noisy_batch = make_code_noise(x_batch, eb_n0_db=target_snr, qam_order=H.qam_order, device=x_batch.device)

#                 # Diffusion 降噪
#                 x_denoised_batch = get_online_samples_denoise_code(active_sampler, x_noisy_batch, noise_t)
                
#                 # 重建
#                 images_batch = get_online_decode_code_into_images(x_denoised_batch, bergan, H)

#                 for idx in range(len(images_batch)):
#                     current_psnr = calculate_psnr(torch.clamp(gt_batch[idx], 0, 1), torch.clamp(images_batch[idx], 0, 1))
#                     total_psnr += current_psnr
#                     if misc.get_rank() == 0:
#                         torchvision.utils.save_image(torch.clamp(images_batch[idx], 0, 1), f"{save_img_dir}/recon_{global_idx:04d}.png")
#                     global_idx += 1

#         # 汇总统计
#         processed_tensor = torch.tensor([global_idx], dtype=torch.float32, device='cuda')
#         psnr_tensor = torch.tensor([total_psnr], dtype=torch.float32, device='cuda')
#         if H.distributed:
#             dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
#             dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)
            
#         avg_psnr = psnr_tensor.item() / processed_tensor.item() if processed_tensor.item() > 0 else 0.0
#         if misc.get_rank() == 0:
#             print(f"-> 🏁 Result for SNR {target_snr}dB: {avg_psnr:.4f} dB")
#             with open(results_txt_path, 'a') as f:
#                 f.write(f"{target_snr:>14}, {noise_t:>14}, {avg_psnr:>16.4f}\n")
#                 f.flush()

# if __name__ == '__main__':
#     H = get_sampler_hparams()
#     config_log(H.log_dir)
#     main(H)