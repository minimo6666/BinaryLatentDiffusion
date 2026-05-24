# import os
# import math
# import torch
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import glob
# from omegaconf import OmegaConf

# from torch.nn import Module
# from models.binaryae import BinaryAutoEncoder
# from hparams import get_vqgan_hparams  # 核心修复1：使用 VQGAN 参数解析器避免冲突
# from ldm.ldm.util import instantiate_from_config
# from utils.m_qam_awgn_util import make_code_noise_single
# from utils.sampler_utils import get_online_decode_code_into_images
# import misc
# import torch.distributed as dist

# def make_code_noise(clean_code, eb_n0_db, qam_order=16, device=None):
#     """显式注入信道 AWGN 噪声"""
#     received_code = make_code_noise_single(clean_code, eb_n0_db=eb_n0_db, qam_order=qam_order, device=device)
#     return received_code

# def calculate_psnr(img1, img2):
#     """计算两张图片之间的 PSNR (输入范围应为 [0, 1])"""
#     mse = torch.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(1.0 / math.sqrt(mse.item()))

# def main(H):
#     # 1. 初始化分布式环境
#     misc.init_distributed_mode(H)
    
#     # 2. 加载 LDM 验证集 DataLoader
#     ldm_ae_dir = "./ldm/models/ldm/ffhq256"
#     base_configs = sorted(glob.glob(os.path.join(ldm_ae_dir, "config.yaml")))
#     config = OmegaConf.load(base_configs[0])
 
#     val_dataset = instantiate_from_config(config.data.params.validation)

#     if H.distributed:
#         num_tasks = misc.get_world_size()
#         global_rank = misc.get_rank()
#         sampler_val = torch.utils.data.DistributedSampler(
#             val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
#         )
#     else:
#         sampler_val = torch.utils.data.SequentialSampler(val_dataset)

#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         num_workers=4,
#         sampler=sampler_val,
#         batch_size=H.batch_size,
#         pin_memory=True,
#         drop_last=False,
#     )

#     # 3. 核心修复2：手动精准加载 BinaryAutoEncoder 权重，剥离冗余前缀
#     bergan = BinaryAutoEncoder(H).cuda()
    
#     ckpt_path_ema = os.path.join(H.load_dir, "saved_models", f"binaryae_ema_{H.load_step}.th")
#     ckpt_path_normal = os.path.join(H.load_dir, "saved_models", f"binaryae_{H.load_step}.th")
    
#     if os.path.exists(ckpt_path_ema):
#         ckpt_path = ckpt_path_ema
#         if misc.get_rank() == 0: print(f"✅ Loading EMA Checkpoint: {ckpt_path}")
#     elif os.path.exists(ckpt_path_normal):
#         ckpt_path = ckpt_path_normal
#         if misc.get_rank() == 0: print(f"✅ Loading Normal Checkpoint: {ckpt_path}")
#     else:
#         raise FileNotFoundError(f"❌ 找不到权重文件！尝试路径: {ckpt_path_ema}")

#     # 解包权重并去除 'ae.' 前缀 (因为之前的模型被包在 BinaryGAN 里)
#     state_dict = torch.load(ckpt_path, map_location='cuda')
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith('ae.'):
#             new_state_dict[k[3:]] = v  # 砍掉前 3 个字符 'ae.'
#         elif k.startswith('encoder.') or k.startswith('generator.') or k.startswith('quantize.'):
#             new_state_dict[k] = v
            
#     bergan.load_state_dict(new_state_dict, strict=True)
#     bergan.eval()  # 切至评估模式

#     # 4. 准备保存路径和记录文件
#     num_eval_images = 500
#     save_img_dir = os.path.join(H.log_dir, "recon_images_val")
#     orig_img_dir = os.path.join(H.log_dir, "original_images_val")
#     results_txt_path = os.path.join(H.log_dir, f"psnr_results_snr_{H.snr}.txt")
    
#     if misc.get_rank() == 0:
#         os.makedirs(save_img_dir, exist_ok=True)
#         os.makedirs(orig_img_dir, exist_ok=True)
#         with open(results_txt_path, 'w') as f:
#             f.write(f"Evaluating Encoder-Decoder Under Channel SNR: {H.snr} dB\n")
#             f.write("SNR_Level, Average_PSNR(dB)\n")
#             f.write("-" * 30 + "\n")

#     total_psnr = 0.0
#     processed_count = 0

#     # 5. 开始推理计算
#     with torch.no_grad():
#         for data in tqdm(val_loader, desc=f"Processing Test SNR {H.snr}", disable=misc.get_rank() != 0):
#             if processed_count >= num_eval_images:
#                 break
                
#             img = data["image"].cuda()
#             img = img / 255.0
#             img = img.permute(0, 3, 1, 2)
            
#             # --- 编码 (Encoder) ---
#             code_val = bergan(img, code_only=True).detach()
#             b, c, h, w = code_val.shape
#             x_val = code_val.view(b, c, -1).permute(0, 2, 1).contiguous()
            
#             # --- 信道加噪 (Channel) ---
#             x_val_noisy_code_t = make_code_noise(x_val, eb_n0_db=H.snr, qam_order=H.qam_order, device=x_val.device)
            
#             # --- 解码 (Decoder) ---
#             images_recon = get_online_decode_code_into_images(x_val_noisy_code_t, bergan, H)
#             images_recon = torch.clamp(images_recon, 0.0, 1.0)
            
#             # --- 计算 PSNR 并保存 ---
#             for i in range(img.size(0)):
#                 if processed_count >= num_eval_images:
#                     break
                
#                 current_psnr = calculate_psnr(img[i], images_recon[i])
#                 total_psnr += current_psnr
                
#                 if misc.get_rank() == 0:
#                     x_hat_np = (images_recon[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
#                     Image.fromarray(x_hat_np).save(os.path.join(save_img_dir, f"recon_{processed_count:04d}.png"))
                    
#                     x_np = (img[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
#                     Image.fromarray(x_np).save(os.path.join(orig_img_dir, f"orig_{processed_count:04d}.png"))
                    
#                 processed_count += 1
                
#     # 6. 统计与输出
#     processed_tensor = torch.tensor([processed_count], dtype=torch.float32, device='cuda')
#     psnr_tensor = torch.tensor([total_psnr], dtype=torch.float32, device='cuda')
    
#     if H.distributed:
#         dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
#         dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)
        
#     global_processed = int(processed_tensor.item())
#     global_psnr = psnr_tensor.item()

#     if global_processed > 0:
#         avg_psnr = global_psnr / global_processed
#     else:
#         avg_psnr = 0.0
        
#     if misc.get_rank() == 0:
#         print(f"-> ⭐ Result for Channel SNR {H.snr}: Average PSNR = {avg_psnr:.4f} dB (Images: {global_processed})")
#         with open(results_txt_path, 'a') as f:
#             f.write(f"{H.snr:>10}, {avg_psnr:>16.4f}\n")

# if __name__ == '__main__':
#     H = get_vqgan_hparams()
#     main(H)

import os
import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
from omegaconf import OmegaConf

from torch.nn import Module
from models.binaryae import BinaryAutoEncoder
from hparams import get_vqgan_hparams  
from ldm.ldm.util import instantiate_from_config
from utils.m_qam_awgn_util import make_code_noise_single
from utils.sampler_utils import get_online_decode_code_into_images
import misc
import torch.distributed as dist

def make_code_noise(clean_code, eb_n0_db, qam_order=16, device=None):
    """显式注入信道 AWGN 噪声"""
    return make_code_noise_single(clean_code, eb_n0_db=eb_n0_db, qam_order=qam_order, device=device)

def calculate_psnr(img1, img2):
    """计算两张图片之间的 PSNR (输入范围应为 [0, 1])"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def main(H):
    # 1. 初始化分布式环境
    misc.init_distributed_mode(H)
    
    # 2. 加载 LDM 验证集 DataLoader
    ldm_ae_dir = "./ldm/models/ldm/ffhq256"
    base_configs = sorted(glob.glob(os.path.join(ldm_ae_dir, "config.yaml")))
    config = OmegaConf.load(base_configs[0])
 
    val_dataset = instantiate_from_config(config.data.params.validation)

    if H.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=4,
        sampler=sampler_val,
        batch_size=H.batch_size,
        pin_memory=True,
        drop_last=False,
    )

    # 3. 手动加载 BinaryAutoEncoder，解决各种底层路径和前缀 Bug
    bergan = BinaryAutoEncoder(H).cuda()
    
    ckpt_path_ema = os.path.join(H.load_dir, "saved_models", f"binaryae_ema_{H.load_step}.th")
    ckpt_path_normal = os.path.join(H.load_dir, "saved_models", f"binaryae_{H.load_step}.th")
    
    if os.path.exists(ckpt_path_ema):
        ckpt_path = ckpt_path_ema
        if misc.get_rank() == 0: print(f"✅ Loading EMA Checkpoint: {ckpt_path}")
    elif os.path.exists(ckpt_path_normal):
        ckpt_path = ckpt_path_normal
        if misc.get_rank() == 0: print(f"✅ Loading Normal Checkpoint: {ckpt_path}")
    else:
        raise FileNotFoundError(f"❌ 找不到权重文件！尝试路径: {ckpt_path_ema}")

    # 解包权重并去除 'ae.' 前缀
    state_dict = torch.load(ckpt_path, map_location='cuda')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('ae.'):
            new_state_dict[k[3:]] = v  
        elif k.startswith('encoder.') or k.startswith('generator.') or k.startswith('quantize.'):
            new_state_dict[k] = v
            
    bergan.load_state_dict(new_state_dict, strict=True)
    bergan.eval()  

    # 4. 准备保存路径和记录文件
    num_eval_images = 50 

    # 获取父目录：例如 .../en_de_train_under_snr_-15
    parent_dir = os.path.dirname(H.log_dir)

    # 统一的原图保存目录（所有 test_snr 共享一份）
    orig_img_dir = os.path.join(parent_dir, "original_images_val")
    # 当前 test_snr 的重构图保存目录
    save_img_dir = os.path.join(H.log_dir, "recon_images_val")
    # 汇总的 PSNR 结果文件
    results_txt_path = os.path.join(parent_dir, "psnr_summary.txt")
    
    if misc.get_rank() == 0:
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(orig_img_dir, exist_ok=True)
        # 如果汇总文件不存在，就写个表头
        if not os.path.exists(results_txt_path):
            with open(results_txt_path, 'w') as f:
                f.write(f"=== PSNR Evaluation Summary ===\n")
                f.write("Test_SNR(dB), Average_PSNR(dB)\n")
                f.write("-" * 35 + "\n")

    total_psnr = 0.0
    processed_count = 0

    # 5. 开始推理计算
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Processing Test SNR {H.snr}", disable=misc.get_rank() != 0):
            if processed_count >= num_eval_images:
                break
                
            img = data["image"].cuda()
            img = img / 255.0
            img = img.permute(0, 3, 1, 2)
            
            # --- 编码 (Encoder) ---
            code_val = bergan(img, code_only=True).detach()
            b, c, h, w = code_val.shape
            x_val = code_val.view(b, c, -1).permute(0, 2, 1).contiguous()
            
            # --- 信道加噪 (Channel) ---
            x_val_noisy_code_t = make_code_noise(x_val, eb_n0_db=H.snr, qam_order=H.qam_order, device=x_val.device)
            
            # --- 解码 (Decoder) ---
            images_recon = get_online_decode_code_into_images(x_val_noisy_code_t, bergan, H)
            images_recon = torch.clamp(images_recon, 0.0, 1.0)
            
            # --- 计算 PSNR 并保存图片 ---
            for i in range(img.size(0)):
                if processed_count >= num_eval_images:
                    break
                
                current_psnr = calculate_psnr(img[i], images_recon[i])
                total_psnr += current_psnr
                
                if misc.get_rank() == 0:
                    # 1. 保存当前 SNR 的重构图
                    x_hat_np = (images_recon[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    Image.fromarray(x_hat_np).save(os.path.join(save_img_dir, f"recon_{processed_count:04d}.png"))
                    
                    # 2. 原图防重复保存逻辑：只有当原图不存在时才保存，极大地节省 IO 时间
                    orig_img_path = os.path.join(orig_img_dir, f"orig_{processed_count:04d}.png")
                    if not os.path.exists(orig_img_path):
                        x_np = (img[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                        Image.fromarray(x_np).save(orig_img_path)
                    
                processed_count += 1
                
    # 6. 统计与输出
    processed_tensor = torch.tensor([processed_count], dtype=torch.float32, device='cuda')
    psnr_tensor = torch.tensor([total_psnr], dtype=torch.float32, device='cuda')
    
    if H.distributed:
        dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)
        
    global_processed = int(processed_tensor.item())
    global_psnr = psnr_tensor.item()

    if global_processed > 0:
        avg_psnr = global_psnr / global_processed
    else:
        avg_psnr = 0.0
        
    if misc.get_rank() == 0:
        print(f"-> ⭐ Result for Channel SNR {H.snr}: Average PSNR = {avg_psnr:.4f} dB (Images: {global_processed})")
        # 实时追加写入，并且强行刷入硬盘 (flush) 保证你能立刻在 txt 里看到
        with open(results_txt_path, 'a') as f:
            f.write(f"{H.snr:>12}, {avg_psnr:>16.4f}\n")
            f.flush() 

if __name__ == '__main__':
    H = get_vqgan_hparams()
    main(H)