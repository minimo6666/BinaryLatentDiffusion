
import sys
sys.path.insert(0, "/home/minimo/Project/BLD_DSC/BinaryLatentDiffusion")
import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
import pdb
from torch import nn
from utils.qam_utils import *

from pyphysim.modulators.fundamental import BPSK, QAM, QPSK, Modulator
from pyphysim.simulations import Result, SimulationResults, SimulationRunner

from utils.m_qam_awgn_util import *

# 设置设备
device = "cpu"



clean = torch.randint(0, 2, (100, 100, 10), dtype=torch.int64, device=device)

qam_order = 16
Eb_N0_dB_values = torch.linspace(15, 0,  64)
q_error_t = general_m_qam_ber(Eb_N0_dB_values, qam_order)


# 打开文件准备写入
file_path = "/home/minimo/Project/BLD_DSC/BinaryLatentDiffusion/utils/test/theory_reality_compare.txt"
with open(file_path, "w") as f:
    f.write("Eb_N0_dB\tTheory_BER\tActual_BER\n")

    for idx, eb_n0_db in enumerate(Eb_N0_dB_values):
        # 加噪
        noise_code = make_code_noise_single(clean, eb_n0_db=eb_n0_db, qam_order=qam_order, device=device)
        
        # 计算BER
        bit_errors = torch.sum(clean != noise_code).item()
        actual_ber = bit_errors / clean.numel()
        # 获取理论BER
        theory_ber = q_error_t[idx]
        
        # 写入文件
        f.write(f"{eb_n0_db:.4f}\t{theory_ber:.6f}\t{actual_ber:.6f}\n")
        
        # 打印进度
        print(f"Eb/N0: {eb_n0_db:.2f} dB, Theory BER: {theory_ber:.6f}, Actual BER: {actual_ber:.6f}")

print(f"Results saved to {file_path}")