
import sys
sys.path.insert(0, "/home/minimo/Project/BLD_DSC/BinaryLatentDiffusion")
from utils.m_qam_awgn_util import *


# 1) 单点 BER（类似：ber = QAM_AWGN(1000,16,0)）
ber = QAM_AWGN_python(10000, 16, 0)            # 自动用 CUDA（若可用），否则 CPU
print(ber)

# 2) Eb/N0 扫描并画图
simulate_and_plot(M=16, n_bits=100_000, EbN0dB=range(0,22,2))

# 3) 按你的接口：把干净 0/1 张量通过 M-QAM+AWGN 后再硬判决回比特（形状保持）
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clean = torch.randint(0, 2, (100, 100, 10), dtype=torch.int64, device=device)
received = make_code_noise_single(clean, eb_n0_db=15, qam_order=16, device=device)
#计算误码率
bit_errors = torch.sum(clean != received).item()
ber = bit_errors / clean.numel()
print(f"Bit Error Rate (BER): {ber}")
assert received.shape == clean.shape
