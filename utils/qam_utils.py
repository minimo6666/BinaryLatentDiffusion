import torch
import math
import numpy as np
import torch
from scipy.special import erfcinv

from scipy.stats import norm

#Convert the QAM symbol to its binary representation. Real value cordinate->binary 
#For example.(1.5,3.5)-> 01 11

def symbol2binary(x_s, qam_order):
    grid_num = math.sqrt(qam_order)
    # 计算每个范围的间距
    segment_width = 4 / grid_num
    # 计算当前的x_s属于第几个范围
    x_int = (x_s // segment_width).long()

    bits_num = int(math.log2(grid_num))

    # 使用广播和位运算替换循环
    # 生成一个二进制掩码，然后利用广播机制和位运算计算二进制表示
    masks = 2**torch.arange(bits_num-1, -1, -1, device=x_s.device).reshape(1, -1)
    x_binary = ((x_int.unsqueeze(-1) & masks) > 0).float()  # 利用位运算和广播
    x_binary = x_binary.view(*x_int.shape[:-1], -1)  # 重塑结果以匹配期望的输出形状

    return x_binary

# 建议把名字改成 SNRdB，或者在内部进行 Eb/N0 到 SNR 的转换
def SNR_dB_to_sigma_in_M_QAM(SNR_dB, qam_order):
    """
    Calculate the standard deviation (sigma) per dimension for AWGN noise addition.
    """
    # 1. 计算未归一化 M-QAM 的平均符号功率
    P = 2.0 * (qam_order - 1) / 3.0
    
    # 2. 将 SNR(dB) 转换为线性值
    SNR_linear = 10.0 ** (SNR_dB / 10.0)
    
    # 3. 计算单维度的标准差 sigma
    # 满足: 信号功率 / (2 * sigma^2) = SNR_linear
    sigma = torch.sqrt(torch.tensor(P / (2.0 * SNR_linear)))
    
    return sigma


def binary2symbol(x_b, qam_order):
    bits_num = int(math.log2(math.sqrt(qam_order)))
    grid_num = math.sqrt(qam_order)

    # 计算每个范围的间距
    segment_width = 4 / grid_num

    # 使用广播和位运算来重构符号
    powers_of_two = 2**torch.arange(bits_num-1, -1, -1, device=x_b.device)
    # 将x_b重新整形成(..., N/bits_num, bits_num)，以便每个符号的二进制表示在最后一个维度
    x_b_reshaped = x_b.view(*x_b.shape[:-1], -1, bits_num)
    # 利用广播机制计算每组bits的十进制值
    x_symbol = torch.sum(x_b_reshaped * powers_of_two, dim=-1)

    # 调整到正确的数值范围
    x_symbol = x_symbol * segment_width + (segment_width / 2.0)

    return x_symbol

#TODO Gray coding: symbol2binary_gray_code binary2symbol_gray_code

def bpsk_symbol2binary(x_s):
   #TODO 小于0.5的为0，大于0.5的为1
   return (x_s > 0.5).float()

#x是16,256,64 y是16,1,1
def g_function_symbol(x, y):
#如果x和y的轴数不一样
  if x.dim() != y.dim():
     y = y.unsqueeze(-1).unsqueeze(-1)
  return x*(1-y) + (1-x)*y

def g_function_binary(x, y):
  if x.dim() != y.dim():
     y = y.unsqueeze(-1).unsqueeze(-1)
  return x*y + (1-x)*(1-y)

eps = 1e-30

def kl_div(q, p):
    '''KL Divergence of two multivariate Bernoulli distributions'''
    q = torch.clip(q, min=1e-10, max=1-(1e-7))
    p = torch.clip(p, min=1e-10, max=1-(1e-7))
    return torch.sum((q * torch.log2((q/p) + eps)) + ((1.0-q) * torch.log2(((1.0-q)/(1.0-p)) + eps)), dim=(1,2))

def propabilitynoisesymbol(noise_symbol,qam_order):
# Determine the device of the input tensor
    device = noise_symbol.device

    # 定义段长和中心点间距
    segment_length = 4 / math.sqrt(qam_order)
    center_point_distance = segment_length

    # 计算tensor中每个元素所在的段索引
    segment_indices = (noise_symbol // segment_length).int()

    # 计算左右中心点的位置
    left_centers = segment_length * segment_indices + center_point_distance / 2
    right_centers = left_centers + center_point_distance

    # 计算到左右中心点的距离
    distances_to_left_centers = torch.abs(noise_symbol - left_centers)
    distances_to_right_centers = torch.abs(noise_symbol - right_centers)

    # 计算概率（距离的倒数作为权重，然后归一化）
    total_inverse_distances = 1 / distances_to_left_centers + 1 / distances_to_right_centers
    probs_left = (1 / distances_to_left_centers) / total_inverse_distances
    probs_right = (1 / distances_to_right_centers) / total_inverse_distances

    # Ensure the random tensor is generated on the same device
    randoms = torch.rand(noise_symbol.shape, device=device)
    decisions = randoms < probs_left

    # 根据决策替换tensor的值
    replaced_tensor = torch.where(decisions, left_centers, right_centers)

    return replaced_tensor


# QAM 16 64 256
P_signal = {
  16: 2.5,
  64: 2.625,
  256: 2.65625
}

def SNR(sigma,qam_order):
    p_signal = P_signal.get(qam_order, 2.5)

    P_noise = sigma**2

    # Calculate SNR in linear scale for the new scenario
    SNR_linear = p_signal / P_noise

    # Convert SNR to dB for the new scenario
    return SNR_linear

def SNR_linear2SNRdb(SNR_linear):
    # Convert SNR to dB for the new scenario
    return 10 * torch.log10(SNR_linear)

def SNR_dB2SNR_linear(SNRdB):
  return 10**(SNRdB/10)

def SNR_Linear2Sigma(SNR_linear,qam_order):
  p_signal = P_signal.get(qam_order, 2.5)
  return torch.sqrt(p_signal/SNR_linear)

def SNR_dB2sigma(SNRdB,qam_order):
  return SNR_Linear2Sigma(10**(SNRdB/10),qam_order)

def sigma2SNRdB(sigma,qam_order):
    P_noise = sigma**2
    # Calculate SNR in linear scale for the new scenario
    SNR_linear = P_signal.get(qam_order, 2.5)/ P_noise

    return 10 * torch.log10(SNR_linear)

def sigma2SNRdB_BPSK(sigma):
    P_noise = sigma**2
    # Calculate SNR in linear scale for the new scenario
    P_signal = 0.5**2
    SNR_linear = P_signal/ P_noise

    return 10 * math.log10(SNR_linear)


#N0=2sigma^2 sqrt(Eb)=0.5
def q_error_t(variance, Eb=0.25):
  N0_t = 2*variance
  return 0.5*torch.erfc(torch.sqrt(Eb/N0_t))

def find_sigma(Q_value, z_value):
    # 计算正态分布的CDF值，即Phi((z/sigma) = 0.7)

    if isinstance(Q_value, torch.Tensor):  # Check if Q_value is a tensor
        Q_value = Q_value.cpu().numpy() 

    phi_value = 1 - Q_value  # 1 - 0.3 = 0.7
    
    # 计算 z/sigma
    z_over_sigma = norm.ppf(phi_value)  # norm.ppf 计算正态分布的逆CDF
    
    # 解出 sigma
    sigma = z_value / z_over_sigma
    return sigma

def error_rate_2_snr_db(error_rate,sqrt_p_signal):
  #here error_rate=Q(sqrt_p_signal)
  sigma_value = find_sigma(error_rate, sqrt_p_signal)
  snr_db = sigma2SNRdB_BPSK(sigma_value)
  return snr_db

def ber_to_snr_linear(P_b):
    # 检查输入是否是PyTorch张量
    if isinstance(P_b, torch.Tensor):
        P_b = P_b.cpu().numpy()  # 将PyTorch张量转换为NumPy数组
    
    # 使用erfcinv函数计算(Eb/N0)
    E_b_N_0 = (erfcinv(2 * P_b)) ** 2
    
    # 将结果转换回PyTorch张量
    return E_b_N_0

def linear_to_DB(snr_linear):
    return 10 * math.log10(snr_linear)

def EbNoDB_toEbNo(EbNoDB):
   return 10 ** (EbNoDB / 10)

#paper: https://faculty.kfupm.edu.sa/ee/naffouri/courses/ee242%20material/Projects/Ronell%20B%20Sicat.pdf
def Eb_No_dB_to_sigma(Eb_No_dB):    #(maby some error in this formula)
    
    d = torch.tensor(1)

    M = torch.tensor(16)

    Eb = torch.tensor(2.5)
      
    Eb_N0_linear = 10 ** (Eb_No_dB / 10)

    N0 = Eb / (10 ** (Eb_No_dB / 10))

    sigma = torch.sqrt(N0 / 2)

    return sigma

#https://dsplog.com/2007/09/23/scaling-factor-in-qam/
def Eb_No_dB_to_sigma_in_M_QAM(Eb_No_dB, qam_order):

        """
        Calculate the standard deviation (sigma) for AWGN noise addition
        given SNR in decibels (SNRdB) and QAM order (M).

        Parameters:
        SNRdB (float): Signal-to-Noise Ratio in decibels.
        M (int): QAM order.

        Returns:
        torch.Tensor: Standard deviation sigma.
        """
        # Calculate signal power for M-QAM
        P = 2.0 * (qam_order - 1) / 3.0
        
        # Convert SNRdB to linear scale
        SNR = 10.0 ** (Eb_No_dB / 10.0)
        
        # Calculate sigma
        sigma = torch.sqrt(P / (2.0 * SNR))
        
        return sigma


def Eb_No_dB_to_sigma_bpsk(SNRdB):
    """
    Calculate the standard deviation (sigma) for AWGN noise addition
    given SNR in decibels (SNRdB) for BPSK.

    Parameters:
    SNRdB (torch.Tensor or float): SNR in decibels. Can be a tensor or a single value.

    Returns:
    torch.Tensor: Sigma corresponding to each SNRdB value.
    """
    # Convert SNRdB to linear scale
    SNR = torch.pow(10.0, SNRdB / 10.0)
    
    # Calculate sigma
    sigma = 1.0 / torch.sqrt(2.0 * SNR)
    
    return sigma

mapping = {
        '0000': (1, 1),
        '0001': (1, 3),
        '0010': (3, 1),
        '0011': (3, 3),
        '0100': (1, -1),
        '0101': (1, -3),
        '0110': (3, -1),
        '0111': (3, -3),
        '1000': (-1, 1),
        '1001': (-1, 3),
        '1010': (-3, 1),
        '1011': (-3, 3),
        '1100': (-1, -1),
        '1101': (-1, -3),
        '1110': (-3, -1),
        '1111': (-3, -3)
        }

def qam_gray_encoding(binary_tensor, M):
    """
    QAM调制的格雷编码
    
    参数:
        binary_tensor (torch.Tensor): 二进制张量，值为0或1
        M (int): QAM调制阶数，如16, 64, 256等
    
    返回:
        torch.Tensor: 格雷编码后的二进制张量
    """
    # 计算每维度的位数
    m = int(math.isqrt(M))
    
    # 确保输入位数是m的倍数
    total_bits = binary_tensor.numel()
    assert total_bits % m == 0, f"输入位数({total_bits})必须是{m}(sqrt({M}))的倍数"
    
    # 重塑为(batch_size, m)的形状
    batch_size = total_bits // m
    binary_reshaped = binary_tensor.view(batch_size, m)
    
    # 将二进制转换为整数
    powers_of_2 = 2 ** torch.arange(m-1, -1, -1, device=binary_tensor.device)
    integers = torch.sum(binary_reshaped * powers_of_2, dim=1).int()
    
    # 将整数转换为格雷码
    gray_integers = torch.bitwise_xor(integers, torch.bitwise_right_shift(integers, 1))
    
    # 将格雷码整数转换回二进制表示
    gray_binary = torch.zeros((batch_size, m), device=binary_tensor.device, dtype=torch.int32)
    for i in range(m):
        gray_binary[:, i] = torch.bitwise_and(torch.bitwise_right_shift(gray_integers, m-1-i), 1)
    
    # 恢复原始形状
    return gray_binary.view_as(binary_tensor)

def qam_gray_decoding(gray_tensor, M):
    """
    QAM调制的格雷解码
    
    参数:
        gray_tensor (torch.Tensor): 格雷编码的二进制张量，值为0或1
        M (int): QAM调制阶数，如16, 64, 256等
    
    返回:
        torch.Tensor: 解码后的原始二进制张量
    """
    # 计算每维度的位数
    m = int(math.isqrt(M))
    
    # 确保输入位数是m的倍数
    total_bits = gray_tensor.numel()
    assert total_bits % m == 0, f"输入位数({total_bits})必须是{m}(sqrt({M}))的倍数"
    
    # 重塑为(batch_size, m)的形状
    batch_size = total_bits // m
    gray_reshaped = gray_tensor.view(batch_size, m)
    
    # 将格雷编码二进制转换为整数
    powers_of_2 = 2 ** torch.arange(m-1, -1, -1, device=gray_tensor.device)
    gray_integers = torch.sum(gray_reshaped * powers_of_2, dim=1).int()
    
    # 使用位操作技巧进行格雷码解码
    binary_integers = gray_integers.clone()
    shift = 1
    while shift < m:
        binary_integers = torch.bitwise_xor(binary_integers, torch.bitwise_right_shift(binary_integers, shift))
        shift <<= 1  # 移位量加倍
    
    # 将自然二进制整数转换回二进制表示
    binary_tensor = torch.zeros((batch_size, m), device=gray_tensor.device, dtype=torch.int32)
    for i in range(m):
        binary_tensor[:, i] = torch.bitwise_and(torch.bitwise_right_shift(binary_integers, m-1-i), 1)
    
    # 恢复原始形状
    return binary_tensor.view_as(gray_tensor)



########################genneral gray coding mapping and demapping M-QAM####################
#from blog https://blog.csdn.net/Null_0_lluN/article/details/129463237
def qam_constellation_with_gray_code(M, normalize=False):
    assert torch.log2(torch.tensor(M)).int().eq(torch.log2(torch.tensor(M))).all()
    m = int(torch.sqrt(torch.tensor(M)))
    x = torch.zeros(m, dtype=torch.int32)
    y = torch.zeros(m, dtype=torch.int32)
    
    def natural2gray(x_tensor):
        return torch.bitwise_xor(x_tensor, torch.bitwise_right_shift(x_tensor, 1))
    
    gray_order = natural2gray(torch.arange(0, m, dtype=torch.int32))
    x[gray_order] = torch.arange(0, 2*m, 2, dtype=torch.int32) - m + 1
    y[gray_order] = torch.arange(0, 2*m, 2, dtype=torch.int32) - m + 1
    
    constellation = torch.zeros((m, m), dtype=torch.complex64)
    for i in range(m):
        for j in range(m):
            # constellation[i, j] = torch.complex(torch.tensor(x[i], dtype=torch.float32), torch.tensor(y[j], dtype=torch.float32))
            constellation[i, j] = torch.complex(x[i].clone().detach().to(torch.float32), y[j].clone().detach().to(torch.float32))
    
    if normalize:
        norm = torch.norm(constellation) / m
        constellation = constellation / norm
    return constellation.view(-1)

def qam_constellation(M, normalize=False):
    # 确保M是平方数
    assert torch.log2(torch.tensor(M)).int().eq(torch.log2(torch.tensor(M))).all()
    m = int(torch.sqrt(torch.tensor(M)))
    
    # 直接生成实部和虚部的坐标值（自然顺序）
    x = torch.arange(0, 2*m, 2, dtype=torch.float32) - m + 1
    y = torch.arange(0, 2*m, 2, dtype=torch.float32) - m + 1
    
    # 创建星座点网格
    real_part = x.repeat(m, 1)
    imag_part = y.repeat(m, 1).t()
    
    # 组合实部和虚部形成复数星座点
    constellation = torch.complex(real_part, imag_part)
    
    if normalize:
        # 计算归一化因子（使平均功率为1）
        norm = torch.norm(constellation) / m
        constellation = constellation / norm
    
    return constellation.view(-1)

def mapping(data, constellation, qam_order):
    M = qam_order
    k = int(torch.log2(torch.tensor(M)))
    assert data.shape[0] % k == 0
    data = data.view(-1, k)
    mask = torch.tensor([2**i for i in range(k-1, -1, -1)], dtype=torch.int32).to(data.device)
    index = torch.sum(data * mask, dim=1).int()
    return constellation[index]

def demapping(qam_symbols, constellation, qam_order):
    M = qam_order
    k = int(torch.log2(torch.tensor(M, dtype=torch.int32)))
    constellation = constellation.view(1, -1)
    qam_symbols = qam_symbols.view(-1, 1)
    
    distance = torch.abs(qam_symbols - constellation)**2
    indices = torch.argmin(distance, dim=1)
    
    # Generate masks for bit extraction
    masks = 2 ** torch.arange(k-1, -1, -1, dtype=indices.dtype, device=indices.device)
    # Extract bits
    bits = (indices.unsqueeze(-1) & masks.unsqueeze(0)) != 0
    binary_data = bits.to(torch.int32).flatten()
    
    return binary_data

########################genneral gray coding mapping and demapping M-QAM####################

def binary_code_to_qam16_symbol_gray_code(binary_tensor):
        # 16QAM灰度编码映射表（使用二进制字符串作为键）
 
    # 确保输入的长度是4的倍数
    if binary_tensor.size(-1) % 4 != 0:
        raise ValueError("输入的二进制tensor长度必须是4的倍数")
    
    # 初始化空列表来存储展开后的数据
    output = []

   
    # 获取输入的形状信息
    batch_shape = binary_tensor.shape[:-1]  # 保留除了最后一维之外的所有维度
    num_elements = binary_tensor.shape[-1]  # 最后一维的元素数量

    # 将输入的最后一维分成4个比特一组
    binary_tensor = binary_tensor.reshape(-1, 4)  # 这里将最后一维重新展开成4个比特一组

    # 初始化输出张量，大小为原始输入形状的最后一维/2
    output = []

    # 处理每4个二进制位
    for bits in binary_tensor:
        # 获取当前4位的二进制字符串
        bits_str = ''.join(str(int(bit)) for bit in bits)
        
        # 查找对应的 (x, y) 坐标
        x, y = mapping[bits_str]
        
        # 将 x 和 y 交替添加到结果列表
        output.append(x)
        output.append(y)

    # 将输出的列表转换为张量，并恢复前几维
    output_tensor = torch.tensor(output).reshape(*batch_shape, -1)  # 最后一维的长度是原来的二进制位长度 / 2

    return output_tensor
   

def qam16_symbol_to_binary_code_gray_code(tensor):
        # 获取 tensor 的形状
    *dims, last_dim = tensor.shape
    # 初始化一个空的结果 tensor，形状为 (*dims, last_dim * 2)
    gray_code_tensor = torch.zeros((*dims, last_dim * 2), dtype=torch.int64)

    # 使用 numpy.ndindex 动态生成索引
    for idx in np.ndindex(*dims):  # 遍历所有前面维度的组合
        # 遍历最后一维中的每两个元素
        for i in range(0, last_dim, 2):
            # 获取当前的 x 和 y 值
            x, y = tensor[idx][i], tensor[idx][i + 1]

            # 对 (x, y) 进行 Gray code 转换
            gray_code_tensor[idx][i*2:i*2+2] = torch.tensor([
                1 if x < 0 else 0,   # 第一位: x < 0 -> 1 else 0
                1 if y < 0 else 0    # 第二位: y < 0 -> 1 else 0
            ], dtype=torch.int64)

            # 这里你可以根据实际需求继续添加更多的转换规则
            # 比如第三位和第四位的转换，可以根据 x 和 y 的值进一步调整
            gray_code_tensor[idx][i*2+2:i*2+4] = torch.tensor([
                0 if -2 <= x < 2 else 1,  # 第三位: -2 <= x < 2 -> 0 else 1
                0 if -2 <= y < 2 else 1   # 第四位: -2 <= y < 2 -> 0 else 1
            ], dtype=torch.int64)

    return gray_code_tensor

#paper: https://faculty.kfupm.edu.sa/ee/naffouri/courses/ee242%20material/Projects/Ronell%20B%20Sicat.pdf  formula (74)(75) their was an error in formula (74), should be + instead of - in torch.floor(j*(2**(k-1))/torch.sqrt(M)+0.5)
#we should use https://ieeexplore.ieee.org/abstract/document/883298 formula (27)
def general_m_qam_ber(Eb_N0_dB, M):
    M = torch.tensor(M)
    Eb_N0 = EbNoDB_toEbNo(Eb_N0_dB)
    r = Eb_N0
    k = torch.log2(torch.sqrt(M))
    p_b = 0.0
    p_b_k = 0.0
    for k in range(1, (torch.log2(torch.sqrt(M))).int() + 1):
        for j in range(0, int((1 - 2 ** (-k)) * torch.sqrt(M))):
            p_b_k += 1/(torch.sqrt(M))*((-1)**torch.floor(j*(2**(k-1))/torch.sqrt(M))*(2**(k-1)-torch.floor(j*(2**(k-1))/torch.sqrt(M)+0.5))*(torch.erfc((2*j+1)*torch.sqrt(3*(torch.log2(M))*r/(2*(M-1)))))) 
    p_b = (1.0 / torch.log2(torch.sqrt(M))) * p_b_k
    return p_b  

        
    
#The reverse process of above
# def binary2symbol(x_b,qam_order):

#     # Convert each element of x_b to its binary representation

#     # bits_num is only real part bits or imag part bits
#     bits_num = int(math.sqrt(qam_order) * 0.5)

#     #binary to decimal
#     x_symbol = torch.zeros(*x_b.shape[:-1], x_b.size(-1) // bits_num, device=x_b.device)
#     for i in range(bits_num):
#         x_symbol += x_b[..., i::bits_num] * 2**(bits_num-i-1)

#     #to qam Constellations center
#     x_symbol += 0.5

#     return x_symbol

# def symbol2binary(x_s,qam_order):

#     grid_num = math.sqrt(qam_order)
#     # symbol I_symbol Q_symbol -> I_grid Q_grid
#     x_int = torch.floor(x_s).clamp(0, grid_num)
#     bits_num = int(math.log2(grid_num))
    
#     # Convert each element of x_int to its binary representation with bits_num bits
#     x_binary = torch.zeros(*x_int.shape[:-1], x_int.size(-1) * bits_num, device=x_int.device)

#     for i in range(bits_num):
#         x_binary[..., i::bits_num] = (x_int // 2**(bits_num-i-1)) % 2
    
#     return x_binary

# def propabilitynoisesymbol(noise_symbols, qam_order):
#     max_symbol_location = math.sqrt(qam_order) - 0.5
#     #对noise_symbol的每一个元素做这样的判定，当元素的值小于0.5时，或者大于max_symbol_location时，保持该元素不变
#     noise_symbols = torch.where((noise_symbols < 0.5) | (noise_symbols > max_symbol_location), noise_symbols, noise_symbols)

#     #对noise_symbol的每一个元素做这样的判定，当元素的值大于0.5时，或者小于max_symbol_location时，从0.5到max_symbol_location中间划分出（math.sqrt(qam_order)-1）段空间，
#     #每一段空间的长度为1，先确定当前的元素落在哪一段空间中，设当前空间的两个端点为 s_small,s_large，设该元素当前的值为s, 现在这个元素以 (s-s_small)的概率设置为s_large,
#     #(1-(s-s_small))的概率设置为s_small

#     # 计算段数
#     segments = int(math.sqrt(qam_order) - 1)

#     # 从0.5到max_symbol_location之间的段落位置
#     segment_starts = torch.arange(0.5, max_symbol_location, 1, device=noise_symbols.device)

#     # 确定每个元素所在的段落
#     indices = torch.clamp((noise_symbols - 0.5).floor(), 0, segments - 1).long()

#     # 计算s_small和s_large
#     s_small = segment_starts[indices]
#     s_large = s_small + 1

#     # 计算概率
#     prob = noise_symbols - s_small

#     # 生成随机数，决定是否更新为s_large或保持为s_small
#     random_values = torch.rand_like(noise_symbols)
#     updated_symbols = torch.where(random_values < prob, s_large, s_small)

#     # 应用条件过滤，只有当元素值在0.5到max_symbol_location之间时才更新
#     condition = (noise_symbols > 0.5) & (noise_symbols < max_symbol_location)
#     noise_symbols = torch.where(condition, updated_symbols, noise_symbols)
#     return noise_symbols



  