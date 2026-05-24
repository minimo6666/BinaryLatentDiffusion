# QAM over AWGN in Python (PyTorch w/ CUDA support; NumPy fallback).
# - Provides a drop-in Python port of the MATLAB QAM_AWGN function.
# - Also provides make_code_noise_single(clean_code, eb_n0_db, qam_order=16, device=device)
#   which returns noisy, hard-decoded bits with the SAME SHAPE as the input.
# - Includes a demo to sweep Eb/N0 and plot BER.
#
# If PyTorch with CUDA is available, computations run on the given device.
# Otherwise, the code falls back to NumPy on CPU automatically.

import math
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# ----------------------------- Common Utilities -----------------------------
def _as_numpy_bits(x):
    """Ensure input is a flat 0/1 NumPy int64 array."""
    x = np.asarray(x)
    # Convert bool/float to int bits safely
    if x.dtype != np.int64:
        x = (x > 0.5).astype(np.int64) if np.issubdtype(x.dtype, np.floating) else x.astype(np.int64)
    return x.reshape(-1)

def _qfunc_np(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))

# ----------------------------- NumPy Implementation -----------------------------
def gray_encode_np(n):
    return np.bitwise_xor(n, n >> 1)

def gray_decode_np(g):
    # iterative Gray -> binary
    b = np.array(g, copy=True)
    shift = 1
    while shift < (g.dtype.itemsize * 8):
        b ^= (b >> shift)
        shift <<= 1
    return b

def int_to_bits_np(n, width):
    # MSB-first bits of n with given width
    n = np.asarray(n, dtype=np.int64)
    return np.stack([((n >> (width - 1 - i)) & 1) for i in range(width)], axis=-1).astype(np.int64)

def bits_to_int_np(bits):
    bits = np.asarray(bits, dtype=np.int64)
    width = bits.shape[-1]
    weights = (1 << np.arange(width)[::-1]).astype(np.int64)
    return (bits * weights).sum(axis=-1).astype(np.int64)

def qam_mod_np(bits, M):
    m = int(round(math.sqrt(M)))
    if m * m != M:
        raise ValueError("M must be a perfect square (e.g., 4, 16, 64).")
    k = int(round(math.log2(M)))
    k2 = k // 2
    s = 1.0 / math.sqrt((2.0 / 3.0) * (m**2 - 1))  # Unit average power
    
    bits = _as_numpy_bits(bits)
    if bits.size % k != 0:
        raise ValueError(f"bits length must be a multiple of log2(M)={k}.")
    symbols = bits.reshape(-1, k)
    
    i_gray = bits_to_int_np(symbols[:, :k2])
    q_gray = bits_to_int_np(symbols[:, k2:])
    i_nat = gray_decode_np(i_gray)
    q_nat = gray_decode_np(q_gray)
    
    levels = np.arange(-(m - 1), m, 2)  # -m+1, -m+3, ..., m-1
    i_amp = levels[i_nat]
    q_amp = levels[q_nat]
    
    return s * (i_amp.astype(np.float64) + 1j * q_amp.astype(np.float64))

def qam_demod_np(sym, M):
    m = int(round(math.sqrt(M)))
    if m * m != M:
        raise ValueError("M must be a perfect square (e.g., 4, 16, 64).")
    k = int(round(math.log2(M)))
    k2 = k // 2
    s = 1.0 / math.sqrt((2.0 / 3.0) * (m**2 - 1))
    invs = 1.0 / s
    levels = np.arange(-(m - 1), m, 2)  # natural order
    
    # Scale back then nearest-level decision
    rI = (sym.real * invs)
    rQ = (sym.imag * invs)
    i_idx = np.round((rI + (m - 1)) / 2.0).astype(np.int64)
    q_idx = np.round((rQ + (m - 1)) / 2.0).astype(np.int64)
    i_idx = np.clip(i_idx, 0, m - 1)
    q_idx = np.clip(q_idx, 0, m - 1)
    
    i_gray = gray_encode_np(i_idx)
    q_gray = gray_encode_np(q_idx)
    
    i_bits = int_to_bits_np(i_gray, k2)
    q_bits = int_to_bits_np(q_gray, k2)
    bits = np.concatenate([i_bits, q_bits], axis=-1)
    return bits.reshape(-1)

def add_awgn_np(sym, eb_n0_db, M):
    k = int(round(math.log2(M)))
    eb_n0 = 10.0 ** (eb_n0_db / 10.0)
    Eb = 1.0 / k  # since Es=1
    sigma2 = Eb / (2.0 * eb_n0)
    sigma = math.sqrt(sigma2)
    noise = sigma * (np.random.randn(*sym.shape) + 1j * np.random.randn(*sym.shape))
    return sym + noise

def qam_awgn_bits_np(bits, M, eb_n0_db):
    bits = _as_numpy_bits(bits)
    k = int(round(math.log2(M)))
    pad = (-bits.size) % k
    if pad:
        bits_padded = np.concatenate([bits, np.zeros(pad, dtype=np.int64)], axis=0)
    else:
        bits_padded = bits
    tx = qam_mod_np(bits_padded, M)
    rx = add_awgn_np(tx, eb_n0_db, M)
    z = qam_demod_np(rx, M)
    return z[:bits.size]

# ----------------------------- PyTorch Implementation -----------------------------
if TORCH_AVAILABLE:
    def _as_torch_bits(x, device=None):
        t = torch.as_tensor(x, device=device)
        if t.dtype.is_floating_point:
            t = (t > 0.5).to(torch.int64)
        else:
            t = t.to(torch.int64)
        return t.reshape(-1)

    def gray_encode_torch(n):
        return n ^ (n >> 1)

    def gray_decode_torch(g):
        b = g.clone()
        shift = 1
        # Determine up to needed bit-width from the max value
        max_bits = max(1, int(torch.ceil(torch.log2(b.max().float() + 1)).item())) if b.numel() > 0 else 1
        while shift < (1 << math.ceil(math.log2(max_bits+1))):
            b = b ^ (b >> shift)
            shift <<= 1
        return b

    def int_to_bits_torch(n, width):
        n = n.to(torch.int64)
        bits = [(n >> (width - 1 - i)) & 1 for i in range(width)]
        return torch.stack(bits, dim=-1).to(torch.int64)

    def bits_to_int_torch(bits):
        bits = bits.to(torch.int64)
        width = bits.shape[-1]
        weights = (2 ** torch.arange(width - 1, -1, -1, device=bits.device)).to(torch.int64)
        return (bits * weights).sum(dim=-1).to(torch.int64)

    def qam_mod_torch(bits, M, device=None):
        m = int(round(math.sqrt(M)))
        if m * m != M:
            raise ValueError("M must be a perfect square (e.g., 4, 16, 64).")
        k = int(round(math.log2(M)))
        k2 = k // 2
        s = 1.0 / math.sqrt((2.0 / 3.0) * (m**2 - 1))
        bits = _as_torch_bits(bits, device=device)
        if bits.numel() % k != 0:
            raise ValueError(f"bits length must be a multiple of log2(M)={k}.")
        symbols = bits.reshape(-1, k)

        i_gray = bits_to_int_torch(symbols[:, :k2])
        q_gray = bits_to_int_torch(symbols[:, k2:])
        i_nat = gray_decode_torch(i_gray)
        q_nat = gray_decode_torch(q_gray)

        levels = torch.arange(-(m - 1), m, 2, device=device, dtype=torch.int64)
        i_amp = levels[i_nat]
        q_amp = levels[q_nat]

        real = (s * i_amp.to(torch.float64))
        imag = (s * q_amp.to(torch.float64))
        return torch.complex(real, imag)

    def qam_demod_torch(sym, M):
        m = int(round(math.sqrt(M)))
        if m * m != M:
            raise ValueError("M must be a perfect square (e.g., 4, 16, 64).")
        k = int(round(math.log2(M)))
        k2 = k // 2
        s = 1.0 / math.sqrt((2.0 / 3.0) * (m**2 - 1))
        invs = 1.0 / s

        rI = (sym.real * invs)
        rQ = (sym.imag * invs)
        i_idx = torch.round((rI + (m - 1)) / 2.0).to(torch.int64)
        q_idx = torch.round((rQ + (m - 1)) / 2.0).to(torch.int64)
        i_idx = torch.clamp(i_idx, 0, m - 1)
        q_idx = torch.clamp(q_idx, 0, m - 1)

        i_gray = gray_encode_torch(i_idx)
        q_gray = gray_encode_torch(q_idx)

        i_bits = int_to_bits_torch(i_gray, k2)
        q_bits = int_to_bits_torch(q_gray, k2)
        bits = torch.cat([i_bits, q_bits], dim=-1).reshape(-1)
        return bits

    def add_awgn_torch(sym, eb_n0_db, M):
        k = int(round(math.log2(M)))
        eb_n0 = 10.0 ** (eb_n0_db / 10.0)
        Eb = 1.0 / k  # Es=1
        sigma2 = Eb / (2.0 * eb_n0)
        sigma = math.sqrt(sigma2)
        noise_real = torch.randn_like(sym.real) * sigma
        noise_imag = torch.randn_like(sym.imag) * sigma
        noise = torch.complex(noise_real, noise_imag)
        return sym + noise

    def qam_awgn_bits_torch(bits, M, eb_n0_db, device=None):
        bits = _as_torch_bits(bits, device=device)
        k = int(round(math.log2(M)))
        pad = (-bits.numel()) % k
        if pad:
            bits_padded = torch.cat([bits, torch.zeros(pad, dtype=torch.int64, device=device)], dim=0)
        else:
            bits_padded = bits
        tx = qam_mod_torch(bits_padded, M, device=device)
        rx = add_awgn_torch(tx, eb_n0_db, M)
        z = qam_demod_torch(rx, M)
        return z[:bits.numel()]

# ----------------------------- Public API -----------------------------

def QAM_AWGN_python(n_bits, M, EbN0dB, device=None):
    """
    Python port of QAM_AWGN: simulate BER for Gray-coded M-QAM over AWGN.
    Uses PyTorch with CUDA if available; otherwise falls back to NumPy.
    """
    if TORCH_AVAILABLE:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Random bits
        x = torch.randint(0, 2, (n_bits,), dtype=torch.int64, device=device)
        z = qam_awgn_bits_torch(x, M, EbN0dB, device=device)
        ber = (z != x).to(torch.float64).mean().item()
        return ber
    else:
        x = np.random.randint(0, 2, size=(n_bits,), dtype=np.int64)
        z = qam_awgn_bits_np(x, M, EbN0dB)
        ber = (z != x).mean()
        return float(ber)

def make_code_noise_single(clean_code, eb_n0_db, qam_order=16, device=None):
    """
    clean_code: 0/1 tensor/array of ANY shape
    Returns: received_code (same shape), after M-QAM modulation, AWGN, hard-decision demod.
    """
    if TORCH_AVAILABLE and isinstance(clean_code, torch.Tensor):
        if device is None:
            device = clean_code.device
        flat = _as_torch_bits(clean_code, device=device)
        z = qam_awgn_bits_torch(flat, qam_order, eb_n0_db, device=device)
        return z.reshape(clean_code.shape).to(dtype=clean_code.dtype, device=device)
    else:
        flat = _as_numpy_bits(clean_code)
        z = qam_awgn_bits_np(flat, qam_order, eb_n0_db)
        return z.reshape(np.asarray(clean_code).shape).astype(np.asarray(clean_code).dtype)

# ----------------------------- Demo: sweep Eb/N0 and plot -----------------------------
def qam_theory_ber_curve(EbN0dB, M):
    k = math.log2(M)
    gamma_b = 10.0 ** (np.array(EbN0dB) / 10.0)
    return (4.0 / k) * (1 - 1.0 / math.sqrt(M)) * np.vectorize(_qfunc_np)(np.sqrt(3 * k / (M - 1) * gamma_b))

def simulate_and_plot(M=16, n_bits=50000, EbN0dB=range(0, 22, 2), device=None):
    ber_sim = []
    for eb in EbN0dB:
        ber_sim.append(QAM_AWGN_python(n_bits, M, eb, device=device))
    ber_theory = qam_theory_ber_curve(EbN0dB, M)

    plt.figure()
    plt.semilogy(list(EbN0dB), ber_sim, 'o-', linewidth=1.2, label='Simulation')
    plt.semilogy(list(EbN0dB), ber_theory, '--', linewidth=1.2, label='Theory (approx.)')
    plt.grid(True, which='both')
    plt.xlabel('E_b/N_0 (dB)')
    plt.ylabel('BER')
    plt.title(f'Gray-coded {M}-QAM over AWGN: BER vs E_b/N_0')
    plt.legend(loc='lower left')
    plt.ylim(1e-6, 1)
    plt.xlim(min(EbN0dB), max(EbN0dB))
    plt.show()
    plt.savefig(f'qam{M}_ber_vs_ebn0.png', dpi=300)