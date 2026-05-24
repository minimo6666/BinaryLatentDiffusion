% run_qam_ber.m
% 扫描 Eb/N0 = 0:2:20 dB，绘制 Gray 映射 M-QAM 在 AWGN 下的 BER 曲线
% 依赖：QAM_AWGN.m（你刚保存的新版函数）

clear; clc; rng default;

% ---------- 参数 ----------
M        = 16;          % QAM 星座阶数（可改为 4/16/64/...）
n_bits   = 2e5;         % 每个Eb/N0点的比特数（越大越准但越慢）
EbN0dB   = 0:2:20;      % 扫描范围

% ---------- 仿真 ----------
ber_sim = zeros(size(EbN0dB));
for ii = 1:numel(EbN0dB)
    ber_sim(ii) = QAM_AWGN(n_bits, M, EbN0dB(ii));
end

% ---------- 理论近似（Gray 映射 M-QAM）----------
% 经典近似：Pb ≈ (4/k)*(1 - 1/sqrt(M)) * Q( sqrt( 3*k/(M-1) * Eb/N0 ) )
k  = log2(M);
Q  = @(x) 0.5*erfc(x/sqrt(2));  % 避免工具箱依赖
gamma_b = 10.^(EbN0dB/10);      % Eb/N0 (线性)
ber_th  = (4/k)*(1 - 1/sqrt(M)) .* Q( sqrt( 3*k/(M-1) .* gamma_b ) );

% ---------- 作图 ----------
figure;
semilogy(EbN0dB, ber_sim, 'o-','LineWidth',1.2); hold on;
semilogy(EbN0dB, ber_th,  '--','LineWidth',1.2);
grid on; xlabel('E_b/N_0 (dB)'); ylabel('BER');
title(sprintf('Gray-coded %d-QAM over AWGN: BER vs. E_b/N_0', M));
legend('Simulation','Theory (approx.)','Location','southwest');
ylim([1e-6 1]);
xlim([EbN0dB(1) EbN0dB(end)]);
