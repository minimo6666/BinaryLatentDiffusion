%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION THAT CALCULATES THE BER OF M-QAM IN AWGN
% n_bits: Input, number of bits
% M: Input, constellation size
% EbNodB: Input, energy per bit to noise power spectral density
% ber: Output, bit error rate
% Copyright RAYmaps (www.raymaps.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% function[ber]= QAM_AWGN(n_bits, M, EbNodB)
% 
% % Transmitter
% k=log2(M);
% EbNo=10^(EbNodB/10);
% x=transpose(round(rand(1,n_bits)));
% h1=qammod(M);
% h1.inputtype='bit';
% h1.symbolorder='gray';
% y=modulate(h1,x);
% 
% % Channel
% Eb=mean((abs(y)).^2)/k;
% sigma=sqrt(Eb/(2*EbNo));
% w=sigma*(randn(1,n_bits/k)+1i*randn(1,n_bits/k));
% r=y+w';
% 
% % Receiver
% h2=qamdemod(M);
% h2.outputtype='bit';
% h2.symbolorder='gray';
% h2.decisiontype='hard decision';
% z=demodulate(h2,r);
% ber=(n_bits-sum(x==z))/n_bits;
% return
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
function ber = QAM_AWGN(n_bits, M, EbNodB)
%QAM_AWGN  BER of Gray-coded M-QAM over AWGN (modern MATLAB)
%   ber = QAM_AWGN(n_bits, M, EbNodB)
%   n_bits : number of bits to simulate
%   M      : QAM order (power of two)
%   EbNodB : Eb/N0 in dB
%
%   Requires Communications Toolbox: qammod/qamdemod
%
%   Uses Gray mapping and unit-average-power normalization.

    arguments
        n_bits (1,1) {mustBeInteger, mustBePositive}
        M      (1,1) {mustBeInteger, mustBePositive}
        EbNodB (1,1) {mustBeReal}
    end
    k = log2(M);
    assert(abs(k - round(k)) < eps, 'M must be a power of two.');
    k = round(k);

    % ----- Transmitter -----
    % random bits (column vector)
    x = randi([0 1], n_bits, 1);

    % pad to multiple of k for bit-wise modulation
    remBits = mod(n_bits, k);
    if remBits ~= 0
        pad = k - remBits;
        x = [x; zeros(pad,1)]; %#ok<AGROW>
    end

    % M-QAM Gray mapping, bit input, unit-average-power
    % (modern qammod usage; default mapping已是'gray'，这里显式指定更清晰)
    tx = qammod(x, M, 'gray', ...
                'InputType','bit', ...
                'UnitAveragePower', true);   % Es = 1

    % ----- AWGN Channel -----
    EbNo   = 10^(EbNodB/10);
    Eb     = 1 / k;                      % since Es=1
    sigma2 = Eb / (2*EbNo);              % complex baseband: N0/2 per dim
    sigma  = sqrt(sigma2);
    w = sigma * (randn(size(tx)) + 1i*randn(size(tx)));
    rx = tx + w;

    % ----- Receiver -----
    % hard-decision bit output; use same normalization/mapping as modulator
    z = qamdemod(rx, M, 'gray', ...
                 'OutputType','bit', ...
                 'UnitAveragePower', true);

    % compute BER only on the original n_bits (exclude padding)
    z = z(1:n_bits);
    ber = sum(z ~= x(1:n_bits)) / n_bits;
end
