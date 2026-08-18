# Fair fixed-SNR Deep JSCC experiment

This replaces the historical C96/32x32 JSCC setup with exactly the same binary
latent interface used by the current BFM experiment:

- FFHQ C64 autoencoder, latent shape 64x16x16 (16,384 bits/image)
- Gray-coded 16-QAM
- unit average symbol energy Es=1
- complex AWGN with sigma^2=Es/(2 log2(M) 10^(Eb/N0/10)) per real dimension
- hard-decision demapping
- the x-axis always means Eb/N0 in dB

Every model starts from the same clean FFHQ BAE_C64 checkpoint. Encoder,
quantizer embedding/projection and decoder are fine-tuned jointly with image
MSE, which directly optimizes PSNR. Eight models are trained at fixed Eb/N0:
-15, -10, -5, 0, 5, 10, 15 and 20 dB. Four GPUs process two models each.

The FFHQ lists contain 60,000 train and 10,000 validation images with zero
filename overlap. A seeded manifest fixes the same 100 validation images for
all JSCC and BFM evaluations.

Run:

    bash experiments/different_snr_jscc/train_all.sh
    bash experiments/different_snr_jscc/eval_all.sh

cycle_ber is the Hamming distance between the source encoder bits and bits
obtained by re-encoding the reconstructed image. It is reported because an
encoder/decoder JSCC system has no explicit corrected-bit output. It must not
be confused with BFM's true post-denoising BER.
