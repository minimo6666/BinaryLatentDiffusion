# Physical-QAM expectation-consistent BFM (2026-08-18)

This experiment initializes the 768-wide, 24-layer, C64/16x16 model from:

    UnifyBinaryFlowDiffusion/.../
    bernoulli_flow_lsun_churches_256_expectation_consistent_aux0p1/
    saved_models/flow_lsun_ema_50000.th

It then fine-tunes on clean FFHQ train-split C64 codes. Each training observation
is produced by the canonical physical channel: Gray 16-QAM, unit Es, complex
AWGN and hard demapping. The sampled path spans Eb/N0 -15 through 20 dB.
Unlike the old training path, the observed bit errors retain within-symbol QAM
correlations instead of being sampled as independent Bernoulli flips.

The expectation-consistent reverse bridge still uses the exact marginal BER
schedule; that is an explicit modeling approximation. Direct X0 supervision is
against the clean bits, and a frozen FFHQ C64 decoder adds image-domain MSE.

Evaluation uses the exact same seeded 100-image ffhqvalidation manifest,
latent dimensions, QAM function, Eb/N0 values and channel seeds as JSCC. It
saves all ground-truth, raw-QAM and BFM-restored images and reports true
post-denoising BER, BER correction rate and PSNR before/after denoising.

Run on the fifth GPU:

    bash experiments/bfm_expetation_consistent_2026_8_18/train.sh
    bash experiments/bfm_expetation_consistent_2026_8_18/eval.sh
