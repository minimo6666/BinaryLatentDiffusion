# Visual previews every 5000 steps

The training script saves a fixed validation preview whenever it saves a
checkpoint (5000, 10000, ..., 30000 by default). The same images, channel seed,
and SNR values are used at every step, so visual changes are comparable.

Outputs are written below:

```text
results/previews/
  README.txt
  step_005000/
    0_db/
      gt/                 # original images
      jscc25/             # frozen JSCC-25 baseline
      jscc25_bld/         # BLD-denoised images
      comparison_grid.png # rows: GT / baseline / BLD
    4_db/
    8_db/
    12_db/
    preview_metrics.json
  step_010000/
  ...
  step_030000/
```

The defaults are four fixed validation images at 0, 4, 8 and 12 dB. Change
`PREVIEW_IMAGES` and `PREVIEW_SNRS` in `config.sh` if needed.

To generate previews for already saved checkpoints without retraining:

```bash
CUDA_VISIBLE_DEVICES=5 \
PYTHONPATH=/home/minimo/Project/BLD_DSC/BinaryLatentDiffusion:. \
/home/minimo/miniconda3/envs/DavinciLUT/bin/python \
experiments/deepjscc_en_de_and_bld_denoise_2026_8_2/preview_checkpoints.py
```
