# Latent Compression Experiment

This folder contains a simple experiment on whether a compressed final video latent can beat H.264/MP4 on size while keeping similar decoded quality.

## Setup

- Source latent: `artifacts/sample_0_latent.pt`
- Reference video: `artifacts/sample_0.mp4`
- Reference raw frames: `artifacts/sample_0_uncompressed.rgb`
- VAE decode target: Open-Sora 1.0 `16x256x256`

## Methods

- `fp16_lzma`: direct float16 latent serialization + LZMA
- `delta_fp16_lzma`: temporal delta on latent frames + float16 + LZMA
- `delta_int8_lzma`: temporal delta + per-channel int8 quantization + LZMA
- `delta_int4_lzma`: temporal delta + per-channel int4 quantization + LZMA

## Key Result

On this sample, the simple latent codecs did **not** achieve both of these at the same time:

- smaller than MP4
- similar quality to MP4 after `decompress -> VAE decode -> video`

Most relevant numbers from `summary.json`:

- Reference MP4: `12,234 bytes`, PSNR vs decoded reference `38.09 dB`
- `delta_int8_lzma`: `28,364 bytes`, decoded-MP4 PSNR `37.74 dB`
- `delta_int4_lzma`: `3,840 bytes`, decoded-MP4 PSNR `28.01 dB`

So for this sample:

- `delta_int8_lzma` is close in quality, but still larger than MP4
- `delta_int4_lzma` is much smaller than MP4, but quality is much worse

## Files

- `latent_compression_experiment.py`: experiment script
- `summary.json`: measured sizes and quality metrics
- `*_decoded.mp4`: decoded videos for each latent compression method
- `*.bin`: compressed latent payloads
