# Latents

Artifacts exported from an Open-Sora 1.0 `16x256x256` inference run on an RTX 4090.

## Files

- `artifacts/sample_0.mp4`: compressed H.264/MP4 video
- `artifacts/sample_0_latent.pt`: final denoised latent before VAE decode
- `artifacts/sample_0_latent.json`: latent metadata
- `artifacts/sample_0_uncompressed.rgb`: raw RGB24 frames, no video compression
- `artifacts/sample_0_uncompressed.nut`: rawvideo container, no video compression

## Sizes

- `sample_0.mp4`: 12,234 bytes
- `sample_0_latent.pt`: 263,556 bytes
- `sample_0_uncompressed.rgb`: 3,145,728 bytes
- `sample_0_uncompressed.nut`: 3,146,520 bytes

The raw `.rgb` file contains 16 frames at `256x256`, `rgb24`, `8 fps`.

## Latent Compression Experiment

See `experiments/latent_compression/` for a simple latent-compression benchmark against H.264/MP4.

High-level result on this sample:

- MP4 (`12,234 bytes`) is still smaller than the near-MP4-quality latent codec baseline
- `delta_int8_lzma` gets close in quality after decode, but is larger (`28,364 bytes`)
- `delta_int4_lzma` gets smaller than MP4 (`3,840 bytes`), but quality drops sharply
