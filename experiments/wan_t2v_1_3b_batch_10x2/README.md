# Wan2.1 T2V 1.3B Batch With Saved Latents

This directory contains a batch `Wan2.1-T2V-1.3B` text-to-video run executed on a local RTX 4090, plus the final denoised latent saved immediately before VAE decode for every sample.

## Contents

- `batch_10x2_latents/`: generated artifacts
- `batch_generate_t2v_with_latents.py`: batch driver script
- `text2video_patched.py`: patched `WanT2V.generate()` that saves the final denoised latent before decode
- `attention_patched.py`: patched attention fallback used to run without `flash_attn`

## Generation Setup

- Model: `Wan2.1-T2V-1.3B`
- Task: `t2v-1.3B`
- Resolution: `832x480`
- Frames: `49`
- Output FPS: `16`
- Sampling steps: `30`
- Solver: `unipc`
- Guide scale: `6.0`
- Sample shift: `8.0`
- Prompt groups: `10`
- Variants per group: `2`
- Total videos: `20`
- Total wall time: `2374.8922 s`

## Artifact Layout

- `batch_10x2_latents/manifest.json`: run manifest
- `batch_10x2_latents/group_XX/prompt.txt`: prompt text for that group
- `batch_10x2_latents/group_XX/*.mp4`: generated videos
- `batch_10x2_latents/group_XX/*_latent.pt`: final denoised latents saved before VAE decode
- `batch_10x2_latents/group_XX/*.json`: per-sample metadata

## Aggregate Stats

- Video count: `20`
- Latent count: `20`
- Total MP4 bytes: `91,866,025`
- Total latent bytes: `103,873,172`
- MP4 size range: `3,588,437 B` to `5,915,623 B`
- Latent size range: `5,193,601 B` to `5,193,665 B`
- Latent shape: `16 x 13 x 60 x 104`

## Notes

- The latent saved in each `*_latent.pt` file is the final denoised latent, not the initial noise and not an intermediate denoising state.
- `manifest.json` records prompt, seed, elapsed time, MP4 size, latent size, and latent shape for every sample.
