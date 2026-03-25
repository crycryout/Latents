# Temporal Latent Codec Experiment

This experiment tests whether a lossy latent codec that uses frame-to-frame similarity can recover latents whose decoded raw frames are as clear as the raw frames obtained by compressing the original decoded video with MP4.

## Goal

Compare two paths against the same reference raw frames:

- MP4 baseline:
  - original latent
  - VAE decode to raw frames
  - H.264/MP4 compression
  - decode MP4 back to frames
- Temporal latent codec:
  - original latent
  - lossy latent compression using keyframes and temporal residuals
  - recover latent
  - VAE decode recovered latent to raw frames

The raw-frame clarity comparison is done against the original VAE-decoded raw frames.

## Codec Design

The latent codec is a temporal predictive codec:

- key latent frames are quantized directly
- non-key frames store quantized residuals relative to the previous reconstructed latent frame
- quantization is per-frame and per-channel
- low-bit packed symbols are compressed with LZMA

This directly exploits inter-frame similarity in latent space.

## Files

- `temporal_latent_codec_experiment.py`: full experiment script
- `summary.json`: full search results
- `temporal_k8_kb8_db3.bin`: smallest tested temporal codec that meets the MP4-quality target
- `temporal_k8_kb8_db3_decoded.mp4`: preview video for that codec
- `temporal_k16_kb8_db3.bin`: smaller-than-MP4 temporal codec that almost reaches MP4 quality but misses the target
- `temporal_k16_kb8_db3_decoded.mp4`: preview video for that codec
- `temporal_k16_kb8_db4.bin`: a stronger temporal codec with clear quality margin over MP4
- `temporal_k16_kb8_db4_decoded.mp4`: preview video for that codec
- `intra_b5.bin`: frame-independent latent quantization baseline
- `intra_b5_decoded.mp4`: preview video for the intra baseline

## Quality Criterion

Operational definition of "comparable to MP4":

- `PSNR >= reference_mp4_psnr_vs_ref - 0.5 dB`

For this sample:

- MP4 baseline size: `12,234 bytes`
- MP4 baseline PSNR vs reference raw frames: `38.0855 dB`

## Main Result

Yes. A temporal latent codec can match or slightly exceed the MP4 baseline in raw-frame quality.

Best small temporal codec that meets the MP4-quality target:

- `temporal_k8_kb8_db3`
- size: `12,752 bytes`
- PSNR vs reference raw frames: `38.4229 dB`
- MAE vs reference raw frames: `1.5529`

That is:

- only `518 bytes` larger than the MP4 file
- `0.337 dB` better than the MP4 baseline in PSNR
- lower error than the MP4 baseline in MAE

## Temporal Benefit

The temporal design matters.

Representative comparison:

- temporal codec `temporal_k16_kb8_db4`
  - `14,488 bytes`
  - `42.1744 dB`
- nearest intra-only baseline `intra_b5`
  - `15,284 bytes`
  - `36.8695 dB`

So the temporal latent codec gains about `5.30 dB` PSNR while using slightly less storage.

## Important Boundary

If the compressed latent must be smaller than MP4, quality drops below the MP4 baseline in this sample.

Example:

- `temporal_k16_kb8_db3`
  - `10,696 bytes`
  - `37.3277 dB`
  - about `0.758 dB` below the MP4 baseline

So the practical conclusion on this sample is:

- matching MP4 quality in latent space is feasible
- doing so at almost the same storage size is feasible
- beating MP4 in size while still matching its quality was not achieved in this experiment
