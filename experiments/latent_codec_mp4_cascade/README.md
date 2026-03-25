# Latent Codec + MP4 Cascade Experiment

This experiment evaluates a cascaded pipeline:

1. Compress the Open-Sora 1.0 denoised latent with a temporal latent codec
2. Restore the latent
3. Decode raw frames with the VAE
4. Encode those frames again with H.264/MP4

The goal is to measure the final visual quality after *both* the latent codec and the MP4 stage, relative to a fair baseline built from:

`original latent -> VAE decode -> libx264 MP4`

## Files

- `latent_codec_mp4_cascade_experiment.py`: experiment script
- `summary.json`: full metric table for all tested codecs
- `reference_from_original_latent.mp4`: baseline MP4 built from the original latent
- `temporal_k4_kb8_db4_decoded.mp4`: smallest tested codec whose final MP4 stays within `0.5 dB` of the baseline MP4
- `temporal_k8_kb8_db3_decoded.mp4`: near-MP4-quality latent codec before the second MP4 stage, but it drops after re-encoding
- `temporal_k16_kb8_db3_decoded.mp4`: smaller-than-MP4 latent codec that drops more clearly after re-encoding

## High-Level Result

- Yes, the latent can be compressed first and the generated frames can still be compressed again with MP4.
- But the second lossy stage matters: a codec that looked near-MP4-quality before re-encoding can still lose about `2 dB` after the MP4 stage.
- On this sample, keeping the final MP4 quality close to the baseline requires a noticeably less aggressive latent codec than the raw-frame-only comparison suggested.
- Heavy latent distortion also makes the final MP4 less compressible, so the output MP4 can become larger than the baseline.
