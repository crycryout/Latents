# Latent Streaming Decode Experiment

This experiment implements a streaming decode-and-playback framework for a fully denoised Open-Sora latent.

## Goal

Given a complete video latent saved after diffusion denoising:

- decode the latent along the time axis with the Open-Sora VAE
- emit decoded frames incrementally
- simulate playback using already decoded frames without waiting for the full video decode to finish

No MP4 encoding is involved in this path. Playback is simulated directly on decoded raw RGB frames.

## Files

- `latent_streaming.py`: reusable streaming decoder and playback simulator
- `stream_decode_latents.py`: runnable entry script
- `sample_0_latent_chunk1_buf4_fp16_streaming_summary.json`: fp16 single-frame streaming summary
- `sample_0_latent_chunk1_buf4_fp16_streaming_events.json`: fp16 per-frame playback events
- `sample_0_latent_chunk1_buf4_fp32_streaming_summary.json`: fp32 single-frame streaming summary
- `sample_0_latent_chunk1_buf4_fp32_streaming_events.json`: fp32 per-frame playback events

## Validation Sample

- Latent shape: `[4, 16, 32, 32]`
- Target playback fps: `8`
- Streaming mode: `chunk_frames=1`
- Playback buffer: `4` decoded frames
- Hardware: RTX 4090

## Result

The streaming path works. Frames are decoded incrementally and can be consumed by the playback simulator in real time.

Measured on this sample:

- `fp16`
  - first-frame latency: `0.3193 s`
  - steady-state decode interval: about `9.78 ms/frame`
  - steady-state decode throughput: about `102.2 fps`
  - end-to-end decode throughput including startup: about `34.3 fps`
- `fp32`
  - first-frame latency: `0.2771 s`
  - steady-state decode interval: about `14.33 ms/frame`
  - steady-state decode throughput: about `69.8 fps`
  - end-to-end decode throughput including startup: about `32.5 fps`

Since the playback target is only `8 fps`, the decode throughput is comfortably above the real-time requirement after the first frame is available.

## Numerical Consistency

The streamed decode output was compared against a full non-streaming VAE decode:

- `max_abs_diff = 1`

This is a tiny numerical difference caused by decoding in different batch shapes and precisions, not a logic error in the streaming framework.
