# Network Latent Streaming Experiment

This experiment implements and validates a networked streaming pipeline for Open-Sora video latents:

- the server stores a fully denoised latent
- the server sends latent chunks over a TCP stream
- the client receives latent chunks over the network
- the client runs local VAE decode on each chunk
- the client simulates playback directly on decoded raw RGB frames

No MP4 encoding is used in this path.

## Files

- `network_latent_streaming.py`: reusable server/client latent streaming utilities
- `network_stream_latents.py`: experiment entry script
- `sample_0_latent_net1mbps_fp16_r8_chunk1_fp16_fp16_summary.json`: long-run success case summary
- `sample_0_latent_net1mbps_fp16_r8_chunk1_fp16_fp16_server_events.json`: server send events
- `sample_0_latent_net1mbps_fp16_r8_chunk1_fp16_fp16_client_chunks.json`: client receive/decode events
- `sample_0_latent_net1mbps_fp16_r8_chunk1_fp16_fp16_playback_events.json`: playback events
- `sample_0_latent_net1mbps_fp32_r8_chunk1_fp32_fp16_summary.json`: long-run constrained-bandwidth failure case summary
- `sample_0_latent_net1mbps_fp32_r8_chunk1_fp32_fp16_server_events.json`: server send events
- `sample_0_latent_net1mbps_fp32_r8_chunk1_fp32_fp16_client_chunks.json`: client receive/decode events
- `sample_0_latent_net1mbps_fp32_r8_chunk1_fp32_fp16_playback_events.json`: playback events

## Validation Setup

- Source latent: Open-Sora 1.0 `16x256x256` sample
- Replay length: repeated `8x` to make `128` frames
- Playback fps: `8`
- Total playback duration: `16 s`
- Transport: localhost TCP with application-level bandwidth shaping
- Artificial network limit: `1.0 Mbps`
- Artificial per-chunk delay: `5 ms`
- Chunk size: `1` latent frame per packet
- Client decode device: RTX 4090

## Key Result

The system can stream latents over a network and play raw decoded frames in real time, as long as the latent transport bitrate stays below the network budget.

### Success case: `fp16` latent transport at `1 Mbps`

From `sample_0_latent_net1mbps_fp16_r8_chunk1_fp16_fp16_summary.json`:

- required wire bitrate for playback: `0.5347 Mbps`
- achieved receive bitrate: `0.9278 Mbps`
- first-frame latency: `0.3929 s`
- steady frame-ready decode throughput: `14.24 fps`
- playback underflow count: `1`
- that single underflow is startup only
- result: `real_time_supported = true`

### Failure case: `fp32` latent transport at `1 Mbps`

From `sample_0_latent_net1mbps_fp32_r8_chunk1_fp32_fp16_summary.json`:

- required wire bitrate for playback: `1.0590 Mbps`
- achieved receive bitrate: `0.9638 Mbps`
- steady frame-ready decode throughput: `7.35 fps`
- playback underflow count: `71`
- result: `real_time_supported = false`

This boundary is the core result: the bottleneck is not VAE decode on the client, but network bitrate once the wire format grows beyond the available transport budget.

## System Metrics

In the successful `fp16` long-run test:

- process CPU percent average: `342.65`
- process RSS peak: `1280.34 MB`
- CUDA max memory allocated: `333.34 MB`
- CUDA max memory reserved: `454.0 MB`

## Interpretation

For this Open-Sora sample, a practical deployment path is:

- keep the denoised latent on the server
- transmit `fp16` latent chunks over the network
- decode with the VAE on the client
- render or play the decoded raw frames locally

That gives a raw-frame playback path without MP4 compression, while still using the latent representation to reduce transport cost.
