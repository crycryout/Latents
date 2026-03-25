import argparse
import json
import os

import numpy as np
import torch

from opensora.registry import MODELS, build_module
from opensora.utils.latent_streaming import LatentStreamDecoder, PlaybackSimulator, decode_full_latent
from opensora.utils.misc import to_torch_dtype


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-path", required=True, type=str, help="Path to a saved latent .pt file")
    parser.add_argument("--vae-pretrained", default="./pretrained_models/sd-vae-ft-ema", type=str)
    parser.add_argument("--dtype", default="fp16", type=str, help="VAE decode dtype")
    parser.add_argument("--chunk-frames", default=1, type=int, help="How many latent frames to decode per chunk")
    parser.add_argument("--buffer-frames", default=8, type=int, help="Decoded-frame queue capacity")
    parser.add_argument("--fps", default=None, type=int, help="Override playback fps")
    parser.add_argument("--save-dir", default=None, type=str, help="Optional directory to save simulation logs")
    parser.add_argument(
        "--verify-full-decode",
        action="store_true",
        help="Verify streamed frames exactly match a full non-streaming decode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(args.dtype)

    payload = torch.load(args.latent_path, map_location="cpu")
    latent = payload["latent"].contiguous()
    fps = args.fps if args.fps is not None else int(payload.get("fps", 8))

    vae = build_module(dict(type="VideoAutoencoderKL", from_pretrained=args.vae_pretrained), MODELS).to(
        device, dtype
    ).eval()

    decoder = LatentStreamDecoder(
        vae=vae,
        device=device,
        dtype=dtype,
        chunk_frames=args.chunk_frames,
        max_buffer_frames=args.buffer_frames,
    )
    player = PlaybackSimulator(fps=fps)

    decoder.start(latent)
    result = player.run(decoder)
    decoder.join()

    summary = {
        "latent_path": args.latent_path,
        "latent_shape": list(latent.shape),
        "fps": fps,
        "chunk_frames": args.chunk_frames,
        "buffer_frames": args.buffer_frames,
        "frames_decoded": int(result["frames"].shape[0]),
        "underflow_count": result["underflow_count"],
        "total_stall_s": result["total_stall_s"],
        "final_playback_time_s": result["final_playback_time_s"],
        "first_frame_latency_s": (float(result["events"][0].ready_at) if result["events"] else 0.0),
        "effective_fps": (
            float(result["frames"].shape[0] / result["final_playback_time_s"])
            if result["final_playback_time_s"] > 0
            else 0.0
        ),
        "events_preview": [
            {
                "frame_index": event.frame_index,
                "ready_at": round(event.ready_at, 6),
                "scheduled_at": round(event.scheduled_at, 6),
                "presented_at": round(event.presented_at, 6),
                "stall_before_present_s": round(event.stall_before_present_s, 6),
                "queue_size_after_pop": event.queue_size_after_pop,
            }
            for event in result["events"][: min(8, len(result["events"]))]
        ],
    }

    if args.verify_full_decode:
        full_frames = decode_full_latent(vae, latent, device=device, dtype=dtype)
        identical = np.array_equal(result["frames"], full_frames)
        summary["verify_full_decode"] = {
            "identical": bool(identical),
            "max_abs_diff": (
                int(np.max(np.abs(result["frames"].astype(np.int16) - full_frames.astype(np.int16))))
                if result["frames"].size > 0
                else 0
            ),
        }

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.latent_path))[0]
        run_tag = f"chunk{args.chunk_frames}_buf{args.buffer_frames}_{args.dtype}"
        summary_path = os.path.join(args.save_dir, f"{base}_{run_tag}_streaming_summary.json")
        events_path = os.path.join(args.save_dir, f"{base}_{run_tag}_streaming_events.json")
        events_payload = [
            {
                "frame_index": event.frame_index,
                "ready_at": event.ready_at,
                "scheduled_at": event.scheduled_at,
                "presented_at": event.presented_at,
                "stall_before_present_s": event.stall_before_present_s,
                "queue_size_after_pop": event.queue_size_after_pop,
            }
            for event in result["events"]
        ]
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        with open(events_path, "w") as f:
            json.dump(events_payload, f, indent=2)
        print(f"Streaming summary saved to: {summary_path}")
        print(f"Streaming events saved to: {events_path}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
