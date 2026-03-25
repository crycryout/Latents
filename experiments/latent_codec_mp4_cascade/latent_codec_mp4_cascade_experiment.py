import argparse
import json
import os
import sys
from collections import OrderedDict

import torch

from opensora.registry import MODELS, build_module

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from temporal_latent_codec_experiment import (
    decode_latent_video,
    decode_video_frames,
    mae_uint8,
    psnr_uint8,
    write_video,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-path", required=True, type=str)
    parser.add_argument("--vae-pretrained", required=True, type=str)
    parser.add_argument("--codec-summary-path", required=True, type=str)
    parser.add_argument("--codec-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    return parser.parse_args()


def mp4_quality_record(
    name,
    preview_mp4_path,
    ref_frames,
    baseline_mp4_frames,
    baseline_psnr,
    baseline_mae,
    baseline_mp4_bytes,
    raw_record,
):
    mp4_frames = decode_video_frames(preview_mp4_path)
    mp4_psnr_vs_ref = psnr_uint8(mp4_frames, ref_frames)
    mp4_mae_vs_ref = mae_uint8(mp4_frames, ref_frames)
    return OrderedDict(
        name=name,
        compressed_bytes=raw_record["compressed_bytes"],
        compressed_kib=raw_record["compressed_kib"],
        smaller_than_reference_mp4=raw_record["compressed_bytes"] < baseline_mp4_bytes,
        raw_codec_psnr_vs_ref=raw_record["psnr_vs_ref"],
        raw_codec_mae_vs_ref=raw_record["mae_vs_ref"],
        mp4_after_codec_bytes=os.path.getsize(preview_mp4_path),
        mp4_after_codec_psnr_vs_ref=mp4_psnr_vs_ref,
        mp4_after_codec_mae_vs_ref=mp4_mae_vs_ref,
        additional_mp4_loss_db=mp4_psnr_vs_ref - raw_record["psnr_vs_ref"],
        additional_mp4_mae_delta=mp4_mae_vs_ref - raw_record["mae_vs_ref"],
        mp4_psnr_margin_vs_reference_mp4=mp4_psnr_vs_ref - baseline_psnr,
        mp4_mae_margin_vs_reference_mp4=mp4_mae_vs_ref - baseline_mae,
        mp4_psnr_vs_reference_mp4_frames=psnr_uint8(mp4_frames, baseline_mp4_frames),
        mp4_mae_vs_reference_mp4_frames=mae_uint8(mp4_frames, baseline_mp4_frames),
        meets_reference_mp4_quality=(mp4_psnr_vs_ref >= baseline_psnr - 0.5),
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    payload = torch.load(args.latent_path, map_location="cpu")
    latent = payload["latent"].contiguous()
    fps = int(payload["fps"])

    vae = build_module(dict(type="VideoAutoencoderKL", from_pretrained=args.vae_pretrained), MODELS).to(
        "cuda", torch.float16
    ).eval()

    ref_frames = decode_latent_video(vae, latent)

    reference_mp4_path = os.path.join(args.output_dir, "reference_from_original_latent.mp4")
    write_video(reference_mp4_path, ref_frames, fps=fps)
    reference_mp4_frames = decode_video_frames(reference_mp4_path)
    reference_mp4_psnr = psnr_uint8(reference_mp4_frames, ref_frames)
    reference_mp4_mae = mae_uint8(reference_mp4_frames, ref_frames)

    codec_summary = json.load(open(args.codec_summary_path))
    codec_results = codec_summary["results"]

    results = []
    for raw_record in codec_results:
        name = raw_record["name"]
        preview_mp4_path = os.path.join(args.codec_dir, f"{name}_decoded.mp4")
        if not os.path.exists(preview_mp4_path):
            continue
        results.append(
            mp4_quality_record(
                name=name,
                preview_mp4_path=preview_mp4_path,
                ref_frames=ref_frames,
                baseline_mp4_frames=reference_mp4_frames,
                baseline_psnr=reference_mp4_psnr,
                baseline_mae=reference_mp4_mae,
                baseline_mp4_bytes=os.path.getsize(reference_mp4_path),
                raw_record=raw_record,
            )
        )

    results.sort(key=lambda item: (not item["meets_reference_mp4_quality"], item["compressed_bytes"]))

    best_meeting = next((item for item in results if item["meets_reference_mp4_quality"]), None)
    smallest_smaller_than_reference_mp4 = next(
        (item for item in results if item["compressed_bytes"] < os.path.getsize(reference_mp4_path)), None
    )
    best_temporal_margin = max(results, key=lambda item: (item["mp4_psnr_margin_vs_reference_mp4"], -item["compressed_bytes"]))

    summary = OrderedDict(
        latent_pt_bytes=os.path.getsize(args.latent_path),
        latent_shape=list(latent.shape),
        reference_mp4_bytes=os.path.getsize(reference_mp4_path),
        reference_mp4_psnr_vs_ref=reference_mp4_psnr,
        reference_mp4_mae_vs_ref=reference_mp4_mae,
        comparable_quality_definition="final_mp4_psnr_vs_ref >= reference_mp4_psnr_vs_ref - 0.5 dB",
        best_meeting_reference_mp4_quality=best_meeting,
        smallest_smaller_than_reference_mp4=smallest_smaller_than_reference_mp4,
        best_mp4_margin=best_temporal_margin,
        results=results,
    )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
