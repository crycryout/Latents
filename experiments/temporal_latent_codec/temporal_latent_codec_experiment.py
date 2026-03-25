import argparse
import json
import lzma
import math
import os
import struct
from collections import OrderedDict

import imageio.v2 as imageio
import numpy as np
import torch

from opensora.registry import MODELS, build_module


MAGIC = b"TLATC01\0"
HEADER_STRUCT = struct.Struct("<8s4I4H")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-path", required=True, type=str)
    parser.add_argument("--mp4-path", required=True, type=str)
    parser.add_argument("--vae-pretrained", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    return parser.parse_args()


def tensor_to_tchw(latent):
    return latent.permute(1, 0, 2, 3).contiguous().cpu().numpy()


def tchw_to_tensor(array):
    return torch.from_numpy(array).permute(1, 0, 2, 3).contiguous()


def encode_lzma(payload):
    return lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)


def decode_lzma(payload):
    return lzma.decompress(payload)


def decode_video_frames(video_path):
    reader = imageio.get_reader(video_path)
    frames = np.stack([frame for frame in reader], axis=0)
    reader.close()
    return frames.astype(np.uint8)


def decode_latent_video(vae, latent_tensor):
    with torch.no_grad():
        sample = vae.decode(latent_tensor.unsqueeze(0).to("cuda", dtype=torch.float16))[0]
    sample = sample.clamp(-1, 1)
    frames = sample.add(1).div(2).mul(255).add_(0.5).clamp_(0, 255)
    return frames.permute(1, 2, 3, 0).to("cpu", torch.uint8).numpy()


def psnr_uint8(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0) - 10 * math.log10(mse)


def mae_uint8(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def write_video(path, frames, fps):
    imageio.mimwrite(path, frames, fps=fps, codec="libx264")


def pack_values(values, bits):
    mask = (1 << bits) - 1
    acc = 0
    acc_bits = 0
    out = bytearray()
    for value in values.reshape(-1).tolist():
        acc = (acc << bits) | (int(value) & mask)
        acc_bits += bits
        while acc_bits >= 8:
            acc_bits -= 8
            out.append((acc >> acc_bits) & 0xFF)
            if acc_bits > 0:
                acc &= (1 << acc_bits) - 1
            else:
                acc = 0
    if acc_bits > 0:
        out.append((acc << (8 - acc_bits)) & 0xFF)
    return bytes(out)


def unpack_values(blob, count, bits):
    mask = (1 << bits) - 1
    out = np.empty(count, dtype=np.uint8)
    acc = 0
    acc_bits = 0
    idx = 0
    for byte in blob:
        acc = (acc << 8) | int(byte)
        acc_bits += 8
        while acc_bits >= bits and idx < count:
            acc_bits -= bits
            out[idx] = (acc >> acc_bits) & mask
            idx += 1
            if acc_bits > 0:
                acc &= (1 << acc_bits) - 1
            else:
                acc = 0
    return out


def quantize_frame_per_channel(frame_chw, bits):
    channels = frame_chw.shape[0]
    qmax = (1 << (bits - 1)) - 1
    flat = frame_chw.reshape(channels, -1)
    max_abs = np.max(np.abs(flat), axis=1)
    max_abs = np.maximum(max_abs, 1e-8)
    scales = (max_abs / qmax).astype(np.float16)
    q = np.round(frame_chw / scales[:, None, None]).clip(-qmax, qmax).astype(np.int16)
    codes = (q + qmax).astype(np.uint8)
    return codes, scales


def dequantize_frame_per_channel(codes, scales, bits):
    qmax = (1 << (bits - 1)) - 1
    q = codes.astype(np.int16) - qmax
    return q.astype(np.float32) * scales.astype(np.float32)[:, None, None]


def encode_temporal_predictive(latent_tchw, key_interval, key_bits, delta_bits, scheme_name):
    frames, channels, height, width = latent_tchw.shape
    recon = np.zeros_like(latent_tchw, dtype=np.float32)
    scales = np.zeros((frames, channels), dtype=np.float16)
    lengths = np.zeros(frames, dtype=np.uint16)
    payload_parts = []

    for frame_index in range(frames):
        is_key = frame_index % key_interval == 0
        bits = key_bits if is_key else delta_bits
        target = latent_tchw[frame_index] if is_key else latent_tchw[frame_index] - recon[frame_index - 1]
        codes, frame_scales = quantize_frame_per_channel(target, bits)
        packed = pack_values(codes.reshape(-1), bits)
        lengths[frame_index] = len(packed)
        scales[frame_index] = frame_scales
        payload_parts.append(packed)

        dequantized = dequantize_frame_per_channel(codes, frame_scales, bits)
        recon[frame_index] = dequantized if is_key else recon[frame_index - 1] + dequantized

    header = HEADER_STRUCT.pack(
        MAGIC,
        frames,
        channels,
        height,
        width,
        key_interval,
        key_bits,
        delta_bits,
        0,
    )
    blob = encode_lzma(header + scales.tobytes(order="C") + lengths.tobytes(order="C") + b"".join(payload_parts))
    return {
        "name": scheme_name,
        "blob": blob,
        "recon_tchw": recon,
        "decode": lambda payload: decode_temporal_predictive(payload),
        "key_interval": key_interval,
        "key_bits": key_bits,
        "delta_bits": delta_bits,
    }


def decode_temporal_predictive(payload):
    raw = decode_lzma(payload)
    header = HEADER_STRUCT.unpack(raw[: HEADER_STRUCT.size])
    magic, frames, channels, height, width, key_interval, key_bits, delta_bits, _ = header
    if magic != MAGIC:
        raise ValueError("Invalid codec payload")

    offset = HEADER_STRUCT.size
    scales_count = frames * channels
    scales_bytes = scales_count * np.dtype(np.float16).itemsize
    scales = np.frombuffer(raw[offset : offset + scales_bytes], dtype=np.float16).reshape(frames, channels)
    offset += scales_bytes

    lengths_bytes = frames * np.dtype(np.uint16).itemsize
    lengths = np.frombuffer(raw[offset : offset + lengths_bytes], dtype=np.uint16)
    offset += lengths_bytes

    recon = np.zeros((frames, channels, height, width), dtype=np.float32)
    value_count = channels * height * width
    for frame_index in range(frames):
        bits = key_bits if frame_index % key_interval == 0 else delta_bits
        packed_len = int(lengths[frame_index])
        packed = raw[offset : offset + packed_len]
        offset += packed_len
        codes = unpack_values(packed, value_count, bits).reshape(channels, height, width)
        dequantized = dequantize_frame_per_channel(codes, scales[frame_index], bits)
        recon[frame_index] = dequantized if frame_index % key_interval == 0 else recon[frame_index - 1] + dequantized
    return recon


def temporal_search_configs():
    configs = []
    for bits in [8, 7, 6, 5, 4]:
        configs.append(("intra", 1, bits, bits))
    for key_interval in [16, 8, 4]:
        for key_bits in [8, 7, 6]:
            for delta_bits in [8, 7, 6, 5, 4, 3]:
                if key_bits < delta_bits:
                    continue
                configs.append(("temporal", key_interval, key_bits, delta_bits))
    return configs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    payload = torch.load(args.latent_path, map_location="cpu")
    latent = payload["latent"].contiguous()
    fps = int(payload["fps"])
    latent_tchw = tensor_to_tchw(latent)

    vae = build_module(dict(type="VideoAutoencoderKL", from_pretrained=args.vae_pretrained), MODELS).to(
        "cuda", torch.float16
    ).eval()

    ref_frames = decode_latent_video(vae, latent)
    mp4_frames = decode_video_frames(args.mp4_path)
    mp4_psnr = psnr_uint8(mp4_frames, ref_frames)
    mp4_mae = mae_uint8(mp4_frames, ref_frames)
    mp4_size = os.path.getsize(args.mp4_path)

    results = []
    for mode, key_interval, key_bits, delta_bits in temporal_search_configs():
        scheme_name = (
            f"{mode}_k{key_interval}_kb{key_bits}_db{delta_bits}"
            if mode == "temporal"
            else f"{mode}_b{key_bits}"
        )
        scheme = encode_temporal_predictive(
            latent_tchw=latent_tchw,
            key_interval=key_interval,
            key_bits=key_bits,
            delta_bits=delta_bits,
            scheme_name=scheme_name,
        )

        recon_tchw = scheme["decode"](scheme["blob"])
        recon_latent = tchw_to_tensor(recon_tchw)
        recon_frames = decode_latent_video(vae, recon_latent)

        compressed_path = os.path.join(args.output_dir, f"{scheme_name}.bin")
        with open(compressed_path, "wb") as f:
            f.write(scheme["blob"])

        preview_mp4_path = os.path.join(args.output_dir, f"{scheme_name}_decoded.mp4")
        write_video(preview_mp4_path, recon_frames, fps=fps)

        result = OrderedDict(
            name=scheme_name,
            mode=mode,
            key_interval=key_interval,
            key_bits=key_bits,
            delta_bits=delta_bits,
            compressed_bytes=len(scheme["blob"]),
            compressed_kib=len(scheme["blob"]) / 1024.0,
            ratio_vs_latent_pt=os.path.getsize(args.latent_path) / len(scheme["blob"]),
            smaller_than_mp4=len(scheme["blob"]) < mp4_size,
            psnr_vs_ref=psnr_uint8(recon_frames, ref_frames),
            mae_vs_ref=mae_uint8(recon_frames, ref_frames),
            psnr_margin_vs_mp4=psnr_uint8(recon_frames, ref_frames) - mp4_psnr,
            mae_margin_vs_mp4=mae_uint8(recon_frames, ref_frames) - mp4_mae,
            meets_mp4_quality=(psnr_uint8(recon_frames, ref_frames) >= mp4_psnr - 0.5),
            decoded_preview_mp4_bytes=os.path.getsize(preview_mp4_path),
        )
        results.append(result)

    best_meeting = None
    for result in sorted(results, key=lambda item: (not item["meets_mp4_quality"], item["compressed_bytes"])):
        if result["meets_mp4_quality"]:
            best_meeting = result
            break

    best_temporal = max(
        [result for result in results if result["mode"] == "temporal"],
        key=lambda item: (item["psnr_margin_vs_mp4"], -item["compressed_bytes"]),
    )
    smallest_temporal_meeting = None
    for result in sorted(
        [result for result in results if result["mode"] == "temporal" and result["meets_mp4_quality"]],
        key=lambda item: item["compressed_bytes"],
    ):
        smallest_temporal_meeting = result
        break

    summary = OrderedDict(
        latent_pt_bytes=os.path.getsize(args.latent_path),
        latent_shape=list(latent.shape),
        latent_dtype=str(latent.dtype),
        reference_mp4_bytes=mp4_size,
        reference_mp4_psnr_vs_ref=mp4_psnr,
        reference_mp4_mae_vs_ref=mp4_mae,
        comparable_quality_definition="PSNR >= reference_mp4_psnr_vs_ref - 0.5 dB",
        best_meeting_mp4_quality=best_meeting,
        smallest_temporal_meeting_mp4_quality=smallest_temporal_meeting,
        best_temporal_quality=best_temporal,
        results=sorted(results, key=lambda item: (not item["meets_mp4_quality"], item["compressed_bytes"])),
    )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
