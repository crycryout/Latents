import argparse
import json
import lzma
import math
import os
import struct
from collections import OrderedDict

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from opensora.registry import MODELS, build_module


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


def pack_nibbles(values):
    if values.size % 2 == 1:
        values = np.concatenate([values, np.zeros(1, dtype=np.uint8)])
    hi = values[0::2] & 0x0F
    lo = values[1::2] & 0x0F
    return ((hi << 4) | lo).astype(np.uint8)


def unpack_nibbles(blob, count):
    packed = np.frombuffer(blob, dtype=np.uint8)
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = (packed >> 4) & 0x0F
    out[1::2] = packed & 0x0F
    return out[:count]


def quantize_symmetric_per_channel(array, bits):
    channels = array.shape[1]
    flat = array.transpose(1, 0, 2, 3).reshape(channels, -1)
    max_abs = np.max(np.abs(flat), axis=1)
    max_abs = np.maximum(max_abs, 1e-8)

    if bits == 8:
        qmax = 127.0
        scales = max_abs / qmax
        q = np.round(array / scales[None, :, None, None]).clip(-127, 127).astype(np.int8)
    elif bits == 4:
        qmax = 7.0
        scales = max_abs / qmax
        q = np.round(array / scales[None, :, None, None]).clip(-7, 7).astype(np.int8)
    else:
        raise ValueError(bits)
    return q, scales.astype(np.float32)


def dequantize_symmetric_per_channel(q, scales):
    return q.astype(np.float32) * scales[None, :, None, None]


def encode_fp16(latent_tchw):
    data = latent_tchw.astype(np.float16)
    blob = encode_lzma(data.tobytes(order="C"))
    return {
        "name": "fp16_lzma",
        "blob": blob,
        "shape": list(data.shape),
        "decode": lambda payload: np.frombuffer(decode_lzma(payload), dtype=np.float16).reshape(data.shape).astype(np.float32),
    }


def encode_delta_fp16(latent_tchw):
    delta = latent_tchw.copy()
    delta[1:] = latent_tchw[1:] - latent_tchw[:-1]
    delta = delta.astype(np.float16)
    blob = encode_lzma(delta.tobytes(order="C"))
    shape = list(delta.shape)

    def decode(payload):
        arr = np.frombuffer(decode_lzma(payload), dtype=np.float16).reshape(shape).astype(np.float32)
        arr = np.cumsum(arr, axis=0)
        return arr

    return {"name": "delta_fp16_lzma", "blob": blob, "shape": shape, "decode": decode}


def encode_delta_int(latent_tchw, bits):
    delta = latent_tchw.copy()
    delta[1:] = latent_tchw[1:] - latent_tchw[:-1]
    q, scales = quantize_symmetric_per_channel(delta, bits=bits)
    header = struct.pack(
        "<4I",
        delta.shape[0],
        delta.shape[1],
        delta.shape[2],
        delta.shape[3],
    ) + scales.tobytes(order="C")

    if bits == 8:
        payload = q.tobytes(order="C")

        def decode(payload_bytes):
            raw = decode_lzma(payload_bytes)
            shape = struct.unpack("<4I", raw[:16])
            scale_arr = np.frombuffer(raw[16 : 16 + 4 * shape[1]], dtype=np.float32)
            q_arr = np.frombuffer(raw[16 + 4 * shape[1] :], dtype=np.int8).reshape(shape)
            out = dequantize_symmetric_per_channel(q_arr, scale_arr)
            return np.cumsum(out, axis=0)

        name = "delta_int8_lzma"
    elif bits == 4:
        payload = pack_nibbles((q.astype(np.int16) + 8).astype(np.uint8).reshape(-1)).tobytes()

        def decode(payload_bytes):
            raw = decode_lzma(payload_bytes)
            shape = struct.unpack("<4I", raw[:16])
            scale_arr = np.frombuffer(raw[16 : 16 + 4 * shape[1]], dtype=np.float32)
            count = math.prod(shape)
            q_flat = unpack_nibbles(raw[16 + 4 * shape[1] :], count).astype(np.int16) - 8
            q_arr = q_flat.astype(np.int8).reshape(shape)
            out = dequantize_symmetric_per_channel(q_arr, scale_arr)
            return np.cumsum(out, axis=0)

        name = "delta_int4_lzma"
    else:
        raise ValueError(bits)

    blob = encode_lzma(header + payload)
    return {"name": name, "blob": blob, "shape": list(delta.shape), "decode": decode}


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

    raw_rgb_path = os.path.join(args.output_dir, "reference_uncompressed.rgb")
    with open(raw_rgb_path, "wb") as f:
        f.write(ref_frames.tobytes())

    schemes = [
        encode_fp16(latent_tchw),
        encode_delta_fp16(latent_tchw),
        encode_delta_int(latent_tchw, bits=8),
        encode_delta_int(latent_tchw, bits=4),
    ]

    results = []
    mp4_size = os.path.getsize(args.mp4_path)
    ref_raw_size = ref_frames.nbytes

    mp4_vs_ref = OrderedDict(
        name="mp4_h264_reference",
        compressed_bytes=mp4_size,
        ratio_vs_raw=ref_raw_size / mp4_size,
        psnr_vs_ref=psnr_uint8(mp4_frames, ref_frames),
        mae_vs_ref=mae_uint8(mp4_frames, ref_frames),
        smaller_than_mp4=False,
    )
    results.append(mp4_vs_ref)

    for scheme in schemes:
        out_path = os.path.join(args.output_dir, f"{scheme['name']}.bin")
        with open(out_path, "wb") as f:
            f.write(scheme["blob"])

        recon_tchw = scheme["decode"](scheme["blob"])
        recon_latent = tchw_to_tensor(recon_tchw)
        recon_frames = decode_latent_video(vae, recon_latent)

        recon_mp4_path = os.path.join(args.output_dir, f"{scheme['name']}_decoded.mp4")
        write_video(recon_mp4_path, recon_frames, fps=fps)

        compressed_size = len(scheme["blob"])
        result = OrderedDict(
            name=scheme["name"],
            compressed_bytes=compressed_size,
            compressed_kib=compressed_size / 1024.0,
            ratio_vs_raw_video=ref_raw_size / compressed_size,
            ratio_vs_latent_pt=os.path.getsize(args.latent_path) / compressed_size,
            smaller_than_mp4=compressed_size < mp4_size,
            psnr_vs_ref=psnr_uint8(recon_frames, ref_frames),
            mae_vs_ref=mae_uint8(recon_frames, ref_frames),
            decoded_mp4_bytes=os.path.getsize(recon_mp4_path),
            decoded_mp4_psnr_vs_ref=psnr_uint8(decode_video_frames(recon_mp4_path), ref_frames),
        )
        results.append(result)

    summary = OrderedDict(
        latent_pt_bytes=os.path.getsize(args.latent_path),
        latent_shape=list(latent.shape),
        latent_dtype=str(latent.dtype),
        reference_raw_video_bytes=ref_raw_size,
        reference_mp4_bytes=mp4_size,
        reference_mp4_psnr_vs_ref=mp4_vs_ref["psnr_vs_ref"],
        reference_mp4_mae_vs_ref=mp4_vs_ref["mae_vs_ref"],
        results=results,
    )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
