import argparse
import json
import os
import socket
import statistics
import threading
import time

import numpy as np
import psutil
import torch

from opensora.registry import MODELS, build_module
from opensora.utils.latent_streaming import PlaybackSimulator, decode_full_latent
from opensora.utils.misc import to_torch_dtype
from opensora.utils.network_latent_streaming import NetworkLatentStreamDecoder, SocketLatentServer


class ProcessMonitor:
    def __init__(self, sample_interval_s=0.05):
        self.sample_interval_s = sample_interval_s
        self.samples = []
        self._thread = None
        self._stop = threading.Event()
        self._process = psutil.Process(os.getpid())

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(
                {
                    "t": time.perf_counter(),
                    "cpu_percent": self._process.cpu_percent(interval=None),
                    "rss_mb": self._process.memory_info().rss / (1024 * 1024),
                }
            )
            time.sleep(self.sample_interval_s)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def summary(self):
        cpu_values = [sample["cpu_percent"] for sample in self.samples]
        rss_values = [sample["rss_mb"] for sample in self.samples]
        result = {
            "process_cpu_percent_avg": float(statistics.mean(cpu_values)) if cpu_values else 0.0,
            "process_cpu_percent_max": float(max(cpu_values)) if cpu_values else 0.0,
            "process_rss_mb_max": float(max(rss_values)) if rss_values else 0.0,
        }
        if torch.cuda.is_available():
            result["cuda_max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            result["cuda_max_memory_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 * 1024))
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-path", required=True, type=str)
    parser.add_argument("--vae-pretrained", default="./pretrained_models/sd-vae-ft-ema", type=str)
    parser.add_argument("--decode-dtype", default="fp16", type=str)
    parser.add_argument("--wire-dtype", default="fp16", type=str, choices=["fp16", "fp32"])
    parser.add_argument("--chunk-frames", default=1, type=int)
    parser.add_argument("--max-chunk-buffer", default=8, type=int)
    parser.add_argument("--max-frame-buffer", default=8, type=int)
    parser.add_argument("--fps", default=None, type=int)
    parser.add_argument("--repeat-times", default=1, type=int)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=0, type=int)
    parser.add_argument("--bandwidth-mbps", default=None, type=float)
    parser.add_argument("--per-chunk-delay-ms", default=0.0, type=float)
    parser.add_argument("--socket-chunk-bytes", default=4096, type=int)
    parser.add_argument("--save-dir", default=None, type=str)
    parser.add_argument("--label", default="run", type=str)
    parser.add_argument("--verify-full-decode", action="store_true")
    return parser.parse_args()


def _steady_state_fps(values):
    if len(values) < 2:
        return {"mean_fps": 0.0, "median_fps": 0.0}
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    positive = [delta for delta in deltas if delta > 0]
    if not positive:
        return {"mean_fps": 0.0, "median_fps": 0.0}
    return {
        "mean_fps": float(1.0 / statistics.mean(positive)),
        "median_fps": float(1.0 / statistics.median(positive)),
    }


def _save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decode_dtype = to_torch_dtype(args.decode_dtype)
    wire_dtype = to_torch_dtype(args.wire_dtype)

    payload = torch.load(args.latent_path, map_location="cpu")
    latent = payload["latent"].contiguous()
    if args.repeat_times > 1:
        latent = latent.repeat(1, args.repeat_times, 1, 1).contiguous()
    fps = args.fps if args.fps is not None else int(payload.get("fps", 8))
    video_duration_s = latent.shape[1] / fps

    vae = build_module(dict(type="VideoAutoencoderKL", from_pretrained=args.vae_pretrained), MODELS).to(
        device, decode_dtype
    ).eval()

    server = SocketLatentServer(
        latent=latent,
        fps=fps,
        host=args.host,
        port=args.port,
        chunk_frames=args.chunk_frames,
        wire_dtype=wire_dtype,
        socket_chunk_bytes=args.socket_chunk_bytes,
        bandwidth_mbps=args.bandwidth_mbps,
        per_chunk_delay_ms=args.per_chunk_delay_ms,
    )
    client = NetworkLatentStreamDecoder(
        vae=vae,
        device=device,
        decode_dtype=decode_dtype,
        max_chunk_buffer=args.max_chunk_buffer,
        max_frame_buffer=args.max_frame_buffer,
    )
    player = PlaybackSimulator(fps=fps)
    monitor = ProcessMonitor()

    monitor.start()
    server.start()
    server.wait_ready()

    experiment_start = time.perf_counter()
    sock = socket.create_connection((args.host, server.port))
    client.start(sock)
    playback_result = player.run(client)
    client.join()
    server.join()
    monitor.stop()
    experiment_wall_s = time.perf_counter() - experiment_start

    chunk_events = client.chunk_events
    ready_times = [event.ready_at for event in playback_result["events"]]
    chunk_receive_times = [event.received_at_s for event in chunk_events]
    network_latencies_ms = [(event.received_at_s - event.sent_at_s) * 1000.0 for event in chunk_events]
    decode_ready_times = [event.decode_ready_at_s for event in chunk_events if event.decode_ready_at_s is not None]
    chunk_decode_ms_values = [event.chunk_decode_ms for event in chunk_events if event.chunk_decode_ms is not None]

    send_duration_s = (
        server.send_finished_at_s - server.send_started_at_s
        if server.send_started_at_s is not None and server.send_finished_at_s is not None
        else 0.0
    )
    receive_duration_s = (
        client.last_chunk_received_at_s - client.first_chunk_received_at_s
        if client.first_chunk_received_at_s is not None and client.last_chunk_received_at_s is not None
        else 0.0
    )

    payload_required_mbps = float((server.total_payload_bytes * 8.0) / video_duration_s / 1_000_000.0)
    wire_required_mbps = float((server.total_wire_bytes * 8.0) / video_duration_s / 1_000_000.0)

    steady_decode_fps = _steady_state_fps(ready_times)
    steady_receive_fps = _steady_state_fps(chunk_receive_times)
    startup_only_underflow = (
        playback_result["underflow_count"] == 1
        and len(playback_result["events"]) > 0
        and playback_result["events"][0].stall_before_present_s > 0
        and all(event.stall_before_present_s == 0.0 for event in playback_result["events"][1:])
    )

    summary = {
        "label": args.label,
        "latent_path": args.latent_path,
        "latent_shape": list(latent.shape),
        "fps": fps,
        "repeat_times": args.repeat_times,
        "video_duration_s": video_duration_s,
        "chunk_frames": args.chunk_frames,
        "wire_dtype": args.wire_dtype,
        "decode_dtype": args.decode_dtype,
        "bandwidth_mbps_limit": args.bandwidth_mbps,
        "per_chunk_delay_ms": args.per_chunk_delay_ms,
        "socket_chunk_bytes": args.socket_chunk_bytes,
        "experiment_wall_s": experiment_wall_s,
        "server": {
            "chunks_sent": len(server.events),
            "payload_bytes": server.total_payload_bytes,
            "header_bytes": server.total_header_bytes,
            "wire_bytes": server.total_wire_bytes,
            "send_duration_s": send_duration_s,
            "payload_required_mbps_for_playback": payload_required_mbps,
            "wire_required_mbps_for_playback": wire_required_mbps,
            "achieved_wire_mbps": (
                float(server.total_wire_bytes * 8.0 / send_duration_s / 1_000_000.0) if send_duration_s > 0 else 0.0
            ),
        },
        "client": {
            "chunks_received": len(chunk_events),
            "payload_bytes": client.total_payload_bytes,
            "header_bytes": client.total_header_bytes,
            "wire_bytes": client.total_wire_bytes,
            "receive_duration_s": receive_duration_s,
            "achieved_wire_mbps": (
                float(client.total_wire_bytes * 8.0 / receive_duration_s / 1_000_000.0) if receive_duration_s > 0 else 0.0
            ),
            "avg_network_latency_ms": (
                float(statistics.mean(network_latencies_ms)) if network_latencies_ms else 0.0
            ),
            "max_network_latency_ms": float(max(network_latencies_ms)) if network_latencies_ms else 0.0,
            "steady_chunk_receive_fps_mean": steady_receive_fps["mean_fps"],
            "steady_chunk_receive_fps_median": steady_receive_fps["median_fps"],
        },
        "decode": {
            "frames_decoded": int(playback_result["frames"].shape[0]),
            "first_frame_ready_s": float(ready_times[0]) if ready_times else 0.0,
            "last_frame_ready_s": float(ready_times[-1]) if ready_times else 0.0,
            "steady_frame_ready_fps_mean": steady_decode_fps["mean_fps"],
            "steady_frame_ready_fps_median": steady_decode_fps["median_fps"],
            "chunk_decode_ms_mean": float(statistics.mean(chunk_decode_ms_values)) if chunk_decode_ms_values else 0.0,
            "chunk_decode_ms_max": float(max(chunk_decode_ms_values)) if chunk_decode_ms_values else 0.0,
            "chunks_decoded": len(chunk_decode_ms_values),
            "last_chunk_decode_ready_s": float(decode_ready_times[-1]) if decode_ready_times else 0.0,
        },
        "playback": {
            "underflow_count": playback_result["underflow_count"],
            "startup_only_underflow": startup_only_underflow,
            "total_stall_s": playback_result["total_stall_s"],
            "first_frame_latency_s": (float(ready_times[0]) if ready_times else 0.0),
            "final_playback_time_s": playback_result["final_playback_time_s"],
            "effective_fps": (
                float(playback_result["frames"].shape[0] / playback_result["final_playback_time_s"])
                if playback_result["final_playback_time_s"] > 0
                else 0.0
            ),
        },
        "system": monitor.summary(),
    }

    summary["real_time_supported"] = bool(
        summary["decode"]["steady_frame_ready_fps_mean"] >= fps and summary["playback"]["startup_only_underflow"]
    )

    if args.verify_full_decode:
        reference_latent = latent.to(dtype=wire_dtype)
        reference_frames = decode_full_latent(vae, reference_latent, device=device, dtype=decode_dtype)
        streamed_frames = playback_result["frames"]
        identical = np.array_equal(streamed_frames, reference_frames)
        summary["verify_full_decode"] = {
            "identical": bool(identical),
            "max_abs_diff": (
                int(np.max(np.abs(streamed_frames.astype(np.int16) - reference_frames.astype(np.int16))))
                if streamed_frames.size > 0
                else 0
            ),
        }

    server_events_payload = [
        {
            "chunk_index": event.chunk_index,
            "frame_start": event.frame_start,
            "frame_count": event.frame_count,
            "payload_bytes": event.payload_bytes,
            "header_bytes": event.header_bytes,
            "wire_bytes": event.wire_bytes,
            "send_start_s": event.send_start_s,
            "send_end_s": event.send_end_s,
        }
        for event in server.events
    ]
    client_chunk_payload = [
        {
            "chunk_index": event.chunk_index,
            "frame_start": event.frame_start,
            "frame_count": event.frame_count,
            "payload_bytes": event.payload_bytes,
            "header_bytes": event.header_bytes,
            "wire_bytes": event.wire_bytes,
            "sent_at_s": event.sent_at_s,
            "received_at_s": event.received_at_s,
            "decode_ready_at_s": event.decode_ready_at_s,
            "chunk_decode_ms": event.chunk_decode_ms,
        }
        for event in chunk_events
    ]
    playback_events_payload = [
        {
            "frame_index": event.frame_index,
            "ready_at": event.ready_at,
            "scheduled_at": event.scheduled_at,
            "presented_at": event.presented_at,
            "stall_before_present_s": event.stall_before_present_s,
            "queue_size_after_pop": event.queue_size_after_pop,
        }
        for event in playback_result["events"]
    ]

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.latent_path))[0]
        tag = f"{args.label}_chunk{args.chunk_frames}_{args.wire_dtype}_{args.decode_dtype}"
        summary_path = os.path.join(args.save_dir, f"{base}_{tag}_summary.json")
        server_events_path = os.path.join(args.save_dir, f"{base}_{tag}_server_events.json")
        client_chunks_path = os.path.join(args.save_dir, f"{base}_{tag}_client_chunks.json")
        playback_events_path = os.path.join(args.save_dir, f"{base}_{tag}_playback_events.json")
        _save_json(summary_path, summary)
        _save_json(server_events_path, server_events_payload)
        _save_json(client_chunks_path, client_chunk_payload)
        _save_json(playback_events_path, playback_events_payload)
        print(f"Summary saved to: {summary_path}")
        print(f"Server events saved to: {server_events_path}")
        print(f"Client chunk events saved to: {client_chunks_path}")
        print(f"Playback events saved to: {playback_events_path}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
