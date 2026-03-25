import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DecodedFrame:
    frame_index: int
    frame: np.ndarray
    ready_at: float
    chunk_index: int
    chunk_decode_ms: float


@dataclass
class PlaybackEvent:
    frame_index: int
    ready_at: float
    scheduled_at: float
    presented_at: float
    stall_before_present_s: float
    queue_size_after_pop: int


def sample_to_uint8_frames(sample):
    sample = sample.clamp(-1, 1)
    sample = sample.add(1).div(2).mul(255).add_(0.5).clamp_(0, 255)
    return sample.permute(1, 2, 3, 0).to("cpu", torch.uint8).numpy()


def decode_full_latent(vae, latent, device, dtype):
    with torch.no_grad():
        sample = vae.decode(latent.unsqueeze(0).to(device=device, dtype=dtype))[0]
    return sample_to_uint8_frames(sample)


class LatentStreamDecoder:
    def __init__(self, vae, device, dtype=torch.float16, chunk_frames=1, max_buffer_frames=16):
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.chunk_frames = chunk_frames
        self.frame_queue = queue.Queue(maxsize=max_buffer_frames)
        self._thread = None
        self._error = None
        self._sentinel = object()

    def start(self, latent):
        if self._thread is not None:
            raise RuntimeError("Decoder already started")
        self._thread = threading.Thread(target=self._decode_worker, args=(latent,), daemon=True)
        self._thread.start()

    def _decode_worker(self, latent):
        try:
            start_time = time.perf_counter()
            total_frames = latent.shape[1]
            for chunk_index, frame_start in enumerate(range(0, total_frames, self.chunk_frames)):
                frame_end = min(frame_start + self.chunk_frames, total_frames)
                latent_chunk = latent[:, frame_start:frame_end].unsqueeze(0).to(device=self.device, dtype=self.dtype)

                decode_start = time.perf_counter()
                with torch.no_grad():
                    decoded = self.vae.decode(latent_chunk)[0]
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                chunk_decode_ms = (time.perf_counter() - decode_start) * 1000.0

                frames = sample_to_uint8_frames(decoded)
                ready_at = time.perf_counter() - start_time
                for offset, frame in enumerate(frames):
                    item = DecodedFrame(
                        frame_index=frame_start + offset,
                        frame=frame,
                        ready_at=ready_at,
                        chunk_index=chunk_index,
                        chunk_decode_ms=chunk_decode_ms,
                    )
                    self.frame_queue.put(item)
        except Exception as exc:
            self._error = exc
        finally:
            self.frame_queue.put(self._sentinel)

    def get(self):
        item = self.frame_queue.get()
        if item is self._sentinel:
            if self._error is not None:
                raise self._error
            return None
        return item

    def join(self):
        if self._thread is not None:
            self._thread.join()


class PlaybackSimulator:
    def __init__(self, fps):
        self.fps = fps
        self.frame_interval = 1.0 / fps

    def run(self, decoder):
        playback_clock = 0.0
        events = []
        frames = []
        underflow_count = 0
        total_stall_s = 0.0

        while True:
            item = decoder.get()
            if item is None:
                break

            scheduled_at = playback_clock
            presented_at = max(playback_clock, item.ready_at)
            stall = max(0.0, item.ready_at - playback_clock)
            if stall > 0:
                underflow_count += 1
                total_stall_s += stall

            events.append(
                PlaybackEvent(
                    frame_index=item.frame_index,
                    ready_at=item.ready_at,
                    scheduled_at=scheduled_at,
                    presented_at=presented_at,
                    stall_before_present_s=stall,
                    queue_size_after_pop=decoder.frame_queue.qsize(),
                )
            )
            frames.append(item.frame)
            playback_clock = presented_at + self.frame_interval

        return {
            "events": events,
            "frames": np.stack(frames, axis=0) if frames else np.empty((0,)),
            "underflow_count": underflow_count,
            "total_stall_s": total_stall_s,
            "final_playback_time_s": playback_clock,
        }
