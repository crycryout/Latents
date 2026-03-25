import json
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from opensora.utils.latent_streaming import DecodedFrame, sample_to_uint8_frames


_PREFIX_STRUCT = struct.Struct("!IQ")


@dataclass
class ServerChunkEvent:
    chunk_index: int
    frame_start: int
    frame_count: int
    payload_bytes: int
    header_bytes: int
    wire_bytes: int
    send_start_s: float
    send_end_s: float


@dataclass
class ClientChunk:
    chunk_index: int
    frame_start: int
    frame_count: int
    tensor: torch.Tensor
    wire_dtype: torch.dtype
    payload_bytes: int
    header_bytes: int
    wire_bytes: int
    sent_at_s: float
    received_at_s: float


@dataclass
class ClientChunkEvent:
    chunk_index: int
    frame_start: int
    frame_count: int
    payload_bytes: int
    header_bytes: int
    wire_bytes: int
    sent_at_s: float
    received_at_s: float
    decode_ready_at_s: float | None = None
    chunk_decode_ms: float | None = None


def torch_dtype_to_name(dtype):
    mapping = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]


def torch_dtype_to_numpy(dtype):
    mapping = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.bfloat16: None,
    }
    np_dtype = mapping.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported wire dtype for numpy serialization: {dtype}")
    return np_dtype


def name_to_torch_dtype(name):
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[name]


def _send_rate_limited(sock, packet, chunk_bytes, bandwidth_bytes_per_s):
    if bandwidth_bytes_per_s is None:
        sock.sendall(packet)
        return

    offset = 0
    while offset < len(packet):
        piece = packet[offset : offset + chunk_bytes]
        sock.sendall(piece)
        offset += len(piece)
        time.sleep(len(piece) / bandwidth_bytes_per_s)


def _send_packet(sock, header, payload, chunk_bytes, bandwidth_bytes_per_s):
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = _PREFIX_STRUCT.pack(len(header_bytes), len(payload))
    packet = prefix + header_bytes + payload
    _send_rate_limited(sock, packet, chunk_bytes=chunk_bytes, bandwidth_bytes_per_s=bandwidth_bytes_per_s)
    return len(header_bytes), len(packet)


def _recv_exact(sock, size):
    parts = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving stream data")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _recv_packet(sock):
    prefix = _recv_exact(sock, _PREFIX_STRUCT.size)
    header_size, payload_size = _PREFIX_STRUCT.unpack(prefix)
    header = json.loads(_recv_exact(sock, header_size).decode("utf-8"))
    payload = _recv_exact(sock, payload_size)
    return header, payload, header_size, _PREFIX_STRUCT.size + header_size + payload_size


class SocketLatentServer:
    def __init__(
        self,
        latent,
        fps,
        host="127.0.0.1",
        port=0,
        chunk_frames=1,
        wire_dtype=torch.float16,
        socket_chunk_bytes=4096,
        bandwidth_mbps=None,
        per_chunk_delay_ms=0.0,
    ):
        self.latent = latent.contiguous()
        self.fps = fps
        self.host = host
        self.port = port
        self.chunk_frames = chunk_frames
        self.wire_dtype = wire_dtype
        self.socket_chunk_bytes = socket_chunk_bytes
        self.bandwidth_mbps = bandwidth_mbps
        self.per_chunk_delay_ms = per_chunk_delay_ms

        self.events = []
        self.total_payload_bytes = 0
        self.total_header_bytes = 0
        self.total_wire_bytes = 0
        self._thread = None
        self._ready = threading.Event()
        self._error = None
        self._listening_socket = None
        self.send_started_at_s = None
        self.send_finished_at_s = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Server already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait_ready(self, timeout=5.0):
        if not self._ready.wait(timeout):
            raise TimeoutError("Server did not become ready in time")
        if self._error is not None:
            raise self._error

    def join(self):
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise self._error

    def _run(self):
        start_perf = time.perf_counter()
        bandwidth_bytes_per_s = None
        if self.bandwidth_mbps is not None:
            bandwidth_bytes_per_s = self.bandwidth_mbps * 1_000_000.0 / 8.0

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listening_socket = server_sock
        try:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.host, self.port))
            server_sock.listen(1)
            self.port = server_sock.getsockname()[1]
            self._ready.set()

            conn, _ = server_sock.accept()
            with conn:
                self.send_started_at_s = time.perf_counter() - start_perf
                total_frames = self.latent.shape[1]
                wire_np_dtype = torch_dtype_to_numpy(self.wire_dtype)
                for chunk_index, frame_start in enumerate(range(0, total_frames, self.chunk_frames)):
                    frame_end = min(frame_start + self.chunk_frames, total_frames)
                    chunk = self.latent[:, frame_start:frame_end].to(dtype=self.wire_dtype).contiguous()
                    payload = chunk.numpy().astype(wire_np_dtype, copy=False).tobytes(order="C")
                    header = {
                        "message_type": "chunk",
                        "chunk_index": chunk_index,
                        "frame_start": frame_start,
                        "frame_count": frame_end - frame_start,
                        "shape": list(chunk.shape),
                        "dtype": torch_dtype_to_name(self.wire_dtype),
                        "fps": self.fps,
                        "sent_at_s": time.perf_counter() - start_perf,
                    }

                    if self.per_chunk_delay_ms > 0:
                        time.sleep(self.per_chunk_delay_ms / 1000.0)

                    send_start_s = time.perf_counter() - start_perf
                    header_bytes, wire_bytes = _send_packet(
                        conn,
                        header=header,
                        payload=payload,
                        chunk_bytes=self.socket_chunk_bytes,
                        bandwidth_bytes_per_s=bandwidth_bytes_per_s,
                    )
                    send_end_s = time.perf_counter() - start_perf
                    event = ServerChunkEvent(
                        chunk_index=chunk_index,
                        frame_start=frame_start,
                        frame_count=frame_end - frame_start,
                        payload_bytes=len(payload),
                        header_bytes=header_bytes,
                        wire_bytes=wire_bytes,
                        send_start_s=send_start_s,
                        send_end_s=send_end_s,
                    )
                    self.events.append(event)
                    self.total_payload_bytes += len(payload)
                    self.total_header_bytes += header_bytes
                    self.total_wire_bytes += wire_bytes

                eos_header = {"message_type": "eos", "sent_at_s": time.perf_counter() - start_perf}
                eos_header_bytes, eos_wire_bytes = _send_packet(
                    conn,
                    header=eos_header,
                    payload=b"",
                    chunk_bytes=self.socket_chunk_bytes,
                    bandwidth_bytes_per_s=bandwidth_bytes_per_s,
                )
                self.total_header_bytes += eos_header_bytes
                self.total_wire_bytes += eos_wire_bytes
                self.send_finished_at_s = time.perf_counter() - start_perf
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            server_sock.close()


class NetworkLatentStreamDecoder:
    def __init__(self, vae, device, decode_dtype=torch.float16, max_chunk_buffer=8, max_frame_buffer=8):
        self.vae = vae
        self.device = device
        self.decode_dtype = decode_dtype
        self.chunk_queue = queue.Queue(maxsize=max_chunk_buffer)
        self.frame_queue = queue.Queue(maxsize=max_frame_buffer)
        self.chunk_events = []
        self.total_payload_bytes = 0
        self.total_header_bytes = 0
        self.total_wire_bytes = 0
        self._receiver_thread = None
        self._decoder_thread = None
        self._error = None
        self._chunk_sentinel = object()
        self._frame_sentinel = object()
        self._start_perf = None
        self.first_chunk_received_at_s = None
        self.last_chunk_received_at_s = None

    def start(self, sock):
        if self._receiver_thread is not None or self._decoder_thread is not None:
            raise RuntimeError("Client decoder already started")
        self._start_perf = time.perf_counter()
        self._receiver_thread = threading.Thread(target=self._recv_worker, args=(sock,), daemon=True)
        self._decoder_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self._receiver_thread.start()
        self._decoder_thread.start()

    def _recv_worker(self, sock):
        try:
            with sock:
                while True:
                    header, payload, header_bytes, wire_bytes = _recv_packet(sock)
                    message_type = header["message_type"]
                    now_s = time.perf_counter() - self._start_perf
                    if message_type == "eos":
                        break

                    wire_dtype = name_to_torch_dtype(header["dtype"])
                    np_dtype = torch_dtype_to_numpy(wire_dtype)
                    array = np.frombuffer(payload, dtype=np_dtype).copy().reshape(header["shape"])
                    tensor = torch.from_numpy(array)
                    chunk = ClientChunk(
                        chunk_index=header["chunk_index"],
                        frame_start=header["frame_start"],
                        frame_count=header["frame_count"],
                        tensor=tensor,
                        wire_dtype=wire_dtype,
                        payload_bytes=len(payload),
                        header_bytes=header_bytes,
                        wire_bytes=wire_bytes,
                        sent_at_s=header["sent_at_s"],
                        received_at_s=now_s,
                    )
                    self.chunk_queue.put(chunk)
                    self.total_payload_bytes += len(payload)
                    self.total_header_bytes += header_bytes
                    self.total_wire_bytes += wire_bytes
                    if self.first_chunk_received_at_s is None:
                        self.first_chunk_received_at_s = now_s
                    self.last_chunk_received_at_s = now_s
        except Exception as exc:
            self._error = exc
        finally:
            self.chunk_queue.put(self._chunk_sentinel)

    def _decode_worker(self):
        try:
            while True:
                item = self.chunk_queue.get()
                if item is self._chunk_sentinel:
                    break

                latent_chunk = item.tensor.unsqueeze(0).to(device=self.device, dtype=self.decode_dtype)
                decode_start = time.perf_counter()
                with torch.no_grad():
                    decoded = self.vae.decode(latent_chunk)[0]
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                chunk_decode_ms = (time.perf_counter() - decode_start) * 1000.0
                ready_at_s = time.perf_counter() - self._start_perf

                event = ClientChunkEvent(
                    chunk_index=item.chunk_index,
                    frame_start=item.frame_start,
                    frame_count=item.frame_count,
                    payload_bytes=item.payload_bytes,
                    header_bytes=item.header_bytes,
                    wire_bytes=item.wire_bytes,
                    sent_at_s=item.sent_at_s,
                    received_at_s=item.received_at_s,
                    decode_ready_at_s=ready_at_s,
                    chunk_decode_ms=chunk_decode_ms,
                )
                self.chunk_events.append(event)

                frames = sample_to_uint8_frames(decoded)
                for offset, frame in enumerate(frames):
                    self.frame_queue.put(
                        DecodedFrame(
                            frame_index=item.frame_start + offset,
                            frame=frame,
                            ready_at=ready_at_s,
                            chunk_index=item.chunk_index,
                            chunk_decode_ms=chunk_decode_ms,
                        )
                    )
        except Exception as exc:
            self._error = exc
        finally:
            self.frame_queue.put(self._frame_sentinel)

    def get(self):
        item = self.frame_queue.get()
        if item is self._frame_sentinel:
            if self._error is not None:
                raise self._error
            return None
        return item

    def join(self):
        if self._receiver_thread is not None:
            self._receiver_thread.join()
        if self._decoder_thread is not None:
            self._decoder_thread.join()
        if self._error is not None:
            raise self._error
