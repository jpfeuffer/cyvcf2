"""Bounded scheduling for native, coordinate-only VCF indexing."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
import struct
import time
import zlib
from urllib.parse import urlsplit


_COMPRESSED_CHUNK_BYTES = 4 << 20
_EOF = (1 << 63) - 1


class _IndexProgress:
    def __init__(self, path, callback, passes):
        from .cyvcf2 import _IndexRemoteFile
        if not callable(callback):
            raise TypeError("progress must be callable or None")
        if b"://" in path:
            with _IndexRemoteFile(path) as stream:
                self.size = stream.size
        else:
            self.size = os.path.getsize(path)
        self.callback = callback
        self.passes = passes
        self.started = time.monotonic()
        self.last_report = self.started
        self.completed = 0

    def report(self, pass_number, offset=0, *, phase=None, force=False):
        now = time.monotonic()
        completed = (pass_number - 1) * self.size + min(offset >> 16, self.size)
        self.completed = max(self.completed, completed)
        if not force and now - self.last_report < 1:
            return
        self.last_report = now
        total = self.size * self.passes
        elapsed = now - self.started
        rate = self.completed / elapsed if elapsed > 0 else 0
        if phase == "complete":
            self.completed = total
            rate = total / elapsed if elapsed > 0 else 0
        self.callback({
            "phase": phase or ("scanning" if self.passes == 2 and pass_number == 1 else "indexing"),
            "pass": pass_number,
            "passes": self.passes,
            "bytes_processed": self.completed,
            "total_bytes": total,
            "fraction": self.completed / total if total else 1.0,
            "elapsed_seconds": elapsed,
            "bytes_per_second": rate,
            "eta_seconds": (total - self.completed) / rate if rate > 0 else None,
        })


def _next_block(stream, target, size):
    stream.seek(target)
    window = stream.read(2 * 65536)
    cursor = 0
    while True:
        cursor = window.find(b"\x1f\x8b\x08\x04", cursor)
        if cursor < 0:
            raise OSError("cannot find a valid BGZF block near coordinate split")
        if cursor + 12 > len(window):
            raise OSError("truncated BGZF split header")
        extra_length, = struct.unpack_from("<H", window, cursor + 10)
        extra_end = cursor + 12 + extra_length
        position = cursor + 12
        block_size = None
        while position + 4 <= min(extra_end, len(window)):
            length, = struct.unpack_from("<H", window, position + 2)
            if position + 4 + length > min(extra_end, len(window)):
                break
            if window[position:position + 2] == b"BC" and length == 2:
                block_size = struct.unpack_from("<H", window, position + 4)[0] + 1
                break
            position += 4 + length
        if (block_size is not None and block_size >= extra_length + 20
                and cursor + block_size <= len(window)):
            try:
                zlib.decompress(window[cursor:cursor + block_size], wbits=31)
            except zlib.error:
                pass
            else:
                return target + cursor
        cursor += 1
        if target + cursor >= size:
            raise OSError("no BGZF boundary before EOF")


def _ranges(path, start, workers=None):
    from .cyvcf2 import _coordinate_boundary, _IndexRemoteFile
    remote = b"://" in path
    if remote and workers is not None:
        from ._schema_scan import _remote_split_ranges
        for begin, end, _ in _remote_split_ranges(path, workers, start, False):
            yield max(start, begin), end
        return
    with (_IndexRemoteFile(path) if remote else open(path, "rb")) as stream:
        size = stream.size if remote else os.path.getsize(path)
        while True:
            target = (start >> 16) + _COMPRESSED_CHUNK_BYTES
            if target >= size:
                yield start, _EOF
                return
            block = _next_block(stream, target, size)
            boundary = _coordinate_boundary(path, block)
            if boundary is None:
                yield start, _EOF
                return
            if boundary <= start:
                raise OSError("coordinate split failed to advance")
            yield start, boundary
            start = boundary


def _run_coordinate_batches(path, threads, workers, builder, progress=None, pass_number=1):
    from .cyvcf2 import _CoordinateReader
    ranges = iter(
        _ranges(path, builder.offset0, workers) if b"://" in path
        else _ranges(path, builder.offset0))
    pending = deque()
    readers = set()
    offset = builder.offset0
    if progress is not None:
        progress.report(pass_number, force=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def submit_range():
                limits = next(ranges, None)
                if limits is None:
                    return
                reader = _CoordinateReader(path, *limits, threads)
                readers.add(reader)
                pending.append((reader, pool.submit(reader.read)))

            for _ in range(workers):
                submit_range()
            while pending:
                reader, future = pending.popleft()
                batch = future.result()
                if batch.done:
                    reader.close()
                    readers.remove(reader)
                    submit_range()
                else:
                    pending.appendleft((reader, pool.submit(reader.read)))
                builder.add(batch)
                offset = batch.offset
                if progress is not None:
                    progress.report(pass_number, offset)
        builder.finish(offset)
    finally:
        for reader in readers:
            reader.close()


def _build_parallel(path, destination, threads, workers, progress=None):
    from .cyvcf2 import _CoordinateIndexBuilder
    reporter = _IndexProgress(path, progress, 1) if progress is not None else None
    builder = _CoordinateIndexBuilder(path, destination)
    _run_coordinate_batches(path, threads, workers, builder, reporter)
    if reporter is not None:
        reporter.report(1, phase="complete", force=True)
    builder.publish()


def _stream_index(path, index_format, threads, workers, output_fd, progress=None):
    from .cyvcf2 import _StreamingCoordinateIndex
    reporter = _IndexProgress(path, progress, 2) if progress is not None else None
    first = _StreamingCoordinateIndex(path, index_format)
    _run_coordinate_batches(path, threads, workers, first, reporter)
    second = _StreamingCoordinateIndex(path, index_format, output_fd, first.names)
    _run_coordinate_batches(path, threads, workers, second, reporter, 2)
    if first.fingerprint != second.fingerprint or first.total != second.total:
        raise OSError("index input changed between streaming passes")
    if reporter is not None:
        reporter.report(2, phase="complete", force=True)


def build_index_stream(path, output, *, index_format="csi", threads=0, workers=1, progress=None):
    """Write a bounded-memory CSI/TBI index to a writable binary stream.

    Requires a seekable BGZF VCF, bundled HTSlib and POSIX /dev/fd. Reads the
    input twice; no local staging files or full in-memory index are used.
    The caller owns output and must discard partial bytes on any exception.
    This function does not flush, close or publish the caller's stream.
    progress(event), if supplied, receives approximate byte progress and ETA
    at most once per second plus pass transitions/completion. It runs in the
    producer thread; callback exceptions abort construction. Completion means
    index construction, not output flush or remote publication.
    """
    from .cyvcf2 import _build_index_to_fd
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb", buffering=0) as source, os.fdopen(write_fd, "wb", buffering=0) as sink:
        def produce():
            with sink:
                _build_index_to_fd(
                    path, sink.fileno(), index_format=index_format,
                    threads=threads, workers=workers,
                    progress=progress,
                )

        with ThreadPoolExecutor(max_workers=1) as pool:
            producer = pool.submit(produce)
            try:
                while data := source.read(1 << 20):
                    pending_bytes = memoryview(data)
                    while pending_bytes:
                        written = output.write(pending_bytes)
                        if not isinstance(written, int) or not 0 < written <= len(pending_bytes):
                            raise OSError("index output stream failed to write bytes")
                        pending_bytes = pending_bytes[written:]
                producer.result()
            finally:
                source.close()


def _validate_input(path):
    if b"://" not in path:
        if path == b"-":
            raise ValueError("index creation requires a local file or a seekable remote input")
        return False
    url = urlsplit(os.fsdecode(path))
    if url.scheme not in ("s3", "s3+http", "s3+https", "http", "https"):
        raise ValueError("index input must be local, S3, or HTTP(S)")
    if not url.netloc or not url.path or url.username or url.query or url.fragment:
        raise ValueError("remote index input requires a bucket/host and key, without credentials or query strings")
    return True
