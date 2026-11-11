"""Bounded scheduling for native, coordinate-only VCF indexing."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
import struct
import zlib
from urllib.parse import urlsplit


_COMPRESSED_CHUNK_BYTES = 4 << 20
_EOF = (1 << 63) - 1


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


def _ranges(path, start):
    from .cyvcf2 import _coordinate_boundary, _IndexRemoteFile
    remote = b"://" in path
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


def _run_coordinate_batches(path, threads, workers, builder):
    from .cyvcf2 import _CoordinateReader
    ranges = iter(_ranges(path, builder.offset0))
    pending = deque()
    readers = set()
    offset = builder.offset0
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
        builder.finish(offset)
    finally:
        for reader in readers:
            reader.close()


def _build_parallel(path, destination, threads, workers):
    from .cyvcf2 import _CoordinateIndexBuilder
    _run_coordinate_batches(path, threads, workers, _CoordinateIndexBuilder(path, destination))


def _stream_index(path, index_format, threads, workers, output_fd):
    from .cyvcf2 import _StreamingCoordinateIndex
    first = _StreamingCoordinateIndex(path, index_format)
    _run_coordinate_batches(path, threads, workers, first)
    second = _StreamingCoordinateIndex(path, index_format, output_fd, first.names)
    _run_coordinate_batches(path, threads, workers, second)
    if first.fingerprint != second.fingerprint or first.total != second.total:
        raise OSError("index input changed between streaming passes")


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
