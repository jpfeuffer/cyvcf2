"""Resource detection for automatic schema-scan concurrency."""

import os
import posixpath
import re
import sys
import warnings
import gzip
import struct
import time
from fractions import Fraction
from pathlib import Path


class _SchemaProgress:
    def __init__(self, path, callback, *, defer_size=False):
        from .cyvcf2 import _schema_input_size
        if not callable(callback):
            raise TypeError("progress must be callable or None")
        self.size = None if defer_size else _schema_input_size(path)
        self.callback = callback
        self.started = self.last_report = time.monotonic()
        self.processed = 0

    def update(self, processed, records, *, force=False, phase="scanning"):
        now = time.monotonic()
        self.processed = max(self.processed, min(processed, self.size))
        if not force and now - self.last_report < 1:
            return
        self.last_report = now
        elapsed = now - self.started
        if phase == "complete":
            self.processed = self.size
        rate = self.processed / elapsed if elapsed > 0 else 0
        self.callback({
            "phase": phase, "pass": 1, "passes": 1,
            "bytes_processed": self.processed, "total_bytes": self.size,
            "fraction": self.processed / self.size if self.size else 1.0,
            "records": records, "elapsed_seconds": elapsed,
            "bytes_per_second": rate,
            "eta_seconds": (self.size - self.processed) / rate if rate > 0 else None,
        })

    def poll(self, paths, ranges):
        processed = records = 0
        for path, (start, end, _) in zip(paths, ranges):
            with open(path, "rb") as source:
                offset, count = struct.unpack("<QQ", source.read(16))
            begin = max(0, start) >> 16
            limit = min(self.size, end >> 16)
            processed += max(0, min(offset, limit) - begin)
            records += count
        self.update(processed, records)


class _SchemaWorkerProgress:
    def __init__(self, path):
        self.path = path
        self.last_report = time.monotonic()

    def update(self, processed, records, *, force=False, phase="scanning"):
        now = time.monotonic()
        if force or now - self.last_report >= 1:
            with open(self.path, "r+b", buffering=0) as output:
                output.write(struct.pack("<QQ", processed, records))
            self.last_report = now


def _schema_progress_result(future, progress, paths, ranges):
    from concurrent.futures import wait
    while not future.done():
        wait([future], timeout=0.1)
        progress.poll(paths, ranges)
    progress.poll(paths, ranges)
    return future.result()


def _remote_split_ranges(path, workers, header_end, is_bcf, progress=None):
    from ._index import _next_block, _validate_input
    from .cyvcf2 import _IndexRemoteFile, _coordinate_boundary
    _validate_input(path)
    boundaries = []
    with _IndexRemoteFile(path) as source:
        if progress is not None and progress.size is None:
            progress.size = source.size
            progress.update(0, 0, force=True)
        for worker in range(1, workers):
            target = source.size * worker // workers
            if target <= header_end >> 16:
                continue
            block = _next_block(source, target, source.size)
            boundary = block << 16 if is_bcf else _coordinate_boundary(path, block)
            if boundary is not None and boundary > header_end:
                if not boundaries or boundary > boundaries[-1]:
                    boundaries.append(boundary)
    starts = [-1] + boundaries
    ends = boundaries + [(1 << 63) - 1]
    return [(start, end, start) for start, end in zip(starts, ends)]


def _index_split_offsets(path, source_size, workers, is_bcf=False):
    """Read CSI/TBI chunk boundaries, not coordinate-query intervals."""
    with open(path, "rb") as source:
        compressed = source.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rb") as source:
        data = source.read()
    cursor = 0

    def read(fmt):
        nonlocal cursor
        size = struct.calcsize(fmt)
        if cursor + size > len(data):
            raise OSError("truncated index: %s" % path)
        values = struct.unpack_from(fmt, data, cursor)
        cursor += size
        return values

    def skip(length):
        nonlocal cursor
        if length < 0 or cursor + length > len(data):
            raise OSError("invalid index length: %s" % path)
        cursor += length

    magic, = read("<4s")
    if magic == b"CSI\x01":
        shift, depth, auxiliary = read("<iii")
        if shift < 0 or depth < 0 or depth > 9 or shift + 3 * depth > 62:
            raise OSError("invalid CSI settings: %s" % path)
        if is_bcf and auxiliary:
            raise OSError("BCF requires a CSI index without tabix metadata")
        skip(auxiliary)
        references, = read("<i")
        csi = True
    elif magic == b"TBI\x01":
        if is_bcf:
            raise OSError("BCF requires a CSI index, not TBI")
        references, = read("<i")
        *_, names_length = read("<7i")
        skip(names_length)
        depth = 5
        csi = False
    else:
        raise OSError("invalid index magic: %s" % path)
    if references < 0 or references > len(data) // 4:
        raise OSError("invalid index reference count: %s" % path)
    metadata_bin = ((1 << (3 * (depth + 1))) - 1) // 7 + 1
    offsets = set()
    total_records = 0
    have_counts = True
    for _ in range(references):
        bins, = read("<i")
        if bins < 0 or bins > len(data) // 8:
            raise OSError("invalid index bin count: %s" % path)
        found_counts = False
        for _ in range(bins):
            bin_id, = read("<I")
            if csi:
                lower_offset, = read("<Q")
                if lower_offset >> 16 > source_size:
                    raise OSError("invalid index lower offset: %s" % path)
            chunks, = read("<i")
            if chunks < 0 or chunks > (len(data) - cursor) // 16:
                raise OSError("invalid index chunk count: %s" % path)
            if bin_id == metadata_bin and chunks != 2:
                raise OSError("invalid index metadata bin: %s" % path)
            for chunk in range(chunks):
                start, end = read("<QQ")
                if bin_id == metadata_bin:
                    if chunk == 1:
                        total_records += start + end
                        found_counts = True
                    continue
                if (bin_id >= metadata_bin or start > end or end >> 16 > source_size
                        or (end >> 16 == source_size and end & 0xffff)):
                    raise OSError("invalid index chunk offsets: %s" % path)
                offsets.update((start, end))
        if not csi:
            intervals, = read("<i")
            if intervals < 0 or intervals > (len(data) - cursor) // 8:
                raise OSError("invalid index interval count: %s" % path)
            for _ in range(intervals):
                offset, = read("<Q")
                if offset >> 16 > source_size:
                    raise OSError("invalid index interval offset: %s" % path)
        have_counts = have_counts and (found_counts or bins == 0)
    if len(data) - cursor == 8:
        unplaced, = read("<Q")
        total_records += unplaced
    elif cursor != len(data):
        raise OSError("unexpected index trailing data: %s" % path)
    # EOF is not a useful intermediate split. Preserve complete header-to-EOF
    # coverage even when an index is sparse or omits some records.
    if offsets:
        offsets.remove(max(offsets))
    offsets = sorted(offset for offset in offsets if offset > 0)
    boundaries = sorted(set(
        min(offsets, key=lambda offset: abs((offset >> 16) - source_size * i // workers))
        for i in range(1, workers)
    )) if offsets else []
    return boundaries, total_records if have_counts else None


def _read_resource(path):
    try:
        return Path(path).read_text().strip()
    except FileNotFoundError:
        return None
    except OSError as error:
        warnings.warn("Cannot read CPU limit %s: %s" % (path, error), RuntimeWarning)
        return None


def _parse_cpuset(value):
    cpus = set()
    for item in value.split(","):
        bounds = item.split("-")
        if len(bounds) == 1:
            start = end = int(bounds[0])
        elif len(bounds) == 2:
            start, end = map(int, bounds)
        else:
            raise ValueError("invalid CPU range")
        if start < 0 or end < start:
            raise ValueError("invalid CPU range")
        cpus.update(range(start, end + 1))
    return cpus


def _mount_path(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup_cpu_limits(membership, mountinfo):
    groups = {}
    for line in membership.splitlines():
        _, controllers, path = line.split(":", 2)
        for controller in controllers.split(","):
            groups[controller] = posixpath.normpath(path)

    quotas = []
    allowed_cpus = None
    for line in mountinfo.splitlines():
        before, after = line.split(" - ", 1)
        fields = before.split()
        filesystem, _, options = after.split()[:3]
        if filesystem == "cgroup2":
            controllers = [""]
        elif filesystem == "cgroup":
            controllers = sorted(set(options.split(",")) & {"cpu", "cpuset"})
        else:
            continue
        root = _mount_path(fields[3])
        mount = Path(_mount_path(fields[4]))
        for controller in controllers:
            if controller not in groups:
                continue
            group = groups[controller]
            if group == root or group == "/":
                directory = mount
            elif group.startswith(root.rstrip("/") + "/"):
                directory = mount / group[len(root):].lstrip("/")
            else:
                warnings.warn(
                    "Cannot locate %s cgroup %s under mount root %s"
                    % (controller or "v2", group, root), RuntimeWarning)
                continue
            while True:
                try:
                    if filesystem == "cgroup2":
                        quota = _read_resource(directory / "cpu.max")
                        if quota is not None:
                            amount, period = quota.split()
                            if amount != "max":
                                limit = Fraction(int(amount), int(period))
                                if limit <= 0:
                                    raise ValueError("non-positive CPU quota")
                                quotas.append(limit)
                        cpuset = _read_resource(directory / "cpuset.cpus.effective")
                    else:
                        cpuset = None
                        if controller == "cpu":
                            quota = _read_resource(directory / "cpu.cfs_quota_us")
                            period = _read_resource(directory / "cpu.cfs_period_us")
                            if quota is not None and period is not None and int(quota) != -1:
                                limit = Fraction(int(quota), int(period))
                                if limit <= 0:
                                    raise ValueError("non-positive CPU quota")
                                quotas.append(limit)
                        else:
                            cpuset = _read_resource(directory / "cpuset.cpus")
                    if cpuset:
                        cpus = _parse_cpuset(cpuset)
                        allowed_cpus = cpus if allowed_cpus is None else allowed_cpus & cpus
                except (ValueError, ZeroDivisionError) as error:
                    warnings.warn(
                        "Invalid CPU limit in %s: %s" % (directory, error), RuntimeWarning)
                if directory == mount:
                    break
                directory = directory.parent
    return quotas, allowed_cpus


def _available_cpu_count():
    count_cpu = getattr(os, "process_cpu_count", os.cpu_count)
    count = count_cpu()
    if count is None:
        warnings.warn("Cannot determine CPU count; using one CPU", RuntimeWarning)
        count = 1
    allowed_cpus = None
    if hasattr(os, "sched_getaffinity"):
        try:
            allowed_cpus = set(os.sched_getaffinity(0))
        except OSError as error:
            warnings.warn("Cannot read CPU affinity: %s" % error, RuntimeWarning)
    quotas = []
    if sys.platform.startswith("linux"):
        membership = _read_resource("/proc/self/cgroup")
        mountinfo = _read_resource("/proc/self/mountinfo")
        if membership is not None and mountinfo is not None:
            quotas, cpuset = _cgroup_cpu_limits(membership, mountinfo)
            if cpuset is not None:
                allowed_cpus = cpuset if allowed_cpus is None else allowed_cpus & cpuset
        else:
            warnings.warn(
                "Cannot discover cgroup CPU limits from /proc/self; "
                "using CPU count and affinity only", RuntimeWarning)
    if allowed_cpus is not None:
        count = min(count, len(allowed_cpus))
    if quotas:
        count = min(count, int(min(quotas)))
    return max(1, count)
