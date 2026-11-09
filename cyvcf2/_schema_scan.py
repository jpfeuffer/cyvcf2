"""Resource detection for automatic schema-scan concurrency."""

import os
import posixpath
import re
import sys
import warnings
from fractions import Fraction
from pathlib import Path


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
