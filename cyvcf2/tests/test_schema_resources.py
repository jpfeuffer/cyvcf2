import os
from fractions import Fraction

import pytest

from cyvcf2 import _schema_scan as resources


@pytest.mark.parametrize("value,expected", [
    ("0", {0}),
    ("0-2,5,8-9", {0, 1, 2, 5, 8, 9}),
])
def test_cpuset(value, expected):
    assert resources._parse_cpuset(value) == expected


@pytest.mark.parametrize("value", ["", "-1", "3-1", "1-2-3"])
def test_invalid_cpuset(value):
    with pytest.raises(ValueError):
        resources._parse_cpuset(value)


def test_cgroup_v2_parent_limits(monkeypatch):
    files = {
        "/sys/fs/cgroup/team/job/cpu.max": "max 100000",
        "/sys/fs/cgroup/team/cpu.max": "350000 100000",
        "/sys/fs/cgroup/cpu.max": "200000 100000",
        "/sys/fs/cgroup/team/job/cpuset.cpus.effective": "1-5",
        "/sys/fs/cgroup/team/cpuset.cpus.effective": "0-3",
    }
    monkeypatch.setattr(resources, "_read_resource", lambda path: files.get(str(path)))
    quotas, cpus = resources._cgroup_cpu_limits(
        "0::/team/job",
        "29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw",
    )
    assert min(quotas) == 2
    assert cpus == {1, 2, 3}


def test_cgroup_v1_separate_controllers(monkeypatch):
    files = {
        "/sys/fs/cgroup/cpu/team/job/cpu.cfs_quota_us": "-1",
        "/sys/fs/cgroup/cpu/team/job/cpu.cfs_period_us": "100000",
        "/sys/fs/cgroup/cpu/team/cpu.cfs_quota_us": "250000",
        "/sys/fs/cgroup/cpu/team/cpu.cfs_period_us": "100000",
        "/sys/fs/cgroup/cpuset/team/job/cpuset.cpus": "",
        "/sys/fs/cgroup/cpuset/team/cpuset.cpus": "0-3,6",
    }
    monkeypatch.setattr(resources, "_read_resource", lambda path: files.get(str(path)))
    quotas, cpus = resources._cgroup_cpu_limits(
        "2:cpu,cpuacct:/team/job\n3:cpuset:/team/job",
        "29 23 0:26 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n"
        "30 23 0:27 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset",
    )
    assert quotas == [Fraction(5, 2)]
    assert cpus == {0, 1, 2, 3, 6}


@pytest.mark.parametrize("membership", ["0::/", "0::/container"])
def test_cgroup_namespace_mount_root(monkeypatch, membership):
    files = {"/limits/cpu.max": "50000 100000"}
    monkeypatch.setattr(resources, "_read_resource", lambda path: files.get(str(path)))
    quotas, _ = resources._cgroup_cpu_limits(
        membership, "29 23 0:26 /container /limits rw - cgroup2 cgroup rw")
    assert quotas == [Fraction(1, 2)]


def test_cgroup_escaped_mount(monkeypatch):
    files = {"/cpu limits/cpu.max": "100000 100000"}
    monkeypatch.setattr(resources, "_read_resource", lambda path: files.get(str(path)))
    quotas, _ = resources._cgroup_cpu_limits(
        "0::/", r"29 23 0:26 / /cpu\040limits rw - cgroup2 cgroup rw")
    assert quotas == [1]


def test_invalid_cgroup_quota_warns(monkeypatch):
    monkeypatch.setattr(
        resources, "_read_resource",
        lambda path: "100000 0" if str(path).endswith("cpu.max") else None)
    with pytest.warns(RuntimeWarning, match="Invalid CPU limit"):
        quotas, _ = resources._cgroup_cpu_limits(
            "0::/", "29 23 0:26 / /limits rw - cgroup2 cgroup rw")
    assert quotas == []


@pytest.mark.parametrize("quota,expected", [
    ("max 100000", 3),
    ("250000 100000", 2),
    ("50000 100000", 1),
])
def test_available_cpus_affinity_cpuset_and_quota(monkeypatch, quota, expected):
    files = {
        "/proc/self/cgroup": "0::/",
        "/proc/self/mountinfo": "29 23 0:26 / /limits rw - cgroup2 cgroup rw",
        "/limits/cpu.max": quota,
        "/limits/cpuset.cpus.effective": "0-5",
    }
    monkeypatch.setattr(resources.sys, "platform", "linux")
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 64, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {3, 4, 5, 6}, raising=False)
    monkeypatch.setattr(resources, "_read_resource", lambda path: files.get(str(path)))
    assert resources._available_cpu_count() == expected


def test_available_cpus_non_linux(monkeypatch):
    monkeypatch.setattr(resources.sys, "platform", "darwin")
    monkeypatch.setattr(os, "process_cpu_count", lambda: 12, raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert resources._available_cpu_count() == 12
