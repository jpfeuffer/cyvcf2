import gzip
import shutil
from pathlib import Path

import pytest

from cyvcf2 import VCF


SOURCE = Path(__file__).with_name("test.vcf.gz")


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("uncompressed", [False, True])
def test_schema_progress(workers, uncompressed, tmp_path):
    source = tmp_path / ("input.vcf" if uncompressed else "input.vcf.gz")
    if uncompressed:
        source.write_bytes(gzip.decompress(SOURCE.read_bytes()))
    else:
        shutil.copyfile(SOURCE, source)
    with VCF(str(source)) as reader:
        expected = reader.schema_scan(workers=workers, threads=1)
    events = []
    with VCF(str(source)) as reader:
        actual = reader.schema_scan(workers=workers, threads=1, progress=events.append)
    assert actual == expected
    assert events[0]["fraction"] == 0
    assert events[0]["eta_seconds"] is None
    assert events[-1]["phase"] == "complete"
    assert events[-1]["fraction"] == 1
    assert events[-1]["records"] == actual["records"]
    assert events[-1]["bytes_processed"] == events[-1]["total_bytes"] == source.stat().st_size
    assert events[-1]["eta_seconds"] == 0
    assert [e["fraction"] for e in events] == sorted(e["fraction"] for e in events)


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("phase", ["scanning", "finalizing", "complete"])
def test_schema_progress_failure_no_index(workers, phase, tmp_path):
    destination = tmp_path / "output.csi"

    def fail(event):
        if event["phase"] == phase:
            raise RuntimeError("progress failed")

    with VCF(str(SOURCE)) as reader:
        with pytest.raises(RuntimeError, match="progress failed"):
            reader.schema_scan(index_path=destination, workers=workers, progress=fail)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_schema_progress_invalid():
    with VCF(str(SOURCE)) as reader:
        with pytest.raises(TypeError, match="progress must be callable"):
            reader.schema_scan(progress=True)


def test_schema_progress_throttle(tmp_path, monkeypatch):
    from cyvcf2 import _schema_scan
    source = tmp_path / "input"
    source.write_bytes(b"x" * 1000)
    clock = [10.0]
    monkeypatch.setattr(_schema_scan.time, "monotonic", lambda: clock[0])
    events = []
    reporter = _schema_scan._SchemaProgress(bytes(source), events.append)
    reporter.update(0, 0, force=True)
    clock[0] = 10.5
    reporter.update(200, 1)
    assert len(events) == 1
    clock[0] = 12.0
    reporter.update(500, 2)
    assert events[-1]["fraction"] == 0.5
    assert events[-1]["bytes_per_second"] == 250
    assert events[-1]["eta_seconds"] == 2
    reporter.update(1000, 4, phase="complete", force=True)
    assert events[-1]["records"] == 4
    assert events[-1]["eta_seconds"] == 0
