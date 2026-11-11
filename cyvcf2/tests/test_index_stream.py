import io
import gzip
from pathlib import Path
import tempfile

import pytest

from cyvcf2 import VCF, build_index, build_index_stream
import cyvcf2.cyvcf2 as native
from .test_reader import coordinate_vcf


def _assert_regions(path, index, reference):
    with VCF(str(path), samples=[]) as actual, VCF(str(path), samples=[]) as expected:
        actual.set_index(str(index))
        expected.set_index(str(reference))
        assert actual.num_records == expected.num_records == 800
        for chrom in ("1", "undeclared"):
            for begin, end in [(1, 490), (1, 1), (150, 150), (400, 490), (500, 600)]:
                region = f"{chrom}:{begin}-{end}"
                assert [(v.POS, v.end, v.ID) for v in actual(region)] == [
                    (v.POS, v.end, v.ID) for v in expected(region)
                ]


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("suffix", ["csi", "tbi"])
def test_bounded_local_index(coordinate_vcf, tmp_path, workers, suffix):
    reference = tmp_path / ("native." + suffix)
    destination = tmp_path / ("bounded." + suffix)
    build_index(coordinate_vcf, reference)
    assert build_index(
        coordinate_vcf, destination, bounded=True, workers=workers, threads=1,
    ) == str(destination)
    _assert_regions(coordinate_vcf, destination, reference)
    saved = destination.read_bytes()
    with pytest.raises(FileExistsError):
        build_index(coordinate_vcf, destination, bounded=True)
    assert destination.read_bytes() == saved
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


@pytest.mark.parametrize("suffix", ["csi", "tbi"])
def test_public_stream_without_staging(coordinate_vcf, tmp_path, monkeypatch, suffix):
    reference = tmp_path / ("native." + suffix)
    build_index(coordinate_vcf, reference)

    def forbid_staging(*args, **kwargs):
        pytest.fail("stream output must not use a staging file")

    for name in ("TemporaryDirectory", "TemporaryFile", "NamedTemporaryFile", "mkstemp"):
        monkeypatch.setattr(tempfile, name, forbid_staging)
    output = io.BytesIO()
    build_index_stream(
        coordinate_vcf, output, index_format=suffix, workers=4, threads=1,
    )
    assert not output.closed
    destination = tmp_path / ("streamed." + suffix)
    destination.write_bytes(output.getvalue())
    _assert_regions(coordinate_vcf, destination, reference)


def test_bounded_publication_race(coordinate_vcf, tmp_path, monkeypatch):
    from cyvcf2 import _index
    original = _index.build_index_stream
    destination = tmp_path / "raced.csi"

    def publish_concurrently(*args, **kwargs):
        original(*args, **kwargs)
        destination.write_bytes(b"other writer")

    monkeypatch.setattr(_index, "build_index_stream", publish_concurrently)
    with pytest.raises(FileExistsError):
        build_index(coordinate_vcf, destination, bounded=True, workers=4)
    assert destination.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_bounded_rejects_bcf_and_cleans_up(tmp_path):
    if not native._coordinate_workers_available:
        pytest.skip("bounded indexing requires bundled HTSlib")
    destination = tmp_path / "bcf.csi"
    with pytest.raises(ValueError, match="BGZF-compressed VCF"):
        build_index(Path(__file__).with_name("multi-contig.bcf"), destination, bounded=True)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


@pytest.mark.parametrize("kwargs,message", [
    ({"index_format": "bai"}, "index_format"),
    ({"workers": 0}, "workers"),
    ({"threads": -1}, "threads"),
])
def test_stream_invalid_arguments(coordinate_vcf, kwargs, message):
    output = io.BytesIO()
    with pytest.raises(ValueError, match=message):
        build_index_stream(coordinate_vcf, output, **kwargs)
    assert output.getvalue() == b""
    assert not output.closed


def test_bounded_option_requires_boolean(coordinate_vcf, tmp_path):
    with pytest.raises(ValueError, match="bounded"):
        build_index(coordinate_vcf, tmp_path / "bad.csi", bounded="yes")


@pytest.mark.parametrize("compressed", [False, True])
def test_bounded_rejects_non_bgzf(coordinate_vcf, tmp_path, compressed):
    data = gzip.decompress(coordinate_vcf.read_bytes())
    path = tmp_path / ("plain.vcf.gz" if compressed else "plain.vcf")
    path.write_bytes(gzip.compress(data) if compressed else data)
    destination = tmp_path / "invalid.csi"
    with pytest.raises(ValueError, match="BGZF-compressed VCF"):
        build_index(path, destination, bounded=True)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_core_stream_output_failure(coordinate_vcf):
    class FailedOutput:
        def write(self, data):
            raise OSError("output failed")

    with pytest.raises(OSError, match="output failed"):
        build_index_stream(coordinate_vcf, FailedOutput(), workers=4)


def test_core_stream_partial_writes(coordinate_vcf):
    class ShortOutput(io.BytesIO):
        def write(self, data):
            return super().write(data[:7])

    reference = io.BytesIO()
    build_index_stream(coordinate_vcf, reference)
    output = ShortOutput()
    build_index_stream(coordinate_vcf, output)
    assert output.getvalue() == reference.getvalue()


@pytest.mark.parametrize("bounded,workers", [(False, 4), (True, 1), (True, 4)])
def test_index_progress(coordinate_vcf, tmp_path, bounded, workers):
    events = []
    destination = tmp_path / "progress.csi"
    build_index(coordinate_vcf, destination, bounded=bounded, workers=workers, progress=events.append)
    passes = 2 if bounded else 1
    assert events[0]["fraction"] == 0
    assert events[0]["eta_seconds"] is None
    assert {event["pass"] for event in events} == set(range(1, passes + 1))
    assert [event["fraction"] for event in events] == sorted(event["fraction"] for event in events)
    assert events[-1]["phase"] == "complete"
    assert events[-1]["fraction"] == 1
    assert events[-1]["eta_seconds"] == 0
    assert events[-1]["total_bytes"] == coordinate_vcf.stat().st_size * passes
    reference = tmp_path / "reference.csi"
    build_index(coordinate_vcf, reference)
    _assert_regions(coordinate_vcf, destination, reference)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("when", ["start", "complete"])
def test_progress_failure_prevents_publication(coordinate_vcf, tmp_path, bounded, when):
    destination = tmp_path / "failed.csi"

    def fail(event):
        if when == "start" or event["phase"] == "complete":
            raise RuntimeError("progress failed")

    with pytest.raises(RuntimeError, match="progress failed"):
        build_index(coordinate_vcf, destination, workers=4, bounded=bounded, progress=fail)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_progress_invalid_or_unavailable(coordinate_vcf, tmp_path):
    with pytest.raises(ValueError, match="progress requires"):
        build_index(coordinate_vcf, tmp_path / "native.csi", progress=lambda event: None)
    with pytest.raises(TypeError, match="progress must be callable"):
        build_index_stream(coordinate_vcf, io.BytesIO(), progress=True)
    with pytest.raises(TypeError, match="progress must be callable"):
        build_index(coordinate_vcf, tmp_path / "bad.csi", workers=4, progress=True)


def test_progress_throttle_and_eta(tmp_path, monkeypatch):
    from cyvcf2 import _index
    path = tmp_path / "input"
    path.write_bytes(b"x" * 1000)
    clock = [10.0]
    monkeypatch.setattr(_index.time, "monotonic", lambda: clock[0])
    events = []
    progress = _index._IndexProgress(bytes(path), events.append, 2)
    progress.report(1, force=True)
    clock[0] = 10.5
    progress.report(1, 200 << 16)
    assert len(events) == 1
    clock[0] = 12.0
    progress.report(1, 500 << 16)
    assert events[-1]["fraction"] == 0.25
    assert events[-1]["bytes_per_second"] == 250
    assert events[-1]["eta_seconds"] == 6
    progress.report(2, force=True)
    assert events[-1]["fraction"] == 0.5
    clock[0] = 14.0
    progress.report(2, 500 << 16)
    assert events[-1]["fraction"] == 0.75
    assert events[-1]["eta_seconds"] == pytest.approx(4 / 3)
    progress.report(2, phase="complete", force=True)
    assert events[-1]["fraction"] == 1
    assert events[-1]["eta_seconds"] == 0
