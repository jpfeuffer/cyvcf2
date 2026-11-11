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
