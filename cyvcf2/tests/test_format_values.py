import gzip

import numpy as np
import pytest

from ..cyvcf2 import VCF, Writer


@pytest.fixture
def string_vcf(tmp_path):
    path = tmp_path / "strings.vcf"
    path.write_text(
        '##fileformat=VCFv4.3\n'
        '##contig=<ID=1,length=100>\n'
        '##FORMAT=<ID=VS,Number=.,Type=String,Description="Values">\n'
        '##FORMAT=<ID=CA,Number=.,Type=Character,Description="Characters">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC\n'
        '1\t1\t.\tA\tC\t.\tPASS\t.\tVS:CA\ta%2Cb,c:X,Y\t.:.\tx,,y,:Z\n'
        '1\t2\t.\tA\tC\t.\tPASS\t.\tVS\tna\u00efve,\u03b2\t.\tlonger,than,before\n'
        '1\t3\t.\tA\tC\t.\tPASS\t.\tDP\t1\t2\t3\n',
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("encoding", ["vcf", "gzip", "bgzf", "bcf", "rawbcf"])
@pytest.mark.parametrize("samples", [None, ["C", "A"], []])
def test_format_values_strings(string_vcf, tmp_path, encoding, samples):
    path = string_vcf
    if encoding == "gzip":
        path = tmp_path / "strings.gzip"
        with gzip.open(path, "wb") as stream:
            stream.write(string_vcf.read_bytes())
    elif encoding != "vcf":
        path = tmp_path / ("strings.vcf.gz" if encoding == "bgzf" else "strings.bcf")
        with VCF(string_vcf) as reader:
            writer = Writer(path, reader, mode="wz" if encoding == "bgzf" else "wb")
            for record in reader:
                writer.write_record(record)
            writer.close()
        if encoding == "rawbcf":
            path.write_bytes(gzip.decompress(path.read_bytes()))
    with VCF(path, samples=samples) as reader:
        selected = reader.samples
        first = next(reader)
        expected = {"A": ["a%2Cb", "c"], "B": ["."], "C": ["x", "", "y", ""]}
        values = first.format_values("VS")
        assert values == [expected[sample] for sample in selected]
        assert first.format_values(b"VS") == values
        if selected:
            unchanged = first.format("VS")
            assert isinstance(unchanged, np.ndarray)
            assert unchanged.tolist() == [",".join(expected[sample]) for sample in selected]
        expected_ca = {"A": ["X", "Y"], "B": ["."], "C": ["Z"]}
        assert first.format_values("CA") == [expected_ca[sample] for sample in selected]
        second = next(reader)
        expected_second = {"A": ["na\u00efve", "\u03b2"], "B": ["."],
                           "C": ["longer", "than", "before"]}
        assert second.format_values("VS") == [expected_second[sample] for sample in selected]
        third = next(reader)
        assert third.format_values("VS") == (None if selected else [])
        reader.close()
        assert first.format_values("VS") == values
        assert values == [expected[sample] for sample in selected]


def test_format_values_rejects_nonstring_and_gt(string_vcf):
    with VCF(string_vcf) as reader:
        record = next(reader)
        for field in ("DP", "GT"):
            with pytest.raises(TypeError, match="String/Character"):
                record.format_values(field)
        with pytest.raises(KeyError):
            record.format_values("absent_from_header")


def test_format_values_invalid_utf8_raises(string_vcf):
    data = string_vcf.read_bytes().replace(b"a%2Cb,c", b"\xff,c")
    string_vcf.write_bytes(data)
    with VCF(string_vcf) as reader:
        record = next(reader)
        with pytest.raises(UnicodeDecodeError):
            record.format_values("VS")
