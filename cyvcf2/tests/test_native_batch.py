import gc
import gzip

import numpy as np
import pytest

from ..cyvcf2 import VCF, Writer


HEADER = """##fileformat=VCFv4.3
##contig=<ID=1,length=10000>
##FILTER=<ID=q10,Description="Low quality">
##INFO=<ID=END,Number=1,Type=Integer,Description="End">
##INFO=<ID=II,Number=.,Type=Integer,Description="Integers">
##INFO=<ID=FF,Number=.,Type=Float,Description="Floats">
##INFO=<ID=SS,Number=.,Type=String,Description="Strings">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Likelihood">
##FORMAT=<ID=FA,Number=.,Type=Float,Description="Float array">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC
"""
ROWS = [
    "1\t10\t.\tA\tC,G\t.\tPASS\tII=1,.,3;FF=.,nan,1.5\tGT:DP:PL:FA\t0|1:10:0,1,2:1,.\t1:.:3:.\t./1/2:5:4,5:2,3",
    "1\t10\trs2\tAT\tA\t12.5\tq10\tII=.;FF=.\tGT:DP:PL\t.|.:.:.\t.:7:1,2\t0|1/0:8:.",
    "1\t20\t.\tA\t<DEL>\t0\t.\tEND=25\tDP\t.:.\t5\t.",
    "1\t30\trs4\tG\t.\t1\tPASS\t.\tGT:PL\t0:.\t1|.:1,2\t.:3",
]


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "input.vcf"
    # Keep a malformed sample out of the valid fixture.
    rows = ROWS.copy()
    rows[2] = rows[2].replace("DP\t.:.", "DP\t.")
    path.write_text(HEADER + "\n".join(rows) + "\n")
    return path


def all_arrays(batch):
    yield from (value for value in batch.values() if isinstance(value, np.ndarray))
    for column in batch["fields"].values():
        yield from column.values()


def packed_row(batch, index):
    offsets = batch["string_offsets"]
    return batch["strings"][offsets[index]:offsets[index + 1]].tobytes().split(b"\t")


def field_row(batch, name, index):
    column = batch["fields"][name]
    offsets = column["offsets"]
    return column["values"][offsets[index]:offsets[index + 1]]


def decode_gt(raw, width, samples):
    raw = raw.reshape(samples, width)
    decoded = np.empty((samples, width + 1), dtype=np.int16)
    decoded[:, :-1] = (raw >> 1) - 1
    decoded[:, :-1][raw == -2147483647] = -2
    # Match the existing Genotypes.array phase policy, not a new GT policy.
    decoded[:, -1] = raw[:, 0 if width == 1 else 1] & 1
    if width > 1:
        haploid = raw[:, 1] == -2147483647
        decoded[haploid, -1] = raw[haploid, 0] & 1
    return decoded


def test_decode_gt_ignores_vector_end_phase_bit():
    raw = np.array([[2, -2147483647], [3, -2147483647],
                    [0, -2147483647], [2, 5]], dtype=np.int32)
    np.testing.assert_array_equal(
        decode_gt(raw, 2, 4),
        [[0, -2, 0], [0, -2, 1], [-1, -2, 0], [0, 1, 1]],
    )


@pytest.mark.parametrize("encoding", ["vcf", "gzip", "bgzf", "bcf", "bgzf0bcf", "rawbcf"])
@pytest.mark.parametrize("samples", [None, ["C", "A"], []])
def test_equivalence(source, tmp_path, encoding, samples):
    path = source
    if encoding == "gzip":
        path = tmp_path / "input.vcf.gz"
        with gzip.open(path, "wb") as stream:
            stream.write(source.read_bytes())
    elif encoding in ("bcf", "rawbcf", "bgzf0bcf", "bgzf"):
        suffix, mode = {"bcf": (".bcf", "wb"), "rawbcf": (".bcf", "wbu"),
                        "bgzf0bcf": (".bcf", "wbu"),
                        "bgzf": (".vcf.gz", "wz")}[encoding]
        path = tmp_path / ("converted" + suffix)
        with VCF(source) as reader:
            writer = Writer(path, reader, mode=mode)
            for record in reader:
                writer.write_record(record)
            writer.close()
        if encoding == "rawbcf":
            path.write_bytes(gzip.decompress(path.read_bytes()))
    fields = ("FORMAT/GT", "FORMAT/DP", "FORMAT/PL", "FORMAT/FA", "INFO/II", "INFO/FF")
    with VCF(path, samples=samples) as oracle, VCF(path, samples=samples) as reader:
        expected = list(oracle)
        observed = []
        while True:
            batch = reader.read_batch(fields, max_rows=2)
            assert batch["nbytes"] == sum(x.nbytes for x in all_arrays(batch))
            if batch["num_rows"] == 0:
                break
            for i in range(batch["num_rows"]):
                variant = expected[len(observed)]
                observed.append(batch["POS"][i])
                assert batch["POS"][i] == variant.POS
                assert batch["rlen"][i] == variant.end - variant.start
                np.testing.assert_array_equal(batch["QUAL"][i],
                                              np.nan if variant.QUAL is None else variant.QUAL)
                strings = packed_row(batch, i)
                assert strings[0].decode() == variant.CHROM
                assert strings[1].decode() == (variant.ID or ".")
                assert [x.decode() for x in strings[2:-1]] == [variant.REF] + variant.ALT
                assert strings[-1].decode().split(";") == variant.FILTERS or (
                    strings[-1] == b"" and variant.FILTERS == [])
                for name in fields:
                    scope, tag = name.split("/")
                    width = batch["fields"][name]["widths"][i]
                    actual = field_row(batch, name, i)
                    if scope == "FORMAT":
                        if batch["num_samples"] == 0:
                            assert width == -1 and len(actual) == 0
                            continue
                        if tag == "GT":
                            if width >= 0:
                                np.testing.assert_array_equal(
                                    actual.reshape(batch["num_samples"], width),
                                    variant.format("GT", int))
                                decoded = decode_gt(actual, width, batch["num_samples"])
                                if width > 1:
                                    np.testing.assert_array_equal(decoded, variant.genotype.array())
                                else:
                                    # Legacy array() reads past each haploid sample for phase.
                                    assert decoded[:, :-1].shape == (batch["num_samples"], 1)
                            continue
                        value = variant.format(tag)
                        if value is None:
                            assert width == -1
                            assert len(actual) == 0
                        else:
                            np.testing.assert_array_equal(actual.reshape(value.shape), value)
                            if actual.dtype == np.float32:
                                np.testing.assert_array_equal(
                                    actual.view("u4"), value.ravel().view("u4"))
                    else:
                        value = variant.INFO.get(tag)
                        if value is None:
                            if tag not in dict(variant.INFO):
                                assert width == -1
                            elif tag == "II":
                                assert actual.tolist() == [-2147483648]
                            else:
                                assert actual.view("u4").tolist() == [0x7F800001]
                        elif tag == "II":
                            values = value if isinstance(value, tuple) else (value,)
                            np.testing.assert_array_equal(
                                actual, [-2147483648 if x is None else x for x in values])
                        else:
                            values = value if isinstance(value, tuple) else (value,)
                            np.testing.assert_array_equal(
                                actual, [np.nan if x is None else x for x in values])
            retained = [x.copy() for x in all_arrays(batch)]
            gc.collect()
            for actual, copy in zip(all_arrays(batch), retained):
                np.testing.assert_array_equal(actual.view("u1"), copy.view("u1"))
        assert observed == [x.POS for x in expected]


def test_caps_ownership_and_mixed_iteration(source):
    fields = ("FORMAT/GT", "FORMAT/DP", "FORMAT/PL")
    with VCF(source) as reader:
        first = reader.read_batch(fields, max_rows=1)
    cap = first["nbytes"]
    reader = VCF(source)
    batch = reader.read_batch(fields, max_rows=100, max_bytes=cap)
    assert batch["num_rows"] == 1
    assert batch["nbytes"] == cap
    saved = [x.copy() for x in all_arrays(batch)]
    with pytest.raises(ValueError, match="pending"):
        next(reader("1:1-100"))
    with pytest.raises(ValueError, match="pending"):
        reader.set_samples(["A"])
    assert next(reader).ID == "rs2"
    assert reader.read_batch(fields, max_rows=1)["POS"].tolist() == [20]
    assert next(reader).POS == 30
    assert reader.read_batch(fields)["num_rows"] == 0
    reader.close()
    del reader
    gc.collect()
    for value, expected in zip(all_arrays(batch), saved):
        np.testing.assert_array_equal(value, expected)
        assert not value.flags.writeable
        if value.size:
            with pytest.raises(ValueError):
                value.setflags(write=True)


def test_oversized_retry_never_drops(source):
    with VCF(source) as reader:
        for _ in range(2):
            with pytest.raises(ValueError, match="single record"):
                reader.read_batch(max_bytes=32)
        assert reader.read_batch(max_rows=1)["POS"].tolist() == [10]
        with pytest.raises(ValueError, match="single record"):
            reader.read_batch(max_bytes=32)
        assert next(reader).ID == "rs2"
    with VCF(source) as reader:
        with pytest.raises(ValueError, match="single record"):
            reader.read_batch(max_bytes=32)
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            reader.read_batch()


def test_raw_gt_encoded_truth(source):
    end = -2147483647
    with VCF(source) as reader:
        batch = reader.read_batch()
    expected = [
        [2, 5, end, 4, end, end, 0, 4, 6],
        [0, 1, end, 0, end, end, 2, 5, 2],
        [],
        [2, end, 4, 1, 0, end],
    ]
    for i, raw in enumerate(expected):
        actual = field_row(batch, "FORMAT/GT", i).copy()
        if actual.size:
            width = batch["fields"]["FORMAT/GT"]["widths"][i]
            # HTSlib versions differ in the first allele's implicit phase bit.
            actual[::width] &= ~1
        assert actual.tolist() == raw
    assert batch["fields"]["FORMAT/GT"]["widths"].tolist() == [3, 3, -1, 2]


@pytest.mark.parametrize("max_rows", [1, 2, 1000])
@pytest.mark.parametrize("max_bytes", [128, 256, 1024])
def test_repeated_byte_caps_are_lossless(source, max_rows, max_bytes):
    with VCF(source) as reader:
        positions = []
        while True:
            cap = max_bytes
            try:
                batch = reader.read_batch(max_rows=max_rows, max_bytes=max_bytes)
            except ValueError as error:
                assert "single record" in str(error)
                batch = reader.read_batch(max_rows=1, max_bytes=1024)
                cap = 1024
            assert batch["num_rows"] <= max_rows
            assert batch["nbytes"] <= cap
            if not batch["num_rows"]:
                break
            positions.extend(batch["POS"].tolist())
    assert positions == [10, 10, 20, 30]


def test_float_nan_bits_and_vector_end(source):
    with VCF(source) as reader:
        batch = reader.read_batch(("FORMAT/FA", "INFO/FF"), max_rows=1)
    assert batch["fields"]["FORMAT/FA"]["values"].view("u4").tolist() == [
        0x3F800000, 0x7F800001, 0x7F800001, 0x7F800002, 0x40000000, 0x40400000]
    bits = batch["fields"]["INFO/FF"]["values"].view("u4")
    assert bits[0] == 0x7F800001
    assert bits[1] != bits[0] and np.isnan(bits[1:].view("f4")[0])


@pytest.mark.parametrize("kwargs,exception", [
    ({"max_rows": 0}, ValueError), ({"max_bytes": -1}, ValueError),
    ({"max_bytes": 1}, ValueError), ({"fields": "FORMAT/GT"}, TypeError),
    ({"fields": ["GT"]}, ValueError),
    ({"fields": ["FORMAT/GT", "FORMAT/GT"]}, ValueError),
    ({"fields": ["INFO/SS"]}, ValueError),
    ({"fields": ["FILTER/PASS"]}, ValueError),
    ({"fields": ["FORMAT/unknown"]}, KeyError),
])
def test_invalid_arguments_do_not_advance(source, kwargs, exception):
    with VCF(source) as reader:
        with pytest.raises(exception):
            reader.read_batch(**kwargs)
        assert next(reader).POS == 10


def test_bad_record_raises(source):
    source.write_text(HEADER + ROWS[0] + "\n1\tbad\t.\tA\tC\t.\t.\t.\tGT\t0\t0\t0\n")
    with VCF(source) as reader:
        with pytest.raises(ValueError, match="bcf_read"):
            reader.read_batch()
