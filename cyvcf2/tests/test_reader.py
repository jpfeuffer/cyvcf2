from __future__ import print_function
import os.path
import platform
import tempfile
import sys
import os
import shutil
import atexit
import gzip
try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport
import warnings

import numpy as np
import pytest

from ..cyvcf2 import VCF, Variant, Writer


HERE = os.path.dirname(__file__)
VCF_PATH = os.path.join(HERE, "test.vcf.gz")
VCF_PATH2 = os.path.join(HERE, "test.snpeff.vcf")
VCF_PATH3 = os.path.join(HERE, "test.mnp.vcf")
VCF_PHASE_PATH = os.path.join(HERE, "test.comp_het.3.vcf")
VCF_ALTFREQ_PATH = os.path.join(HERE, "test_gt_alt_freqs.vcf")

try:
    basestring
except NameError:
    basestring = (str, bytes)

def test_init():
    # string
    v = VCF(VCF_PATH)
    assert v
    expected_count = sum(1 for _ in v)
    v.close()

    # Path
    v = VCF(Path(VCF_PATH))
    value = sum(1 for _ in v)
    assert value == expected_count

    # file descriptor
    with open(VCF_PATH) as fp:
        fd = fp.fileno()
        v = VCF(fd)
        assert sum(1 for _ in v) == expected_count
        v.close()  # this should not close the file descriptor originally opened

    # file-like object
    with open(VCF_PATH) as fp:
        v = VCF(fp)
        assert sum(1 for _ in v) == expected_count
        v.close()  # this should not close the file descriptor originally opened

def test_type():
    vcf = VCF(VCF_PATH)
    for v in vcf:
        if len(v.REF) == 1 and len(v.ALT[0]) == 1:
            assert v.var_type == 'snp'
        elif v.ALT[0][0] != "<":
            assert v.var_type == 'indel'
        else:
            print(v.var_type, v.REF, v.ALT)

def test_type_mnp():
    vcf = VCF(VCF_PATH3)
    for v in vcf:
        if len(v.ALT) == 1:
            if (v.REF, v.ALT[0]) in [("CGT","CGG"), ("AGG","CGA")]:
                assert v.var_type == 'mnp'
            if (v.REF, v.ALT[0]) in [("GCA","GA")]:
                assert v.var_type == 'indel'
        if len(v.ALT) == 2:
            if (v.REF, v.ALT[0], v.ALT[1]) in [("TCGGT","GCGGG","GCGGT")]:
                assert v.var_type == 'mnp'
            if (v.REF, v.ALT[0], v.ALT[1]) in [("ATTAC","ATAC","AATAC")]:
                assert v.var_type == 'indel'
        if len(v.ALT) == 3:
            if (v.REF, v.ALT[0], v.ALT[1], v.ALT[2]) in [("GCC","TCC","GCA","GCG")]:
                assert v.var_type == 'mnp'

def test_format_str():
    vcf = VCF(os.path.join(HERE, "test-format-string.vcf"))

    f = next(vcf).format("RULE")
    assert list(f) == ['F', 'G']
    f = next(vcf).format("RULE")
    assert list(f) == ['F2,F3,F4', 'G2,G3,G4']

def test_missing_samples():
    samples = ['101976-101976', 'sample_not_in_vcf']
    with warnings.catch_warnings(record=True) as w:
        vcf = VCF(VCF_PATH, gts012=True, samples=samples)
        assert len(w) == 1
        assert "not all requested samples found" in str(w[-1].message)
    assert len(vcf.samples) == 1
    vcf.close()
    samples = '101976-101976,sample_not_in_vcf'
    with warnings.catch_warnings(record=True) as w:
        vcf = VCF(VCF_PATH, gts012=True, samples=samples)
        assert len(w) == 1
        assert "not all requested samples found" in str(w[-1].message)
    assert len(vcf.samples) == 1

def test_ibd():
    samples = ['101976-101976', '100920-100920', '100231-100231']
    vcf = VCF(VCF_PATH, gts012=True, samples=samples)
    res = vcf.ibd()
    assert len(res) == 3, (len(res))
    arr = res[('101976-101976', '100920-100920')]
    assert len(arr) > 0

def test_relatedness():
    vcf = VCF(VCF_PATH, gts012=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore", category=DeprecationWarning)
        df = vcf.relatedness(gap=0, linkage_max=2)
        assert len(w) == 1
        assert "tested:" in str(w[-1].message)
    assert "ibs0" in df, df
    assert "rel" in df
    #vcf = VCF(VCF_PATH, gts012=True)
    #for r in viter:
    #    print r['pair'], r['ibs0'], r['ibs2'], r['ibs2*']

def test_pls():
    vcf = VCF(VCF_PATH)
    v = next(vcf)

    assert v.gt_phred_ll_homref[0] == 0, v.gt_phred_ll_homref[0]
    assert v.gt_phred_ll_het[0] == 7, v.gt_phred_ll_het[0]
    assert v.gt_phred_ll_homalt[0] == 922, v.gt_phred_ll_homalt[0]

    import numpy as np
    imax = np.iinfo(np.int32(0)).max
    # missing
    assert v.gt_phred_ll_homref[1] == imax, v.gt_phred_ll_homref[1]
    assert v.gt_phred_ll_het[1] == imax, v.gt_phred_ll_het[1]
    assert v.gt_phred_ll_homalt[1] == imax, v.gt_phred_ll_homalt[1]

def test_gt_alt_freqs():
    vcf = VCF(VCF_ALTFREQ_PATH)

    v = next(vcf)
    assert v.gt_alt_freqs[0] == 0.2
    assert v.gt_alt_freqs[1] == 1.0

    v = next(vcf)
    assert v.gt_alt_freqs[0] == 0.5
    assert v.gt_alt_freqs[1] == 0.9

    v = next(vcf)
    assert v.gt_alt_freqs[0] == 0.0
    assert v.gt_alt_freqs[1] == 0.0

    v = next(vcf)
    assert v.gt_alt_freqs[0] == 0.0
    assert v.gt_alt_freqs[1] == 1.0

    v = next(vcf)
    assert v.gt_alt_freqs[0] == 0.0
    assert v.gt_alt_freqs[1] == 0.0

    v = next(vcf)
    assert v.gt_alt_freqs[0] == -1
    assert v.gt_alt_freqs[1] == -1

def test_str():
    vcf = VCF(VCF_PATH, lazy=True)
    v = next(vcf)
    s = str(v)

    assert "10172\t.\tCCCTAA\t" in s
    assert "fitcons_float=0.1266" in s


def test_region():
    vcf = VCF(VCF_PATH)

    start = 12783
    end = 13783
    k = 0
    reg = '1:%d-%d' % (start, end)
    for var in vcf(reg):
        k += 1
        assert var.start <= end, var
        assert var.end >= start, var
        assert isinstance(var.REF, basestring)
        assert isinstance(var.ALT, list)
    assert k == 28, k

def test_empty_info():
    for v in VCF(VCF_PHASE_PATH):
        dict(v.INFO)

def test_phases():
    vcf = VCF(VCF_PHASE_PATH)

    v = next(vcf)
    assert all(v.gt_phases), v.gt_phases

    v = next(vcf)
    assert all(v.gt_phases)

    v = next(vcf)
    assert all(v.gt_phases[1::2])
    assert not any(v.gt_phases[0::2])

    v = next(vcf)
    assert not any(v.gt_phases[:-1])
    assert v.gt_phases[-1]

    v = next(vcf)
    assert not any(v.gt_phases)

def test_bad_init():
    with pytest.raises(Exception):
        VCF("XXXXX")

def test_samples():
    v = VCF(VCF_PATH)
    assert len(v.samples) == 189

def test_next():
    v = VCF(VCF_PATH)
    variant = next(v)
    assert isinstance(variant, Variant)

def test_variant():
    with pytest.raises(TypeError):
        Variant()

def test_info_dict():
    v = VCF(VCF_PATH)
    variant = next(v)
    d = dict(variant.INFO)
    assert d != {}, d
    toks = _get_line_for(variant)

    info = toks[7].split(";")
    keys = [x.split('=')[0] for x in info]
    for k in keys:
        assert k in d, (k, info)


def test_attrs():
    # 1 10172   .       CCCTAA  C       92.0    PASS
    v = VCF(VCF_PATH)
    variant = next(v)
    assert variant.POS == 10172
    assert variant.CHROM == "1"
    assert variant.ID is None, variant.ID
    assert variant.start == 10171
    assert variant.end == 10177, variant.end
    assert variant.FILTER is None
    assert variant.QUAL == 92.0

    assert variant.REF == "CCCTAA"
    assert variant.ALT == ["C"]

def test_empty():
    p = os.path.join(HERE, "empty.vcf")
    assert os.path.exists(p)
    with pytest.raises(IOError):
        VCF(p)

def test_format_field():
    vcf = VCF(VCF_PATH)
    for v in vcf:
        assert isinstance(v.FORMAT, list)

def test_writer_from_string():

    header = """##fileformat=VCFv4.1
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr2,length=249250621,assembly=hg19>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	samplea
"""

    w = Writer.from_string("out.vcf", header)
    w.write_header()
    v = w.variant_from_string("chr1\t234\t.\tA\tC\t40\tPASS\t.\tGT\t0/0")
    w.write_record(v)
    w.close()

def test_variant_from_string_malformed():
    # vcf_parse() returns 0 on success and a negative value on failure, so a
    # malformed line must raise rather than return a partially-filled record.
    # Reading REF or ID off such a record segfaults.
    header = """##fileformat=VCFv4.1
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=249250621,assembly=hg19>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	samplea
"""

    output_path = tempfile.mktemp(suffix=".vcf")
    atexit.register(os.unlink, output_path)
    w = Writer.from_string(output_path, header)
    with pytest.raises(Exception, match="error parsing"):
        w.variant_from_string("chr1\tNOTANUMBER\t.\tA\tC\t40\tPASS\t.\tGT\t0/0")
    w.close()

def test_isa():
    vcf = VCF(os.path.join(HERE, "test.isa.vcf"))
    for i, v in enumerate(vcf):
        if i in {0, 1, 2, 3}:
            assert v.is_indel
            assert not v.is_snp
        if i in {4}:
            assert v.is_snp




def run_writer(writer, filename, rec):
    rec.INFO["AC"] = "3"
    rec.FILTER = ["LowQual"]
    writer.write_record(rec)

    rec.FILTER = ["LowQual", "VQSRTrancheSNP99.90to100.00"]
    writer.write_record(rec)

    rec.FILTER = "PASS"
    writer.write_record(rec)

    writer.close()

    expected = ["LowQual", "LowQual;VQSRTrancheSNP99.90to100.00", None]

    for i, variant in enumerate(VCF(filename)):
        assert variant.FILTER == expected[i], (variant.FILTER, expected[i])

def test_writer():
    v = VCF(VCF_PATH)
    f = tempfile.mktemp(suffix=".vcf")
    atexit.register(os.unlink, f)
    rec = next(v)

    # string
    run_writer(Writer(f, v), f, rec)

    # Path
    path = Path(f)
    run_writer(Writer(path, v), f, rec)

    # file descriptor
    with open(VCF_PATH) as fp:
        fd = fp.fileno()
        run_writer(Writer(fd, v), f, rec)

    # file-like object
    with open(VCF_PATH) as fp:
        run_writer(Writer(fp, v), f, rec)

def test_filters():
    v = VCF(VCF_PATH)
    f = tempfile.mktemp(suffix=".vcf")
    atexit.register(os.unlink, f)
    rec = next(v)

    writer = Writer(f, v)

    rec.INFO["AC"] = "3"
    rec.FILTER = ["LowQual"]
    writer.write_record(rec)

    rec.FILTER = ["LowQual", "VQSRTrancheSNP99.90to100.00"]
    writer.write_record(rec)

    rec.FILTER = "PASS"
    writer.write_record(rec)

    rec.FILTER = []
    writer.write_record(rec)

    writer.close()

    expected_filter = ["LowQual", "LowQual;VQSRTrancheSNP99.90to100.00", None, None]
    expected_filters = [["LowQual"], ["LowQual", "VQSRTrancheSNP99.90to100.00"], ["PASS"], []]

    for i, variant in enumerate(VCF(f)):
        assert variant.FILTER == expected_filter[i], (variant.FILTER, expected_filter[i])
        assert variant.FILTERS == expected_filters[i], (variant.FILTERS, expected_filters[i])

def test_add_info_to_header():
    v = VCF(VCF_PATH)
    v.add_info_to_header({'ID': 'abcdefg', 'Description': 'abcdefg',
        'Type':'Character', 'Number': '1'})
    # NOTE that we have to add the info to the header of the reader,
    # not the writer because the record will be associated with the reader
    f = tempfile.mktemp(suffix=".vcf")
    atexit.register(os.unlink, f)
    w = Writer(f, v)
    import sys
    rec = next(v)

    rec.INFO["abcdefg"] = "XXX"
    w.write_record(rec)
    w.close()

    v = next(VCF(f))
    ret = v.INFO["abcdefg"]
    if isinstance(ret, bytes):
        ret = ret.decode()
    assert ret == "XXX", (dict(v.INFO), v.INFO["abcdefg"])

def test_read_flag():
    vcf = VCF(VCF_PATH)
    for v in vcf:
        assert ("in_exac_flag" in str(v)) == v.INFO.get('in_exac_flag', False)

def test_add_flag():
    vcf = VCF(VCF_PATH)
    vcf.add_info_to_header({'ID': 'myflag', 'Description': 'myflag',
        'Type':'Flag', 'Number': '0'})
    # NOTE that we have to add the info to the header of the reader,
    # not the writer because the record will be associated with the reader
    f = tempfile.mktemp(suffix=".vcf")
    atexit.register(os.unlink, f)
    w = Writer(f, vcf)
    rec = next(vcf)

    rec.INFO["myflag"] = True
    w.write_record(rec)
    w.close()

    fh = VCF(f)
    v = next(fh)
    fh.close()
    assert v.INFO["myflag"] is True, dict(v.INFO)

    f = tempfile.mktemp(suffix=".vcf")
    atexit.register(os.unlink, f)
    w = Writer(f, vcf)
    rec.INFO["myflag"] = False
    w.write_record(rec)
    w.close()
    fh = VCF(f)
    v = next(fh)
    fh.close()
    with pytest.raises(KeyError):
        v.INFO["myflag"]

def test_issue198():
    vcf = VCF(os.path.join(HERE, "issue_198.vcf"), strict_gt=True)
    for v in vcf:
        assert all(v.gt_types == [2]), v.gt_types

    vcf = VCF(os.path.join(HERE, "issue_198.vcf"), strict_gt=False)
    for v in vcf:
        assert all(v.gt_types == [0]), v.gt_types

def test_constants():
    v = VCF(VCF_PATH)
    assert v.HOM_REF == 0
    assert v.HET == 1
    assert v.UNKNOWN == 2
    assert v.HOM_ALT == 3

    v = VCF(VCF_PATH, gts012=True)
    assert v.HOM_REF == 0
    assert v.HET == 1
    assert v.HOM_ALT == 2
    assert v.UNKNOWN == 3


def test_add_filter_to_header():
    v = VCF(VCF_PATH)
    # NOTE that we have to add the filter to the header of the reader,
    # not the writer because the record will be associated with the reader
    v.add_filter_to_header({'ID': 'abcdefg', 'Description': 'abcdefg'})

    f = tempfile.mktemp(suffix=".vcf")
    atexit.register(os.unlink, f)
    w = Writer(f, v)
    rec = next(v)

    rec.FILTER = ["abcdefg"]
    w.write_record(rec)
    w.close()

    v = next(VCF(f))
    ret = v.FILTER
    if isinstance(ret, bytes):
        ret = ret.decode()

    assert ret == "abcdefg", v.FILTER

def test_seqnames():
    v = VCF(VCF_PATH)
    assert v.seqnames ==  [u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'10', u'11', u'12', u'13', u'14', u'15', u'16', u'17', u'18', u'19', u'20', u'21', u'22', u'X', u'Y', u'MT', u'GL000207.1', u'GL000226.1', u'GL000229.1', u'GL000231.1', u'GL000210.1', u'GL000239.1', u'GL000235.1', u'GL000201.1', u'GL000247.1', u'GL000245.1', u'GL000197.1', u'GL000203.1', u'GL000246.1', u'GL000249.1', u'GL000196.1', u'GL000248.1', u'GL000244.1', u'GL000238.1', u'GL000202.1', u'GL000234.1', u'GL000232.1', u'GL000206.1', u'GL000240.1', u'GL000236.1', u'GL000241.1', u'GL000243.1', u'GL000242.1', u'GL000230.1', u'GL000237.1', u'GL000233.1', u'GL000204.1', u'GL000198.1', u'GL000208.1', u'GL000191.1', u'GL000227.1', u'GL000228.1', u'GL000214.1', u'GL000221.1', u'GL000209.1', u'GL000218.1', u'GL000220.1', u'GL000213.1', u'GL000211.1', u'GL000199.1', u'GL000217.1', u'GL000216.1', u'GL000215.1', u'GL000205.1', u'GL000219.1', u'GL000224.1', u'GL000223.1', u'GL000195.1', u'GL000212.1', u'GL000222.1', u'GL000200.1', u'GL000193.1', u'GL000194.1', u'GL000225.1', u'GL000192.1', u'NC_007605', u'hs37d5', u'phix'], v.seqnames

    b = VCF('{}/test.snpeff.bcf'.format(HERE), threads=3)
    assert b.seqnames[0] == 'chr1', b.seqnames
    assert b.seqnames[-1] == 'chrY', b.seqnames

def test_different_index():
    b = VCF('{}/test.snpeff.bcf'.format(HERE), threads=3)
    b.set_index("{}/test-diff.csi".format(HERE))
    s = 0
    for r in b("chr1:69427-69429"):
        s += 1
    assert s == 1

def test_var_type():
    v = VCF(VCF_PATH)
    variant = next(v)
    assert variant.var_type == "indel", variant.var_type
    # 1 10172   .       CCCTAA  C       92.0    PASS
    for variant in v:
        if variant.POS == 10478: break
    else:
        raise Exception
    assert variant.var_type == "snp", variant.var_type

def _get_line_for(v):
    import gzip

    with gzip.open(VCF_PATH) as f:
        for i, line in enumerate(f, start=1):
            line = line.decode()
            if line[0] == "#": continue
            toks = line.strip().split("\t")
            if not (toks[0] == v.CHROM and int(toks[1]) == v.POS): continue
            if toks[3] != v.REF: continue
            if toks[4] not in v.ALT: continue
            return toks
        else:
            raise Exception("not found")


def _get_samples(v):
    import numpy as np

    def _get_gt(s):
        if not ":" in s:
            return 2
        s = s.split(":", 1)[0]
        if s in ("0/0", "0|0", "0/.", "./0", "0|."):
            return 0
        if s in ("0/1", "0|1", "1/.", "./1", "1/."):
            return 1
        if s in ("1/1", "1|1"):
            return 3
        return 2
    toks = _get_line_for(v)
    samples = toks[9:]
    return np.array([_get_gt(s) for s in samples], np.int32)

def test_header_info():
    v = VCF(VCF_PATH)
    csq = v['CSQ']
    assert csq['ID'] == "CSQ"
    assert "Description" in csq


    with pytest.raises(KeyError):
        v[b'XXXXX']

def test_snpeff_header():
    v = VCF(VCF_PATH2)

    f = v['SnpEffVersion']
    assert f != {}, f
    assert 'SnpEffVersion' in f

#def test_info_update():
#    vcf = VCF(VCF_PATH)
#    v = next(vcf)
#    ret = v.INFO.update({'k': 22})
    #print ret
    #assert v.INFO['k'] == 22
    #assert ret == 0, ret

def test_gt_types():
    v = VCF(VCF_PATH)
    for variant in v:

        gt_types = variant.gt_types
        o = _get_samples(variant)
        assert (gt_types == o).all(), (variant, variant.CHROM, variant.POS, zip(gt_types, o))

def test_raw_header():
    v = VCF(VCF_PATH)
    h = v.raw_header.strip().split("\n")
    s = h[0]
    assert s == "##fileformat=VCFv4.1", s
    assert len(h) == 185, len(h)



def test_iterate():

    for i, v in enumerate(VCF(VCF_PATH), start=1):
        pass
    assert i == 115, i


def test_empty_call():

    for i, v in enumerate(VCF(VCF_PATH)(), start=1):
        pass
    assert i == 115, i
    del i

    for i, v in enumerate(VCF(VCF_PATH)(''), start=1):
        pass
    assert i == 115, i

def test_call_restarts_each_time():
    vcf = VCF(VCF_PATH)
    first = sum(1 for _ in vcf())
    second = sum(1 for _ in vcf())
    assert first == 115, first
    assert second == 115, second

def test_region_query_does_not_advance_empty_call():
    vcf = VCF(VCF_PATH)
    chr1_count = sum(1 for _ in vcf("1"))
    full_count = sum(1 for _ in vcf())
    assert chr1_count > 0, chr1_count
    assert full_count == 115, full_count

def test_empty_call_without_index_is_empty():
    # test.snpeff.vcf is not indexed.
    vcf = VCF(VCF_PATH2)
    assert sum(1 for _ in vcf()) == 0



def test_haploid():

    for (gts012, (HOM_REF, HOM_ALT, UNKNOWN)) in ((False, [0, 3, 2]), (True, [0, 2, 3])):
        vcf = VCF("%s/test-haploidX.vcf" % HERE, gts012=gts012)
        for i, v in enumerate(vcf):
            assert not any("/" in b for b in v.gt_bases), (v.start + 1, v.gt_bases)
            if i == 0:
                assert (v.gt_types == [HOM_ALT, HOM_ALT, HOM_ALT]).all(), v.gt_types
            elif v.start == 2800676:
                assert (v.gt_types == [UNKNOWN, HOM_ALT, UNKNOWN]).all(), v.gt_types
            elif v.start == 2832771:
                assert (v.gt_types == [HOM_REF, HOM_ALT, HOM_ALT]).all(), v.gt_types
                break

            if v.start == 2700156:
                assert (v.gt_bases == ['A', 'A', 'A']).all(), v.gt_bases
        break


def test_diploid():
    vcf = VCF("%s/test.comp_het.3.vcf" % HERE)
    for v in vcf:
        if v.start == 17362:
            assert (v.gt_bases == ['TTCT|TTCT', 'TTCT|TTCT', 'TTCT|TTCT',
                                   'TTCT|TTCT', 'TTCT|TTCT', 'TTCT|TTCT', 'TTCT|T', 'TTCT|T',
                                   'TTCT|TTCT', 'TTCT|T', 'TTCT|T',
                                   'TTCT|TTCT']).all(), v.gt_bases


def test_format():

    vcf = VCF('{}/test.vcf.gz'.format(HERE))
    for v in vcf:
        a = v.format('PL', int)
        assert a.shape == (189, 3)

        a = v.format('SB', int)
        assert a is None

        a = v.format('DP', int)
        assert a.shape == (189, 1)

def test_header_stuff():
    vcf = VCF('{}/test.vcf.gz'.format(HERE))
    import sys
    seen_formats, seen_infos = 0, 0
    for h in vcf.header_iter():
        i = h.info(extra=True)
        assert isinstance(i, dict)
        seen_formats += i['HeaderType'] == 'FORMAT'
        seen_infos += i['HeaderType'] == 'INFO'
    assert seen_formats == 9, seen_formats
    assert seen_infos == 73, seen_infos


def test_bcf_hdr_remove():
    v = VCF(VCF_PATH)
    v.add_info_to_header({'ID': 'TESTINFO', 'Description': 'Test INFO field',
        'Type': 'String', 'Number': '1'})

    assert v.get_header_type("TESTINFO")

    v.remove_header('TESTINFO')
    with pytest.raises(KeyError):
        v.get_header_type('TESTINFO')


def test_bcf():
    vcf = VCF('{}/test.snpeff.bcf'.format(HERE))
    l = sum(1 for _ in vcf)
    assert l == 10, l

    # __call__ should return an iterator from the beginning each time.
    l = sum(1 for _ in vcf())
    assert l == 10, l


    viter = vcf("1:69260-69438")
    sys.stderr.write("\nOK\n")
    sys.stderr.flush()
    with warnings.catch_warnings(record=True) as w:
        l = list(viter)
        assert len(w) == 1
        assert "no intervals found" in str(w[-1].message)
    assert len(l) == 0, len(l)

    iter = vcf("chr1:69260-69438")
    l = list(iter)
    assert len(l) == 2, len(l)

def test_vcf_no_intervals():
    vcf = VCF('{}/test.vcf.gz'.format(HERE))
    viter = vcf("not_a_chrom")
    with warnings.catch_warnings(record=True) as w:
        l = list(viter)
        assert len(w) == 1
        assert "no intervals found" in str(w[-1].message)
    assert len(l) == 0, len(l)

def test_issue12():
    fields = "ADP_ALL ADPD ADPO ADP_PASS ADPR AFR AMBIG BMF_PASS BMF_QUANT AF_FAILED FA_FAILED FM_FAILED FP_FAILED FR_FAILED MD_FAILED IMPROPER MQ_FAILED OVERLAP PV_FAILED QSS".split()

    vcf = VCF('{}/bug.vcf.gz'.format(HERE))
    for v in vcf:
        for f in fields:
            vals = v.format(f)
            if vals is not None:
                assert vals.dtype in (np.int32, np.float32), (f, vals.dtype)

        vals = v.format("RVF")
        assert vals.dtype in (np.float32, np.float64)

    with pytest.raises(KeyError):
        v.format("RULE")

def test_gt_bases_nondiploid():
    """Ensure gt_bases works with more complex base representations.
    """
    vcf = VCF('{}/test_gt_bases.vcf.gz'.format(HERE))
    expected = {0: ['C/C', 'C/C'], 1: ['C/T', 'C'], 2: ['C', 'C/T'], 3: ['C', 'C']}
    for i, v in enumerate(vcf):
        assert v.gt_bases.tolist() == expected[i], (v.gt_bases.tolist(), expected[i])

def fmap(fn, a):
    return list(map(fn, a))

def allclose(a, b):
    return np.allclose(np.array(a, dtype=float), np.array(b, dtype=float))

def test_set_format_float():
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    assert vcf.add_format_to_header(dict(ID="PS", Number=1, Type="Float", Description="PS example")) == 0
    v = next(vcf)
    v.set_format("PS", np.array([0.555, 1.111], dtype=np.float32))
    assert allclose(fmap(float, get_gt_str(v, "PS")), np.array([0.555, 1.111]))

    v.set_format("PS", np.array([8.555, 11.111], dtype=np.float64))
    assert allclose(fmap(float, get_gt_str(v, "PS")), [8.555, 11.111])

    v.set_format("PS", np.array([9998.555, 99911.111], dtype=np.float32))
    obs = fmap(float, get_gt_str(v, "PS"))
    assert allclose(obs, [9998.555, 99911.111]), obs

def test_set_format_int_a():
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    assert vcf.add_format_to_header(dict(ID="PI", Number=1, Type="Integer", Description="Int example")) == 0
    v = next(vcf)
    v.set_format("PI", np.array([5, 1], dtype=int))
    assert allclose(fmap(float, get_gt_str(v, "PI")), [5, 1])

def test_set_format_int_b():
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    assert vcf.add_format_to_header(dict(ID="PI", Number=1, Type="Integer", Description="Int example")) == 0
    v = next(vcf)

    v.set_format("PI", np.array([855, 11], dtype=np.int64))
    assert allclose(fmap(float, get_gt_str(v, "PI")), [855, 11])

def test_set_format_int_c():
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    assert vcf.add_format_to_header(dict(ID="PI", Number=1, Type="Integer", Description="Int example")) == 0
    v = next(vcf)

    v.set_format("PI", np.array([9998, 99911], dtype=np.int32))
    obs = fmap(float, get_gt_str(v, "PI"))
    assert allclose(obs, [9998, 99911]), obs

def test_set_format_int3():
    "test that we can handle multiple (in this case 3) values per sample"
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    assert vcf.add_format_to_header(dict(ID="P3", Number=3, Type="Integer", Description="Int example")) == 0
    v = next(vcf)
    exp = np.array([[1, 11, 111], [2, 22, 222]], dtype=int)
    v.set_format("P3", exp)
    res = get_gt_str(v, "P3")
    assert res == ["1,11,111", "2,22,222"], (res, str(v))

    assert np.allclose(v.format("P3"), exp)

def test_set_format_str_bytes_second_longer():
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    assert vcf.add_format_to_header(dict(ID="STR", Number=1, Type="String", Description="String example")) == 0
    v = next(vcf)

    v.set_format("STR", np.array([b'foo', b'barbaz']))
    assert np.all(v.format('STR') == np.array(['foo', 'barbaz']))

def test_set_format_str_bytes_first_longer():
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    assert vcf.add_format_to_header(dict(ID="STR", Number=1, Type="String", Description="String example")) == 0
    v = next(vcf)

    v.set_format("STR", np.array([b'foobar', b'baz']))
    assert np.all(v.format('STR') == np.array(['foobar', 'baz']))

def test_set_format_str_bytes_number3():
    # Confirm currently not supported
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    assert vcf.add_format_to_header(dict(ID="STR", Number=3, Type="String", Description="String example")) == 0
    v = next(vcf)

    contents = np.array([[b'foo', b'barbaz', b'biz'], [b'blub', b'bloop', b'blop']])
    with pytest.raises(Exception):
        v.set_format("STR", contents)

def test_set_gts():
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    v = next(vcf)

    v.genotypes = [[1, 1, True], [0, 0, False]]
    assert get_gt_str(v) == ["1|1", "0/0"]

    v.genotypes = [[-1, 1, False], [-1, 0, False]]
    assert get_gt_str(v) == ["./1", "./0"]

    v.genotypes = [[-1, -1, False], [0, 0, True]]
    assert get_gt_str(v) == ["./.", "0|0"]

    v.genotypes = [[2, 2, True], [0, 2, True]]
    assert get_gt_str(v) == ["2|2", "0|2"]

    v.genotypes = [[0, 1, 2, False], [1, 2, True]]
    s = get_gt_str(v)
    assert s == ["0/1/2", "1|2"]

    v.genotypes = [[1, 2, False], [0, 1, 2, True]]
    assert get_gt_str(v) == ["1/2", "0|1|2"]

    v.genotypes = [[0, 1, 2, False], [0, 1, 2, True]]
    assert get_gt_str(v) == ["0/1/2", "0|1|2"]

def test_info_del():
    vcf = VCF(os.path.join(HERE, "test-hemi.vcf"))
    v = next(vcf)
    d = str(v)
    assert ";DP=" in d
    del v.INFO["DP"]
    d = str(v)
    assert not ";DP=" in d

def test_filter_id():
    vcf = VCF(os.path.join(HERE, "test-hemi.vcf"))
    v = next(vcf)
    assert v.ID == "ID1"
    assert v.FILTER == "FAIL"


def get_gt_str(variant, key="GT"):
    idx = variant.FORMAT.index(key)
    return [x.split(":")[idx] for x in str(variant).strip().split("\t")[9:]]

@pytest.mark.parametrize("gts012", [False, True])
@pytest.mark.parametrize("first_phased", [False, True])
@pytest.mark.parametrize("calls", [["0"], ["."], ["0", "1", ".", "1"]])
def test_haploid_phase_accessors(tmp_path, calls, first_phased, gts012):
    path = tmp_path / "haploid.vcf"
    samples = ["sample{}".format(i) for i in range(len(calls))]
    path.write_text(
        '##fileformat=VCFv4.2\n'
        '##contig=<ID=1>\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
        + '\t'.join(samples) + '\n'
        '1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t' + '\t'.join(calls) + '\n'
    )
    with VCF(path, gts012=gts012) as vcf:
        variant = next(vcf)
        genotype = variant.genotype
        phases = [first_phased if i % 2 == 0 else not first_phased
                  for i in range(len(calls))]
        for i, phased in enumerate(phases):
            genotype[i][0].phased = phased
        variant.genotype = genotype

        alleles = [-1 if call == "." else int(call) for call in calls]
        assert variant.genotypes == [
            [allele, phased] for allele, phased in zip(alleles, phases)
        ]
        genotype = variant.genotype
        assert [genotype.phased(i) for i in range(len(calls))] == phases
        np.testing.assert_array_equal(
            genotype.array(), list(zip(alleles, phases))
        )
        np.testing.assert_array_equal(variant.gt_phases, phases)


@pytest.mark.parametrize("calls, expected", [
    (["1/2", "0|1|2"], [[1, 2, False], [0, 1, 2, True]]),
    (["1|2", "0/1/2"], [[1, 2, True], [0, 1, 2, False]]),
    (["./.", "0|1|2"], [[-1, -1, False], [0, 1, 2, True]]),
    (["0/1|2", "0|1/2"], [[0, 1, 2, False], [0, 1, 2, True]]),
])
def test_mixed_ploidy_phase_accessors(tmp_path, calls, expected):
    path = tmp_path / "mixed-ploidy.vcf"
    path.write_text(
        '##fileformat=VCFv4.2\n'
        '##contig=<ID=1>\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n'
        '1\t1\t.\tA\tC,G\t.\tPASS\t.\tGT\t' + '\t'.join(calls) + '\n'
    )
    with VCF(path) as vcf:
        variant = next(vcf)
        assert variant.genotypes == expected
        phases = [gt[-1] for gt in expected]
        genotype = variant.genotype
        assert [genotype.phased(i) for i in range(2)] == phases
        np.testing.assert_array_equal(genotype.array()[:, -1], phases)
        np.testing.assert_array_equal(variant.gt_phases, phases)


def test_padded_haploid_missing_genotype_is_not_phased(tmp_path):
    path = tmp_path / "padded-haploid.vcf"
    path.write_text(
        '##fileformat=VCFv4.3\n'
        '##contig=<ID=1>\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DI,Number=.,Type=Integer,Description="Integer list">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n'
        '1\t1\t.\tA\tC\t.\tPASS\t.\tGT:DI\t./.:.,.\t.:.\n'
    )
    with VCF(path) as vcf:
        variant = next(vcf)
        assert variant.genotypes == [[-1, -1, False], [-1, False]]
        assert [variant.genotype.phased(i) for i in range(2)] == [False, False]
        np.testing.assert_array_equal(
            variant.genotype.array(), [[-1, -1, 0], [-1, -2, 0]]
        )
        np.testing.assert_array_equal(variant.gt_phases, [False, False])


def test_access_gts():
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    """
7	55086956	.	C	G	0	.	.	GT:ADP_ALL:RULE	0/0:6728,1:F	1|1:22,1:G
7	55086957	.	T	A,C,G	0	.	.	GT:ADP_ALL:RULE	1/2:6768,2,2,1:F2,F3,F4	2|3:1,2,3,4:G2,G3,G4
7	55086958	.	T	G	0	.	.	GT:ADP_ALL:RULE	0/1/.:6768,2,2,1:F2,F3,F4	0:1,2,3,4:G2,G3,G4
7	55086959	.	T	G,T	0	.	.	GT:ADP_ALL:RULE	.	0|2:1,2,3,4:G2,G3,G4
    """

    v = next(vcf)
    gts = v.genotypes
    assert gts == [[0, 0, False], [1, 1, True]], gts

    v = next(vcf)
    assert v.genotypes == [[1, 2, False], [2, 3, True]], v.genotypes

    v = next(vcf)
    haploid_phase = bool(v.format("GT", int)[1, 0] & 1)
    assert v.genotypes == [[0, 1, -1, False], [0, haploid_phase]], v.genotypes

    v = next(vcf)
    assert v.genotypes == [[-1, False], [0, 2, True]], v.genotypes

def test_access_genotype():
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    v = next(vcf)
    gts = v.genotype
    #print(str(v), file=sys.stderr)

    # indexing directly gives a list of Allele objects (for diploid, the list
    # has length 2)
    alleles = gts[0]
    assert alleles[0].value == 0
    assert alleles[1].value == 0
    assert alleles[0].phased == False

    alleles = gts[1]
    assert alleles[0].value == 1
    assert alleles[1].value == 1
    assert alleles[1].phased == True

    assert np.all(gts.array()[:, :2] == np.array([[0, 0], [1, 1]]))
    assert np.all(gts.array()[:, 2] == np.array([False, True]))

    assert alleles[1].phased == True
    alleles[1].phased = False
    assert alleles[1].phased == False
    alleles = gts[1]
    assert alleles[1].phased == False

    assert np.all(gts.array()[:, 2] == np.array([False, False]))

    # can also just get the phased stats of the nth sample:
    assert gts.phased(0) == False
    # note this got updated above
    assert gts.phased(1) == False

    # and the alleles of the nth sample.
    assert gts.alleles(0) == [0, 0]
    #print(gts.alleles(1), file=sys.stderr)
    assert gts.alleles(1) == [1, 1]


    alleles = gts[0]
    assert alleles[0].value == 0
    alleles[0].value = 1
    assert alleles[0].value == 1
    alleles = gts[0]
    assert alleles[0].value == 1
    assert alleles[0].phased == False

    assert np.all(gts.array()[:, :2] == np.array([[1, 0], [1, 1]]))

    gts[1][0].value = 0

    assert np.all(gts.array()[:, :2] == np.array([[1, 0], [0, 1]]))

    # update the varint
    v.genotype = gts
    assert "1/0:6728,1:F	0/1:22,1:G" in str(v)

def test_access_genotype_array():
    vcf = VCF('{}/test-format-string.vcf'.format(HERE))
    """
7	55086956	.	C	G	0	.	.	GT:ADP_ALL:RULE	0/0:6728,1:F	1|1:22,1:G
7	55086957	.	T	A,C,G	0	.	.	GT:ADP_ALL:RULE	1/2:6768,2,2,1:F2,F3,F4	2|3:1,2,3,4:G2,G3,G4
7	55086958	.	T	G	0	.	.	GT:ADP_ALL:RULE	0/1/.:6768,2,2,1:F2,F3,F4	0:1,2,3,4:G2,G3,G4
7	55086959	.	T	G,T	0	.	.	GT:ADP_ALL:RULE	.	0|2:1,2,3,4:G2,G3,G4
    """

    v = next(vcf)
    np.testing.assert_array_equal(
        v.genotype.array(),
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int16)
    )

    v = next(vcf)
    np.testing.assert_array_equal(
        v.genotype.array(),
        np.array([[1, 2, 0], [2, 3, 1]], dtype=np.int16)
    )

    v = next(vcf)
    haploid_phase = v.format("GT", int)[1, 0] & 1
    np.testing.assert_array_equal(
        v.genotype.array(),
        np.array([[0, 1, -1, 0], [0, -2, -2, haploid_phase]], dtype=np.int16)
    )

    v = next(vcf)
    np.testing.assert_array_equal(
        v.genotype.array(),
        np.array([[-1, -2, 0], [0, 2, 1]], dtype=np.int16)
    )

    # test fill value
    np.testing.assert_array_equal(
        v.genotype.array(fill=-9),
        np.array([[-1, -9, 0], [0, 2, 1]], dtype=np.int16)
    )

def test_alt_homozygous_gt():
    vcf = VCF(os.path.join(HERE, "test-multiallelic-homozygous-alt.vcf.gz"))
    assert vcf is not None
    v = next(vcf)
    assert v
    assert v.gt_bases[0] == '<*:DEL>/<*:DEL>'

    vcf = VCF(os.path.join(HERE, "test-multiallelic-homozygous-alt.vcf.gz"), gts012=True)
    assert vcf is not None
    v = next(vcf)
    assert v
    assert v.gt_bases[0] == '<*:DEL>/<*:DEL>'

def test_write_missing_contig():
    input_vcf = VCF("{}/seg.vcf.gz".format(HERE))
    if platform.system() == "Windows":
        output_vcf = "__o.vcf"
    else:
        output_vcf = "/dev/null"
    output_vcf = Writer(output_vcf, input_vcf)
    for v in input_vcf:
        v.genotypes = [[1, 1, False]]
        output_vcf.write_record(v)
    output_vcf.close()

    if platform.system() == "Windows":
        os.unlink("__o.vcf")

def test_set_samples():
    vcf = VCF(VCF_PATH)
    assert len(vcf.samples) == 189, len(vcf.samples)
    vcf.set_samples([vcf.samples[2]])
    assert len(vcf.samples) == 1
    v = next(vcf)
    assert len(v.gt_types) == 1

def test_hrec():

    vcf = VCF(VCF_PATH)
    for item in vcf.header_iter():
        info = item.info()
        if info['HeaderType'] != 'GENERIC':
            assert 'ID' in info

def test_issue44():
    vcf = VCF('{}/issue_44.vcf'.format(HERE))
    w = Writer('__o.vcf', vcf)
    for v in vcf:
        tmp = v.genotypes
        #print(tmp, file=sys.stderr)
        v.genotypes = tmp
        w.write_record(v)
    w.close()
    #           "./."            "."          ".|."           "0|0"
    expected = [[-1, -1, False], [-1, False], [-1, -1, True], [0, 0, True]]
    #print("", file=sys.stderr)

    t = VCF('__o.vcf')
    for i, v in enumerate(t):
        #print(v.genotypes, file=sys.stderr)
        assert v.genotypes == [expected[i]], (i, v.genotypes, expected[i])

    # file should be closed before delete
    t.close()
    os.unlink("__o.vcf")

def test_id_field_updates():
    # 1 10172   .       CCCTAA  C       92.0    PASS
    v = VCF(VCF_PATH)
    variant = next(v)
    assert variant.ID is None, variant.ID

    variant.ID = 'foo'
    assert variant.ID == 'foo', variant.ID

    variant.ID = 100
    assert variant.ID == '100', variant.ID

    variant.ID = 100.1
    assert variant.ID == '100.1', variant.ID

    variant.ID = '.'
    assert variant.ID is None, variant.ID

    variant.ID = None
    assert variant.ID is None, variant.ID

def test_set_pos():
    test_vcf = '{}/test-strict-gt-option-flag.vcf.gz'.format(HERE)
    vcf = VCF(test_vcf, gts012=False)
    v = next(vcf)

    orig_pos, orig_start = v.POS, v.start
    v.set_pos(22)
    assert v.start == 22
    assert v.POS == 23

def test_set_chrom_when_contig_not_in_header():
    test_vcf = '{}/test-strict-gt-option-flag.vcf.gz'.format(HERE)
    new_chrom = "NEW"
    vcf = VCF(test_vcf, gts012=False)
    original_seqnames = vcf.seqnames
    assert new_chrom not in original_seqnames
    v = next(vcf)

    with warnings.catch_warnings(record=True) as w:
        v.CHROM = new_chrom
        assert len(w) == 1
        assert "added new contig" in str(w[-1].message)
    assert v.CHROM == new_chrom
    expected_seqnames = sorted(original_seqnames + [new_chrom])
    assert vcf.seqnames == expected_seqnames

def test_set_chrom_after_contig_is_added_to_header():
    test_vcf = '{}/test-strict-gt-option-flag.vcf.gz'.format(HERE)
    new_chrom = "NEW"
    vcf = VCF(test_vcf, gts012=False)
    original_seqnames = vcf.seqnames
    vcf.add_to_header("##contig=<ID={},length=15>".format(new_chrom))
    expected_seqnames = sorted(original_seqnames + [new_chrom])
    assert vcf.seqnames == expected_seqnames
    v = next(vcf)

    v.CHROM = new_chrom
    assert v.CHROM == new_chrom

def test_set_qual():
    v = VCF(VCF_PATH)
    variant = next(v)
    assert variant.QUAL == 92.0

    variant.QUAL = 30.0
    assert variant.QUAL == 30.0

    with pytest.raises(TypeError):
        variant.QUAL = "30.0"

    variant.QUAL = None
    assert variant.QUAL is None, 'variant.QUAL is {}'.format(variant.QUAL)

def test_strict_gt_option_flag():
    test_vcf = '{}/test-strict-gt-option-flag.vcf.gz'.format(HERE)

     #T, C
     #0/0     1/1     0/1     1/0     1/.     ./1     0/.     ./0     ./.

    truth_gt_bases = ('T/T', 'C/C', 'T/C', 'C/T', 'C/.', './C', 'T/.', './T', './.')
    truth_genotypes = (
        [0, 0, False],
        [1, 1, False],
        [0, 1, False],
        [1, 0, False],
        [1, -1, False],
        [-1, 1, False],
        [0, -1, False],
        [-1, 0, False],
        [-1, -1, False],
    )

    vcf = VCF(test_vcf, gts012=False)
    variant = next(vcf)

    msg = "VCF(gts012=False, strict_gt=False) not working"
    truth_gt_types = (0, 3, 1, 1, 1, 1, 0, 0, 2)
    print(variant.gt_bases.tolist(), file=sys.stderr)
    print(variant.gt_types.tolist(), file=sys.stderr)
    assert bool(tuple(variant.gt_bases.tolist()) == truth_gt_bases), '{} [gt_bases]'.format(msg)

    """
    assert bool(tuple(variant.gt_types.tolist()) == truth_gt_types), '{} [gt_types]'.format(msg)
    assert bool(tuple(variant.genotypes) == truth_genotypes), '{} (genotypes)'.format(msg)

    vcf = VCF(test_vcf, gts012=False, strict_gt=True)
    variant = next(vcf)

    msg = "VCF(gts012=False, strict_gt=True) not working"
    truth_gt_types = (0, 3, 1, 1, 2, 2, 2, 2, 2)
    assert bool(tuple(variant.gt_bases.tolist()) == truth_gt_bases), '{} [gt_bases]'.format(msg)
    assert bool(tuple(variant.gt_types.tolist()) == truth_gt_types), '{} [gt_types]'.format(msg)
    assert bool(tuple(variant.genotypes) == truth_genotypes), '{} (genotypes)'.format(msg)


    vcf = VCF(test_vcf, gts012=True)
    variant = next(vcf)

    msg = "VCF(gts012=True, strict_gt=False) not working"
    truth_gt_types = (0, 2, 1, 1, 1, 1, 0, 0, 3)
    assert tuple(variant.gt_bases.tolist()) == truth_gt_bases, '{} [gt_bases]'.format(msg)
    #sys.stderr.write("\nobs:%s\n" % variant.gt_types.tolist())
    #sys.stderr.write("exp:%s\n" % list(truth_gt_types))
    assert tuple(variant.gt_types.tolist()) == truth_gt_types, '{} [gt_types]'.format(msg)
    assert tuple(variant.genotypes) == truth_genotypes, '{} (genotypes)'.format(msg)

    vcf = VCF(test_vcf, gts012=True, strict_gt=True)
    variant = next(vcf)

    msg = "VCF(gts012=True, strict_gt=True) not working"
    truth_gt_types = (0, 2, 1, 1, 3, 3, 3, 3, 3)
    assert tuple(variant.gt_bases.tolist()) == truth_gt_bases, '{} [gt_bases]'.format(msg)
    assert tuple(variant.gt_types.tolist()) == truth_gt_types, '{} [gt_types]'.format(msg)
    assert tuple(variant.genotypes) == truth_genotypes, '{} (genotypes)'.format(msg)
    """

def test_alt_repr():
    v = os.path.join(HERE, "test-alt-repr.vcf")
    vcf = VCF(v, gts012=True, strict_gt=False)
    v = next(vcf)
    assert np.all(v.gt_types == np.array([0, 1, 2, 0, 1, 3]))

    v = os.path.join(HERE, "test-alt-repr.vcf")
    vcf = VCF(v, gts012=False, strict_gt=False)
    v = next(vcf)
    assert np.all(v.gt_types == np.array([0, 1, 3, 0, 1, 2]))


"""
def test_seqlens():
    v = VCF(VCF_PATH)
    assert v.seqlens == [249250621, 243199373, 198022430, 191154276,
        180915260, 171115067, 159138663, 146364022, 141213431, 135534747,
        135006516, 133851895, 115169878, 107349540, 102531392, 90354753,
        81195210, 78077248, 59128983, 63025520, 48129895, 51304566, 155270560,
        59373566, 16569, 4262, 15008, 19913, 27386, 27682, 33824, 34474, 36148,
        36422, 36651, 37175, 37498, 38154, 38502, 38914, 39786, 39929, 39939,
        40103, 40531, 40652, 41001, 41933, 41934, 42152, 43341, 43523, 43691,
        45867, 45941, 81310, 90085, 92689, 106433, 128374, 129120, 137718,
        155397, 159169, 161147, 161802, 164239, 166566, 169874, 172149, 172294,
        172545, 174588, 179198, 179693, 180455, 182896, 186858, 186861, 187035,
        189789, 191469, 211173, 547496, 171823, 35477943, 5386], v.seqlens
        """

def test_closed_iter():
    path = os.path.join(HERE, "test-alt-repr.vcf")
    vcf = VCF(path, gts012=True, strict_gt=False)
    vcf.close()

    with pytest.raises(Exception):
        next(vcf)

def test_issue72():
    path = os.path.join(HERE, "test-alt-repr.vcf")
    vcf = VCF(path, gts012=True, strict_gt=False)

    v = next(vcf)
    assert v.INFO['DQ'] == 1

    assert v.format('DQ') is not None

def test_is_transition():
    vcf = VCF(VCF_ALTFREQ_PATH)

    for r in vcf:
        assert r.is_transition

def test_decomposed():
    vcf = VCF(os.path.join(HERE, "decomposed.vcf"))
    v = next(vcf)
    #0/.	./0	1/.	./1	./.
    assert np.all(v.gt_types == np.array([vcf.HOM_REF, vcf.HOM_REF, vcf.HET, vcf.HET, vcf.UNKNOWN]))


def test_fd():

    fh = open(os.path.join(HERE, "decomposed.vcf"))
    fn = fh.fileno()

    vcf = VCF(fn)
    v = next(vcf)
    assert np.all(v.gt_types == np.array([vcf.HOM_REF, vcf.HOM_REF, vcf.HET, vcf.HET, vcf.UNKNOWN]))
    fh.close()
    vcf.close()


def test_set_reference():
    fh = open(os.path.join(HERE, "decomposed.vcf"))
    fn = fh.fileno()

    vcf = VCF(fn)
    for v in vcf:
      v.REF = "CCCCA"
      assert  "CCCCA" in str(v)

def test_set_alternates():
    fh = open(os.path.join(HERE, "decomposed.vcf"))
    fn = fh.fileno()

    vcf = VCF(fn)
    for v in vcf:
      v.ALT = "TTT,GGG"
      assert  "TTT,GGG" in str(v)

      v.ALT = ["AAAC", "CCCA"]
      assert "AAAC,CCCA" in str(v)

def test_no_seqlen():

    vcf_path = os.path.join(HERE, "no-seq-len.vcf")
    vcf = VCF(vcf_path)
    assert vcf.seqnames == ["3"]
    with pytest.raises(AttributeError):
        vcf.seqlens

def test_set_unknown_format():
    vcf = VCF(VCF_PATH)
    vcf.add_format_to_header({'ID':'NEW', 'Type':'Float', 'Number':1, 'Description':'...'})

    v = next(vcf)
    arr = np.array([[1.1] for s in vcf.samples])
    arr[-1][0] = np.nan
    v.set_format('NEW', arr)
    record = str(v)
    parts = record.split()
    assert parts[-1][-1] == '.'
    assert parts[-2][-3:] == '1.1'

def test_invalid_header():
    # htslib produces the error "Empty sample name: trailing spaces/tabs in the header line?"
    p = os.path.join(HERE, "test-invalid-header.vcf")
    assert os.path.exists(p)
    with pytest.raises(Exception):
        VCF(p)


def test_genotypes():
    """
. 1
./. ./1
./. 1
0/1 0
0 0/1
     """
    exp_array = [[-1,  0],
       [-1, -1,  0],
       [-1, -1,  0],
       [0, 1,  0],
       [0, -2, 1],
       ]

    non_strict_exp_num = [
    [0, 0, 1, 1],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    ]

    strict_exp_num = [x[:] for x in non_strict_exp_num]
    strict_exp_num[1] = [0, 0, 2, 0] # both unknown

    for strict_gt in (False, True):
        vcf = VCF(os.path.join(HERE, "test-genotypes.vcf"), strict_gt=strict_gt)

        exp_num = strict_exp_num if strict_gt else non_strict_exp_num


        for i, v in enumerate(vcf):
            #if i != 3: continue
            obs = [v.num_hom_ref, v.num_het, v.num_unknown, v.num_hom_alt]
            assert obs == exp_num[i], ("error with num_*")

            a = v.genotype.array()[0] # only 0'th item
            print("i:", i, " a:", v.genotype.array()[0], " exp:", exp_array[i])
            expected = exp_array[i][:]
            if i == 4:
                expected[-1] = v.format("GT", int)[0, 0] & 1
            assert (a == expected).all(), " error with array"

def test_issue236():


    for strict_gt in (False, True):
        res = [[], []]

        for gts in (False, True):
            vcf = VCF(os.path.join(HERE, "test-genotypes.vcf"), gts012=gts, strict_gt=strict_gt)
            for v in vcf:
                res[int(gts)].append(v.num_unknown)

        assert res[0] == res[1]
        assert len(res[0]) > 0


def test_issue17_no_gt():
    vcf = VCF(os.path.join(HERE, "test-no-genotypes.vcf"))
    with pytest.raises(Exception):
        for v in vcf:
            v.num_called  # Used to give segmentation fault


@pytest.mark.parametrize("path", [
    "test.vcf.gz",
    "test-multiallelic-homozygous-alt.vcf.gz",
    "test-strict-gt-option-flag.vcf.gz",
    "test-strict-gt-option-flag.vcf.gz",
    "multi-contig.vcf.gz",
    "multi-contig.bcf",
    "test.snpeff.bcf",
    ])
def test_num_records_indexed(path):
    vcf = VCF(os.path.join(HERE, path))
    n = len(list(vcf))
    assert n == vcf.num_records
    vcf = VCF(os.path.join(HERE, path))
    assert n == vcf.num_records

@pytest.mark.parametrize("suffix", ["csi", "tbi"])
def test_num_records_indexed_csi_tabix(suffix):
    path = "multi-contig.vcf.gz"
    index_file = os.path.join(HERE, "multi-contig.vcf.gz.{}".format(suffix))
    vcf = VCF(os.path.join(HERE, path))
    n = len(list(vcf))
    # Explicitly set the index
    vcf.set_index(index_file)
    assert n == vcf.num_records
    vcf = VCF(os.path.join(HERE, path))
    vcf.set_index(index_file)
    assert n == vcf.num_records

def test_num_records_set_index_multiple_times():
    path = os.path.join(HERE, "multi-contig.vcf.gz")
    csi_index = path + ".csi"
    tbi_index = path + ".tbi"
    vcf = VCF(path)
    n = len(list(vcf))
    assert n == vcf.num_records
    vcf.set_index(csi_index)
    assert n == vcf.num_records

    vcf = VCF(path)
    assert n == vcf.num_records
    vcf.set_index(tbi_index)
    assert n == vcf.num_records

    vcf = VCF(path)
    vcf.set_index(csi_index)
    assert n == vcf.num_records

    vcf = VCF(path)
    for _ in range(10):
        vcf.set_index(csi_index)
        assert n == vcf.num_records
        vcf.set_index(tbi_index)
        assert n == vcf.num_records

def test_num_records_set_wrong_index():
    path = os.path.join(HERE, "multi-contig.vcf.gz")
    index = os.path.join(HERE, "test.vcf.gz.tbi")
    vcf = VCF(path)
    vcf.set_index(index)
    # We compute the number of records from the index, and don't report an
    # error
    assert vcf.num_records == 115
    assert vcf.num_records != len(list(vcf))

@pytest.mark.parametrize("path", [
    "test-genotypes.vcf",
    ])
def test_num_records_no_index(path):
    vcf = VCF(os.path.join(HERE, path))
    with pytest.raises(ValueError, match="must be indexed"):
        vcf.num_records

@pytest.mark.parametrize("mode", ["vcf", "wz", "wb", "raw_bcf"])
@pytest.mark.parametrize("values,logical,width", [
    (("longword", "x,y"), {"min": 1, "max": 2}, 9),
    (("a,b", "x"), {"min": 1, "max": 2}, 4),
    (("a%2Cb", "x%252Cy"), {"min": 1, "max": 1}, 8),
    (("\u00e9,\u03bb", "x"), {"min": 1, "max": 2}, 6),
    ((".,.", "."), {"min": 1, "max": 2}, 4),
])
def test_schema_scan_logical_string_cardinality(tmp_path, mode, values, logical, width):
    path = tmp_path / "input.vcf"
    path.write_text(
        '##fileformat=VCFv4.3\n'
        '##contig=<ID=1>\n'
        '##INFO=<ID=IS,Number=.,Type=String,Description="Strings">\n'
        '##INFO=<ID=IC,Number=2,Type=Character,Description="Characters">\n'
        '##INFO=<ID=US,Number=.,Type=String,Description="Unused">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
        '##FORMAT=<ID=VS,Number=.,Type=String,Description="Strings">\n'
        '##FORMAT=<ID=VC,Number=.,Type=Character,Description="Characters">\n'
        '##FORMAT=<ID=UV,Number=.,Type=String,Description="Unused">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n'
        '1\t1\t.\tA\tC\t.\tPASS\tIS=%s;IC=a,b\tGT:VS:VC\t0/1:%s:a,b\t0/0:%s:c\n'
        % (values[0], values[0], values[1]),
        encoding="utf-8",
    )
    if mode != "vcf":
        source = VCF(str(path))
        output = str(tmp_path / ("input.bcf" if mode in ("wb", "raw_bcf") else "input.vcf.gz"))
        writer = Writer(output, source, mode="wb" if mode == "raw_bcf" else mode)
        for variant in source:
            writer.write_record(variant)
        writer.close()
        source.close()
        path = output
        if mode == "raw_bcf":
            raw = tmp_path / "raw.bcf"
            with gzip.open(path, "rb") as stream:
                raw.write_bytes(stream.read())
            path = raw

    summary = VCF(str(path)).schema_scan(threads=1)
    fields = summary["fields"]
    assert fields["FORMAT"]["VS"]["logical_cardinality"] == logical
    assert fields["FORMAT"]["VS"]["cardinality"] == {"min": width, "max": width}
    info_count = values[0].count(",") + 1
    assert fields["INFO"]["IS"]["logical_cardinality"] == {"min": info_count, "max": info_count}
    assert fields["INFO"]["IC"]["logical_cardinality"] == {"min": 2, "max": 2}
    assert fields["FORMAT"]["VC"]["logical_cardinality"] == {"min": 1, "max": 2}
    for section, name in (("INFO", "US"), ("FORMAT", "UV")):
        assert fields[section][name]["logical_cardinality"] == {"min": None, "max": None}
    assert "logical_cardinality" not in fields["FORMAT"]["GT"]
    assert summary["ploidy"] == {2: 2}


@pytest.mark.parametrize("mode", ["wz", "wb", "vcf", "raw_bcf"])
def test_schema_scan_logical_string_cardinality_parallel(tmp_path, mode):
    path = tmp_path / "input.vcf"
    path.write_text(
        '##fileformat=VCFv4.3\n'
        '##contig=<ID=1>\n'
        '##INFO=<ID=PAD,Number=1,Type=String,Description="Long scalar">\n'
        '##FORMAT=<ID=VS,Number=.,Type=String,Description="Strings">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n'
        + "".join(
            "1\t%d\t.\tA\tC\t.\tPASS\tPAD=%s\tVS\t%s\t%s\n"
            % (i + 1, "a" * 4000, "longword" if i < 50 else "a,b,c", "x,y")
            for i in range(100)
        )
    )
    if mode != "vcf":
        source = VCF(str(path))
        output = str(tmp_path / ("input.vcf.gz" if mode == "wz" else "input.bcf"))
        writer = Writer(output, source, mode="wb" if mode == "raw_bcf" else mode)
        for variant in source:
            writer.write_record(variant)
        writer.close()
        source.close()
        path = output
        if mode == "raw_bcf":
            raw = tmp_path / "raw.bcf"
            with gzip.open(path, "rb") as stream:
                raw.write_bytes(stream.read())
            path = raw
    serial = VCF(str(path)).schema_scan(threads=1)
    parallel = VCF(str(path)).schema_scan(threads=1, workers=3)
    assert parallel == serial
    assert parallel["fields"]["FORMAT"]["VS"]["logical_cardinality"] == {"min": 1, "max": 3}
    assert parallel["fields"]["INFO"]["PAD"]["logical_cardinality"] == {"min": 1, "max": 1}


@pytest.mark.parametrize("filename,suffix", [
    ("multi-contig.vcf.gz", "csi"),
    ("multi-contig.vcf.gz", "tbi"),
    ("multi-contig.bcf", "csi"),
])
@pytest.mark.parametrize("threads", [0, 1, 4])
def test_build_index_queries(tmp_path, filename, suffix, threads):
    from cyvcf2 import build_index
    path = os.path.join(HERE, filename)
    with VCF(path) as source:
        expected = [(v.CHROM, v.POS, v.end) for v in source]
    destination = tmp_path / ("native." + suffix)
    assert build_index(path, destination, threads=threads) == str(destination)
    with VCF(path) as reader:
        reader.set_index(str(destination))
        assert reader.num_records == len(expected)
        for chrom, pos, end in (expected[0], expected[len(expected) // 2], expected[-1]):
            assert (chrom, pos, end) in [
                (v.CHROM, v.POS, v.end) for v in reader("%s:%d-%d" % (chrom, pos, pos))
            ]
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_build_index_default_and_no_overwrite(tmp_path):
    from cyvcf2 import build_index
    path = tmp_path / "input.vcf.gz"
    shutil.copyfile(VCF_PATH, path)
    destination = build_index(path)
    original = Path(destination).read_bytes()
    with pytest.raises(FileExistsError):
        build_index(os.fsencode(path))
    assert Path(destination).read_bytes() == original
    with VCF(str(path)) as reader:
        assert reader.num_records == sum(1 for _ in reader)
    Path(destination).write_bytes(b"invalid index")
    with pytest.raises(FileExistsError):
        build_index(path)
    assert Path(destination).read_bytes() == b"invalid index"
    Path(destination).unlink()
    Path(destination).symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        build_index(path)
    assert Path(destination).is_symlink()


def test_build_index_concurrent_destination(tmp_path, monkeypatch):
    import cyvcf2.cyvcf2 as native
    destination = tmp_path / "race.csi"
    original_link = os.link

    def competing_link(source, target):
        destination.write_bytes(b"other writer")
        return original_link(source, target)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FileExistsError):
        native.build_index(VCF_PATH, destination)
    assert destination.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


@pytest.mark.parametrize("threads", [-1, 1.5, True, None, "4"])
def test_build_index_invalid_threads(tmp_path, threads):
    from cyvcf2 import build_index
    with pytest.raises(ValueError, match="non-negative integer"):
        build_index(VCF_PATH, tmp_path / "bad.csi", threads=threads)
    assert not list(tmp_path.iterdir())


def test_build_index_errors(tmp_path):
    from cyvcf2 import build_index
    destination = tmp_path / "bad.csi"
    with pytest.raises(OSError, match="existing local file"):
        build_index(tmp_path / "missing.vcf.gz", destination)
    with pytest.raises(ValueError, match="BGZF"):
        build_index(VCF_PATH2, destination)
    ordinary_gzip = tmp_path / "ordinary.vcf.gz"
    ordinary_gzip.write_bytes(gzip.compress(Path(VCF_PATH2).read_bytes()))
    with pytest.raises(ValueError, match="BGZF"):
        build_index(ordinary_gzip, destination)
    with pytest.raises(OSError, match="BCF requires"):
        build_index(os.path.join(HERE, "multi-contig.bcf"), tmp_path / "bad.tbi")
    with pytest.raises(ValueError, match="end in"):
        build_index(VCF_PATH, tmp_path / "bad.idx")
    with pytest.raises(ValueError, match="local"):
        build_index("unsupported://example.org/input.vcf.gz", destination)
    with pytest.raises(ValueError, match="destination"):
        build_index(VCF_PATH, "https://example.org/input.csi")
    with pytest.raises(ValueError, match="NUL"):
        build_index(VCF_PATH + "\0", destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_build_index_unsorted(tmp_path):
    from cyvcf2 import build_index
    path = str(tmp_path / "unsorted.vcf.gz")
    with VCF(VCF_PATH2) as source:
        records = list(source)
        with Writer(path, source) as writer:
            for record in reversed(records):
                writer.write_record(record)
    destination = tmp_path / "bad.csi"
    with pytest.raises(OSError, match="coordinate-sorted"):
        build_index(path, destination, threads=2)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


@pytest.mark.parametrize("kind", ["empty", "tabix_only", "wrong_format", "truncated"])
@pytest.mark.parametrize("workers", [1, 2])
def test_build_index_tabix_semantics(tmp_path, kind, workers):
    import struct
    from cyvcf2 import build_index
    import cyvcf2.cyvcf2 as native
    if workers > 1 and not native._coordinate_workers_available:
        pytest.skip("coordinate workers require bundled HTSlib")
    header = (
        '##fileformat=VCFv4.3\n'
        '##contig=<ID=1,length=1000>\n'
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n'
    )
    text = header
    if kind == "tabix_only":
        text += (
            '1\t1\t.\tA\t<DEL>\t.\tPASS\tEND=100\tGT\tinvalid\n'
            '1\t200\t.\tA\tC\t.\tPASS\t.\tGT\tinvalid\n'
            'unparsable\n'
        )
    elif kind == "wrong_format":
        text = "not a VCF\n"
    compressed = gzip.compress(text.encode(), mtime=0)
    block_size = len(compressed) + 8
    block = (
        compressed[:3] + b"\x04" + compressed[4:10]
        + b"\x06\x00BC\x02\x00" + struct.pack("<H", block_size - 1)
        + compressed[10:]
    )
    path = tmp_path / "input.vcf.gz"
    path.write_bytes(block[:-5] if kind == "truncated" else block)
    destination = tmp_path / "output.csi"
    if kind in ("wrong_format", "truncated"):
        with pytest.raises((OSError, ValueError)):
            build_index(path, destination, workers=workers)
        assert not destination.exists()
    else:
        build_index(path, destination, threads=2, workers=workers)
        with VCF(str(path), samples=[]) as reader:
            reader.set_index(str(destination))
            assert reader.num_records == (0 if kind == "empty" else 2)
            if kind == "tabix_only":
                assert next(reader("1:50-50")).POS == 1
                assert next(reader("1:200-200")).POS == 200
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def _write_index_test_bgzf(path, data):
    import struct
    with path.open("wb") as stream:
        for start in range(0, len(data), 32000):
            compressed = gzip.compress(data[start:start + 32000], mtime=0)
            stream.write(
                compressed[:3] + b"\x04" + compressed[4:10]
                + b"\x06\x00BC\x02\x00" + struct.pack("<H", len(compressed) + 7)
                + compressed[10:]
            )
        stream.write(bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000"))


@pytest.fixture
def coordinate_vcf(tmp_path, monkeypatch):
    import cyvcf2.cyvcf2 as native
    from cyvcf2 import _index
    if not native._coordinate_workers_available:
        pytest.skip("coordinate workers require bundled HTSlib")
    monkeypatch.setattr(_index, "_COMPRESSED_CHUNK_BYTES", 512)
    header = (
        '##fileformat=VCFv4.3\n'
        '##contig=<ID=1,length=10000>\n'
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End">\n'
        '##INFO=<ID=PAD,Number=1,Type=String,Description="Padding">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n'
    )
    rows = []
    for chrom in ("1", "undeclared"):
        for i in range(1, 401):
            alt = "<DEL>" if i % 47 == 0 else ("<NON_REF>" if i % 31 == 0 else "C")
            padding = "a" * (300000 if i == 150 else 1024)
            rows.append(
                f"{chrom}\t{i}\tr{i}\tA\t{alt}\t.\tPASS\tEND={i + 90};PAD={padding}\tGT\tinvalid\n"
            )
    data = (header + "".join(rows)).encode()
    path = tmp_path / "coordinate.vcf.gz"
    _write_index_test_bgzf(path, data)
    return path


@pytest.mark.parametrize("workers,threads", [(2, 0), (4, 1), (8, 0)])
@pytest.mark.parametrize("suffix", ["csi", "tbi"])
def test_build_index_parallel_boundaries(coordinate_vcf, tmp_path, workers, threads, suffix):
    from cyvcf2 import build_index
    from cyvcf2 import _index
    import cyvcf2.cyvcf2 as native
    reference = tmp_path / ("serial." + suffix)
    destination = tmp_path / ("parallel." + suffix)
    build_index(coordinate_vcf, reference)
    builder = native._CoordinateIndexBuilder(os.fsencode(coordinate_vcf), os.fsencode(destination))
    ranges = list(_index._ranges(os.fsencode(coordinate_vcf), builder.offset0))
    assert len(ranges) > workers
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    build_index(coordinate_vcf, destination, workers=workers, threads=threads)
    assert gzip.decompress(destination.read_bytes()) == gzip.decompress(reference.read_bytes())
    with VCF(str(coordinate_vcf), samples=[]) as serial, VCF(str(coordinate_vcf), samples=[]) as parallel:
        serial.set_index(str(reference))
        parallel.set_index(str(destination))
        assert serial.num_records == parallel.num_records == 800
        for chrom in ("1", "undeclared"):
            for begin, end in [(1, 490), (1, 1), (150, 150), (400, 490), (500, 600)] + [
                (position, position + 5) for position in range(20, 400, 17)
            ]:
                region = "%s:%d-%d" % (chrom, begin, end)
                expected = [(v.CHROM, v.POS, v.end, v.ID, v.ALT) for v in serial(region)]
                assert [(v.CHROM, v.POS, v.end, v.ID, v.ALT) for v in parallel(region)] == expected


def test_build_index_parallel_rejects_bad_handoff(coordinate_vcf, tmp_path, monkeypatch):
    from cyvcf2 import build_index, _index
    original = _index._ranges

    def damaged_ranges(path, start):
        for begin, end in original(path, start):
            yield begin, end + 1 if end != _index._EOF else end

    monkeypatch.setattr(_index, "_ranges", damaged_ranges)
    destination = tmp_path / "bad.csi"
    with pytest.raises(OSError, match="handoff"):
        build_index(coordinate_vcf, destination, workers=4)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_build_index_parallel_no_overwrite(coordinate_vcf, tmp_path, monkeypatch):
    from cyvcf2 import build_index
    destination = tmp_path / "race.csi"
    original = os.link

    def competing_link(source, target):
        destination.write_bytes(b"other writer")
        original(source, target)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FileExistsError):
        build_index(coordinate_vcf, destination, workers=2)
    assert destination.read_bytes() == b"other writer"
    with pytest.raises(FileExistsError):
        build_index(coordinate_vcf, destination, workers=2)
    assert destination.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_build_index_parallel_format_and_worker_errors(coordinate_vcf, tmp_path):
    from cyvcf2 import build_index
    destination = tmp_path / "bad.csi"
    for workers in (0, -1, True, 1.5, None):
        with pytest.raises(ValueError, match="positive integer"):
            build_index(coordinate_vcf, destination, workers=workers)
    for filename in (VCF_PATH2, os.path.join(HERE, "multi-contig.bcf")):
        with pytest.raises(ValueError, match="BGZF-compressed VCF"):
            build_index(filename, destination, workers=2)
    assert not destination.exists()


def test_build_index_parallel_batch_limit(coordinate_vcf, tmp_path, monkeypatch):
    from cyvcf2 import build_index, _index
    monkeypatch.setattr(_index, "_COMPRESSED_CHUNK_BYTES", 1 << 30)
    path = tmp_path / "dense.vcf.gz"
    text = (
        "##fileformat=VCFv4.3\n##contig=<ID=1,length=100000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        + "".join("1\t%d\t.\tA\tC\t.\tPASS\t.\n" % i for i in range(1, 70001))
    )
    _write_index_test_bgzf(path, text.encode())
    serial, parallel = tmp_path / "serial.csi", tmp_path / "parallel.csi"
    build_index(path, serial)
    build_index(path, parallel, workers=4)
    assert gzip.decompress(serial.read_bytes()) == gzip.decompress(parallel.read_bytes())
    with VCF(str(path)) as reader:
        reader.set_index(str(parallel))
        assert reader.num_records == 70000
        assert [v.POS for v in reader("1:65535-65538")] == list(range(65535, 65539))


def test_build_index_parallel_unsorted_and_truncated(coordinate_vcf, tmp_path):
    from cyvcf2 import build_index
    data = gzip.decompress(coordinate_vcf.read_bytes())
    _write_index_test_bgzf(coordinate_vcf, data.replace(b"1\t400\t", b"1\t1\t"))
    destination = tmp_path / "bad.csi"
    with pytest.raises(OSError, match="coordinate-sorted"):
        build_index(coordinate_vcf, destination, workers=4)
    _write_index_test_bgzf(coordinate_vcf, data)
    coordinate_vcf.write_bytes(coordinate_vcf.read_bytes()[:-50])
    with pytest.raises(OSError):
        build_index(coordinate_vcf, destination, workers=4)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_schema_scan_summary():
    path = os.path.join(HERE, "test.snpeff.vcf")
    vcf = VCF(path)
    expected_records = sum(1 for _ in VCF(path))
    expected_next_pos = list(VCF(path))[2].POS
    next(vcf)
    next(vcf)
    summary = vcf.schema_scan(threads=1)

    assert summary["records"] == expected_records
    assert summary["samples"] == len(vcf.samples)
    assert summary["index"] == {"available": False, "built": False, "path": None}
    assert sum(summary["alleles_per_record"].values()) == expected_records
    assert sum(summary["ploidy"].values()) > 0
    assert summary["fields"]["INFO"]["DP"]["numeric_bounds"] is not None
    assert summary["fields"]["FORMAT"]["GT"]["cardinality"]["max"] >= 1
    assert summary["fields"]["FORMAT"]["GT"]["total_values"] >= expected_records * len(vcf.samples)
    assert next(vcf).POS == expected_next_pos

def test_schema_scan_builds_csi_and_tbi_indexes(tmp_path):
    path = os.path.join(HERE, "multi-contig.vcf.gz")
    expected_records = sum(1 for _ in VCF(path))
    first_record = next(VCF(path))

    for suffix in ("csi", "tbi"):
        index_path = str(tmp_path / ("scan." + suffix))
        summary = VCF(path).schema_scan(index_path=index_path, threads=2)
        assert summary["records"] == expected_records
        assert summary["index"] == {
            "available": True,
            "built": True,
            "path": index_path,
        }

        indexed_vcf = VCF(path)
        indexed_vcf.set_index(index_path)
        assert indexed_vcf.num_records == expected_records
        region = "%s:%d-%d" % (
            first_record.CHROM, first_record.POS, first_record.POS)
        assert list(indexed_vcf(region))
        reused = VCF(path).schema_scan(index_path=index_path)
        assert reused["index"] == {
            "available": True,
            "built": False,
            "path": index_path,
        }

def test_schema_scan_csi_index_is_discovered_for_region_queries(tmp_path):
    source_path = os.path.join(HERE, "multi-contig.vcf.gz")
    path = str(tmp_path / "input.vcf.gz")
    shutil.copyfile(source_path, path)
    first_record = next(VCF(path))
    VCF(path).schema_scan(index_path=path + ".csi")

    region = "%s:%d-%d" % (
        first_record.CHROM, first_record.POS, first_record.POS)
    assert list(VCF(path)(region))

@pytest.mark.parametrize("suffix", ["csi", "tbi"])
def test_schema_scan_indexes_only_contigs_with_records(tmp_path, suffix):
    source = VCF(os.path.join(HERE, "multi-contig.vcf.gz"))
    record = next(record for record in source if record.CHROM == "20")
    path = str(tmp_path / "single-contig.vcf.gz")
    writer = Writer(path, source)
    writer.write_record(record)
    writer.close()

    index_path = path + "." + suffix
    summary = VCF(path).schema_scan(index_path=index_path)
    assert summary["records"] == 1
    region = "%s:%d-%d" % (record.CHROM, record.POS, record.POS)
    assert len(list(VCF(path)(region))) == 1

def test_schema_scan_reuses_existing_index():
    path = VCF_PATH
    index_path = path + ".tbi"
    summary = VCF(path).schema_scan(index_path=index_path)

    assert summary["index"] == {
        "available": True,
        "built": False,
        "path": index_path,
    }

def test_schema_scan_rejects_unsorted_index_input(tmp_path):
    source = VCF(os.path.join(HERE, "test.snpeff.vcf"))
    records = list(source)
    output_path = str(tmp_path / "unsorted.vcf.gz")
    writer = Writer(output_path, source)
    for record in reversed(records):
        writer.write_record(record)
    writer.close()

    index_path = str(tmp_path / "unsorted.csi")
    with pytest.raises(ValueError, match="coordinate-sorted"):
        VCF(output_path).schema_scan(index_path=index_path)
    assert not os.path.exists(index_path)

@pytest.mark.parametrize("suffix", ["csi", "tbi"])
def test_schema_scan_parallel_workers_builds_index(tmp_path, suffix):
    path = VCF_PATH
    expected = VCF(path).schema_scan()
    index_path = str(tmp_path / ("parallel." + suffix))

    parallel = VCF(path).schema_scan(index_path=index_path, workers=2)

    assert parallel["records"] == expected["records"]
    assert parallel["samples"] == expected["samples"]
    assert parallel["alleles_per_record"] == expected["alleles_per_record"]
    assert parallel["allele_counts"] == expected["allele_counts"]
    assert parallel["ploidy"] == expected["ploidy"]
    assert parallel["fields"] == expected["fields"]
    assert parallel["index"] == {
        "available": True,
        "built": True,
        "path": index_path,
    }
    indexed_vcf = VCF(path)
    indexed_vcf.set_index(index_path)
    assert indexed_vcf.num_records == expected["records"]
    reused = VCF(path).schema_scan(index_path=index_path, workers=2)
    assert reused["index"] == {
        "available": True,
        "built": False,
        "path": index_path,
    }

@pytest.mark.parametrize("cpu_budget,explicit_workers,expected_workers", [
    (1, None, 1),
    (2, None, 1),
    (3, None, 1),
    (8, None, 4),
    (12, None, 6),
    (12, 2, 2),
    (12, 1, 1),
])
def test_schema_scan_auto_concurrency(
        tmp_path, monkeypatch, cpu_budget, explicit_workers, expected_workers):
    import concurrent.futures
    import shutil
    import cyvcf2.cyvcf2 as cyvcf2_module
    from cyvcf2 import _schema_scan as resources

    path = str(tmp_path / "unindexed.vcf.gz")
    shutil.copyfile(VCF_PATH, path)
    monkeypatch.setattr(resources, "_available_cpu_count", lambda: cpu_budget)
    original_ranges = cyvcf2_module._schema_bgzf_ranges
    original_worker = cyvcf2_module._schema_scan_worker
    selected_workers = []
    selected_threads = []

    def recording_ranges(path, workers):
        selected_workers.append(workers)
        return original_ranges(path, workers)

    def recording_worker(args):
        selected_threads.append(args[4])
        return original_worker(args)

    class InProcessPool(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, max_workers=None, mp_context=None):
            super().__init__(max_workers)

    monkeypatch.setattr(cyvcf2_module, "_schema_bgzf_ranges", recording_ranges)
    monkeypatch.setattr(cyvcf2_module, "_schema_scan_worker", recording_worker)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", InProcessPool)
    expected = VCF(path).schema_scan()
    kwargs = {} if explicit_workers is None else {"workers": explicit_workers}
    assert VCF(path).schema_scan(threads=0, **kwargs) == expected
    assert selected_workers == ([expected_workers] if expected_workers > 1 else [])
    if selected_threads:
        assert set(selected_threads) == {1}


@pytest.mark.parametrize("threads", [-1, 1.5, "0"])
def test_schema_scan_invalid_threads(threads):
    with pytest.raises(ValueError, match="non-negative integer"):
        VCF(VCF_PATH).schema_scan(threads=threads)

def test_schema_scan_auto_process_workers(tmp_path, monkeypatch):
    from cyvcf2 import _schema_scan as resources

    monkeypatch.setattr(resources, "_available_cpu_count", lambda: 4)
    bcf_path = _write_multiblock_bcf(tmp_path)
    for path in (bcf_path, str(tmp_path / "multiblock.vcf")):
        assert VCF(path).schema_scan(threads=0) == VCF(path).schema_scan()


def _write_multiblock_bcf(
        tmp_path, n_records=4000, n_samples=40, contig_length=None, start_position=1):
    vcf_path = tmp_path / "multiblock.vcf"
    samples = ["s%d" % i for i in range(n_samples)]
    lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1%s>" % (
            ",length=%d" % contig_length if contig_length is not None else ""),
        "##contig=<ID=2%s>" % (
            ",length=%d" % contig_length if contig_length is not None else ""),
        '##FILTER=<ID=LowQ,Description="low quality">',
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="depth">',
        '##INFO=<ID=AF,Number=A,Type=Float,Description="frequency">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">',
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="depths">',
        "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
                   "INFO", "FORMAT"] + samples),
    ]
    for i in range(n_records):
        chrom = "1" if i < n_records // 2 else "2"
        alts = ",".join(["T", "G", "CA"][:1 + i % 3])
        n_alts = 1 + i % 3
        genotypes = [
            "%d/%d:%s" % (s % (n_alts + 1), (s + i) % (n_alts + 1),
                          ",".join(str(s + i + k) for k in range(n_alts + 1)))
            for s in range(n_samples)
        ]
        lines.append("\t".join([
            chrom, str(10 * i + start_position), "rs%d" % i, "A", alts, str(i % 99),
            "LowQ" if i % 7 == 0 else "PASS",
            "DP=%d;AF=%s" % (i, ",".join(["0.5"] * n_alts)), "GT:AD",
        ] + genotypes))
    vcf_path.write_text("\n".join(lines) + "\n")
    source = VCF(str(vcf_path))
    bcf_path = str(tmp_path / "multiblock.bcf")
    writer = Writer(bcf_path, source, mode="wb")
    for record in source:
        writer.write_record(record)
    writer.close()
    return bcf_path

@pytest.mark.parametrize("is_bcf", [False, True])
@pytest.mark.parametrize("contig_length,vcf_depth,bcf_depth", [
    (None, 8, 6),
    (100000, 6, 1),
    ((1 << 29) - 256, 6, 5),
    ((1 << 29) - 255, 6, 6),
    (1 << 32, 7, 7),
])
def test_schema_scan_csi_depth_matches_htslib(
        tmp_path, is_bcf, contig_length, vcf_depth, bcf_depth):
    import ctypes
    import gzip
    import struct
    import cyvcf2.cyvcf2 as cyvcf2_module

    start_position = (1 << 29) + 1 if contig_length == 1 << 32 else 1
    bcf_path = _write_multiblock_bcf(
        tmp_path, contig_length=contig_length, start_position=start_position)
    if is_bcf:
        path = bcf_path
    else:
        path = str(tmp_path / "multiblock.vcf.gz")
        with VCF(str(tmp_path / "multiblock.vcf")) as source:
            writer = Writer(path, source, mode="wz")
            for record in source:
                writer.write_record(record)
            writer.close()

    serial_index = str(tmp_path / "serial.csi")
    parallel_index = str(tmp_path / "parallel.csi")
    reference_index = str(tmp_path / "reference.csi")
    serial = VCF(path).schema_scan(index_path=serial_index)
    parallel = VCF(path).schema_scan(index_path=parallel_index, workers=4, threads=1)
    serial.pop("index")
    parallel.pop("index")
    assert serial == parallel

    native_build = ctypes.CDLL(cyvcf2_module.__file__).bcf_index_build3
    native_build.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                             ctypes.c_int, ctypes.c_int]
    native_build.restype = ctypes.c_int
    assert native_build(os.fsencode(path), os.fsencode(reference_index), 14, 0) == 0
    expected_bytes = (tmp_path / "reference.csi").read_bytes()
    assert (tmp_path / "serial.csi").read_bytes() == expected_bytes
    assert (tmp_path / "parallel.csi").read_bytes() == expected_bytes
    assert struct.unpack("<4sii", gzip.decompress(expected_bytes)[:12]) == (
        b"CSI\x01", 14, bcf_depth if is_bcf else vcf_depth)
    with VCF(path) as indexed:
        indexed.set_index(parallel_index)
        assert len(list(indexed("2:%d-%d" % (
            start_position + 29999, start_position + 34999)))) == 500


def test_schema_scan_tbi_matches_htslib(tmp_path):
    import ctypes
    import cyvcf2.cyvcf2 as cyvcf2_module

    reference_index = str(tmp_path / "reference.tbi")
    native_build = ctypes.CDLL(cyvcf2_module.__file__).bcf_index_build3
    native_build.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                             ctypes.c_int, ctypes.c_int]
    native_build.restype = ctypes.c_int
    assert native_build(os.fsencode(VCF_PATH), os.fsencode(reference_index), 0, 0) == 0
    expected_bytes = (tmp_path / "reference.tbi").read_bytes()
    for workers in (1, 4):
        index_path = tmp_path / ("scan-%d.tbi" % workers)
        VCF(VCF_PATH).schema_scan(index_path=str(index_path), workers=workers)
        assert index_path.read_bytes() == expected_bytes


def test_schema_scan_parallel_bcf_matches_serial(tmp_path):
    path = _write_multiblock_bcf(tmp_path)
    serial_index = str(tmp_path / "serial.csi")
    serial = VCF(path).schema_scan(index_path=serial_index)
    serial.pop("index")
    assert serial["records"] == 4000

    for workers in (2, 5, 16):
        index_path = str(tmp_path / ("parallel-%d.csi" % workers))
        parallel = VCF(path).schema_scan(index_path=index_path, workers=workers)
        assert parallel.pop("index") == {
            "available": True,
            "built": True,
            "path": index_path,
        }
        assert parallel == serial
        with open(index_path, "rb") as a, open(serial_index, "rb") as b:
            assert a.read() == b.read()

    indexed = VCF(path)
    indexed.set_index(serial_index)
    assert len(list(indexed("1:1-100"))) == 10
    assert len(list(indexed("2:20001-20101"))) == 11
    reused = VCF(path).schema_scan(index_path=serial_index, workers=4)
    assert reused["index"]["built"] is False

def test_schema_scan_parallel_bcf_rescans_unverified_starts(tmp_path, monkeypatch):
    import concurrent.futures
    import cyvcf2.cyvcf2 as cyvcf2_module

    path = _write_multiblock_bcf(tmp_path)
    serial = VCF(path).schema_scan()
    original_worker = cyvcf2_module._schema_scan_worker
    exact_rescans = []

    def corrupting_worker(args):
        exact_start = args[7]
        if exact_start:
            exact_rescans.append(args[1])
        summary = original_worker(args)
        if not exact_start and args[1] >= 0:
            summary["_actual_start"] = None
        return summary

    class InProcessPool(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, max_workers=None, mp_context=None):
            super().__init__(max_workers)

    monkeypatch.setattr(cyvcf2_module, "_schema_scan_worker", corrupting_worker)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", InProcessPool)
    parallel = VCF(path).schema_scan(workers=4)

    assert exact_rescans
    parallel.pop("index")
    serial.pop("index")
    assert parallel == serial

def test_schema_scan_parallel_raw_bcf(tmp_path):
    import gzip

    bgzf_path = _write_multiblock_bcf(tmp_path)
    raw_path = str(tmp_path / "raw.bcf")
    with open(bgzf_path, "rb") as source, open(raw_path, "wb") as target:
        target.write(gzip.decompress(source.read()))
    serial = VCF(raw_path).schema_scan()
    assert serial["records"] == 4000

    for workers in (2, 5, 16):
        assert VCF(raw_path).schema_scan(workers=workers) == serial

    index_path = str(tmp_path / "raw.csi")
    with pytest.raises(ValueError, match="BGZF"):
        VCF(raw_path).schema_scan(index_path=index_path)
    with pytest.raises(ValueError, match="BGZF"):
        VCF(raw_path).schema_scan(index_path=index_path, workers=2)
    assert not os.path.exists(index_path)

def test_schema_scan_parallel_plain_vcf(tmp_path):
    _write_multiblock_bcf(tmp_path)
    path = str(tmp_path / "multiblock.vcf")
    serial = VCF(path).schema_scan()
    assert serial["records"] == 4000

    for workers in (2, 5, 16, 64):
        assert VCF(path).schema_scan(workers=workers) == serial

    index_path = str(tmp_path / "plain.csi")
    with pytest.raises(ValueError, match="BGZF"):
        VCF(path).schema_scan(index_path=index_path, workers=2)
    assert not os.path.exists(index_path)

@pytest.mark.parametrize("plain", [False, True])
def test_schema_scan_parallel_vcf_handoffs(tmp_path, monkeypatch, plain):
    import concurrent.futures
    import cyvcf2.cyvcf2 as cyvcf2_module

    if plain:
        _write_multiblock_bcf(tmp_path)
        path = str(tmp_path / "multiblock.vcf")
    else:
        import shutil
        path = str(tmp_path / "unindexed.vcf.gz")
        shutil.copyfile(VCF_PATH, path)
    serial = VCF(path).schema_scan()
    original_worker = cyvcf2_module._schema_scan_worker
    exact_rescans = []
    corrupt = []

    def recording_worker(args):
        exact_start = args[7]
        if exact_start:
            exact_rescans.append(args[1])
        summary = original_worker(args)
        if corrupt and not exact_start and args[1] >= 0:
            summary["_actual_start"] = None
        return summary

    class InProcessPool(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, max_workers=None, mp_context=None):
            super().__init__(max_workers)

    monkeypatch.setattr(cyvcf2_module, "_schema_scan_worker", recording_worker)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", InProcessPool)
    assert VCF(path).schema_scan(workers=4) == serial
    assert not exact_rescans

    corrupt.append(True)
    assert VCF(path).schema_scan(workers=4) == serial
    assert exact_rescans

@pytest.mark.parametrize("suffix", ["tbi", "csi", "bcf.csi"])
def test_schema_scan_existing_index_splits(tmp_path, monkeypatch, suffix):
    import cyvcf2.cyvcf2 as native
    bcf_path = _write_multiblock_bcf(tmp_path)
    if suffix == "bcf.csi":
        path = bcf_path
    else:
        path = str(tmp_path / "indexed.vcf.gz")
        with VCF(str(tmp_path / "multiblock.vcf")) as source:
            writer = Writer(path, source, mode="wz")
            for record in source:
                writer.write_record(record)
            writer.close()
    index_path = path + (".tbi" if suffix == "tbi" else ".csi")
    expected = VCF(path).schema_scan(index_path=index_path)
    original_bytes = Path(index_path).read_bytes()
    planned = []
    original = native._schema_index_ranges

    def indexed_ranges(*args):
        result = original(*args)
        planned.append(result)
        return result

    def forbidden(*args):
        raise AssertionError("existing index should avoid BGZF discovery")

    monkeypatch.setattr(native, "_schema_index_ranges", indexed_ranges)
    monkeypatch.setattr(native, "_schema_bgzf_ranges", forbidden)
    for destination in (None, index_path):
        parallel = VCF(path).schema_scan(index_path=destination, workers=4)
        parallel.pop("index")
        assert parallel == {key: value for key, value in expected.items() if key != "index"}
    assert planned and any(len(ranges) > 1 for ranges, _ in planned)
    assert Path(index_path).read_bytes() == original_bytes


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("explicit", [False, True])
def test_schema_scan_corrupted_index_is_not_overwritten(tmp_path, workers, explicit):
    import shutil
    path = str(tmp_path / "input.vcf.gz")
    shutil.copyfile(VCF_PATH, path)
    index_path = path + ".csi"
    Path(index_path).write_bytes(b"not an index")
    with pytest.raises(OSError, match="existing index"):
        VCF(path).schema_scan(
            index_path=index_path if explicit else None, workers=workers)
    assert Path(index_path).read_bytes() == b"not an index"


@pytest.mark.parametrize("workers", [1, 4])
def test_schema_scan_index_created_concurrently(tmp_path, monkeypatch, workers):
    index_path = tmp_path / "raced.csi"
    original_link = os.link

    def racing_link(source, destination):
        index_path.write_bytes(b"other writer")
        return original_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        VCF(VCF_PATH).schema_scan(index_path=str(index_path), workers=workers)
    assert index_path.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_schema_scan_indexed_long_overlapping_records(tmp_path, monkeypatch):
    import cyvcf2.cyvcf2 as native
    text_path = tmp_path / "long.vcf"
    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=1000000>\n"
        "##contig=<ID=2,length=1000000>\n"
        '##INFO=<ID=END,Number=1,Type=Integer,Description="end">\n'
        '##INFO=<ID=BLOB,Number=1,Type=String,Description="long value">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    lines = [
        "%s\t%d\t.\tA\t<DEL>\t.\tPASS\tEND=999999;BLOB=%s\n"
        % (chrom, i * 10000 + 1, "A" * 180000)
        for chrom in ("1", "2") for i in range(12)
    ]
    text_path.write_text(header + "".join(lines))
    path = str(tmp_path / "long.vcf.gz")
    with VCF(str(text_path)) as source:
        writer = Writer(path, source, mode="wz")
        for record in source:
            writer.write_record(record)
        writer.close()
    expected = VCF(path).schema_scan()
    index_path = path + ".csi"
    VCF(path).schema_scan(index_path=index_path)
    original = native._schema_index_ranges
    plans = []

    def recording_plan(*args):
        result = original(*args)
        plans.append(result)
        return result

    monkeypatch.setattr(native, "_schema_index_ranges", recording_plan)
    result = VCF(path).schema_scan(workers=8)
    result.pop("index")
    expected.pop("index")
    assert result == expected
    assert any(len(ranges) > 1 for ranges, _ in plans)


@pytest.mark.parametrize("corruption", ["boundary", "count"])
def test_schema_scan_index_split_validation(tmp_path, monkeypatch, corruption):
    import cyvcf2.cyvcf2 as native
    bcf_path = _write_multiblock_bcf(tmp_path)
    index_path = bcf_path + ".csi"
    VCF(bcf_path).schema_scan(index_path=index_path)
    original = native._schema_index_ranges

    def corrupted_plan(*args):
        ranges, count = original(*args)
        assert len(ranges) > 1
        if corruption == "count":
            return ranges, count + 1
        offset = ranges[1][0] + 1
        ranges[0] = (ranges[0][0], offset, ranges[0][2])
        ranges[1] = (offset, ranges[1][1], offset)
        return ranges, count

    monkeypatch.setattr(native, "_schema_index_ranges", corrupted_plan)
    original_bytes = Path(index_path).read_bytes()
    with pytest.raises((OSError, ValueError)):
        VCF(bcf_path).schema_scan(workers=4)
    assert Path(index_path).read_bytes() == original_bytes


def test_schema_scan_includes_unused_header_fields(tmp_path):
    path = tmp_path / "empty.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1>\n"
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="depth">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n"
    )

    summary = VCF(str(path)).schema_scan()
    assert summary["records"] == 0
    assert summary["fields"]["INFO"]["DP"]["records"] == 0
    assert summary["fields"]["INFO"]["DP"]["cardinality"] == {
        "min": None,
        "max": None,
    }
    assert summary["fields"]["FORMAT"]["GT"]["records"] == 0

def test_reader_context_manager():
    with VCF(VCF_PATH) as vcf:
        pass
    with pytest.raises(Exception, match="attempt to iterate over closed"):
        next(vcf)
