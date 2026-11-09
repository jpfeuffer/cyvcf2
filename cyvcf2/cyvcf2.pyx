#cython: profile=False
#cython:language_level=3

#cython: embedsignature=True
from __future__ import print_function
import os
import sys
from collections import defaultdict
import atexit
import tempfile
import numpy as np
from array import array
import math
import ctypes
import warnings
import struct
try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport

from libc cimport stdlib
cimport numpy as np
np.seterr(invalid='ignore')
np.import_array()
import locale

ENC = locale.getpreferredencoding()

from cython cimport view

from cpython.version cimport PY_MAJOR_VERSION
from cpython.bytes cimport (
    PyBytes_FromStringAndSize,
    PyBytes_AS_STRING,
)

def set_htslib_log_level(int level):
    """
    Set htslib log level. 

    Parameters:
    level (int): The desired log level. Per htslib's documentation (https://www.htslib.org/doc/samtools.html), the following values are supported:
        - 0 or 1: Suppress some error messages.
        - 2: Reduce warning messages.
        - 3: Default level (HTS_LOG_WARNING).
        - >3: Increase verbosity with additional warnings and logging messages.
    """
    hts_set_log_level(level)


def _init_logging_from_env():
    """
    If a valid CYVCF2_HTSLIB_LOG_LEVEL environment variable exists, use it to set htslib log level. 
    Otherwise use the htslib default: 3 (HTS_LOG_WARNING).
    """
    HTSLIB_DEFAULT = 3

    s = stdlib.getenv("CYVCF2_HTSLIB_LOG_LEVEL")
    if s is NULL:
        set_htslib_log_level(HTSLIB_DEFAULT)
        return

    py_s = s.decode()
    try:
        level = int(py_s)
        if level < 0:
            warnings.warn(f"CYVCF2_HTSLIB_LOG_LEVEL={py_s!r} is not a non-negative integer; using htslib default")
            level = HTSLIB_DEFAULT
    except (ValueError, TypeError):
        warnings.warn(f"CYVCF2_HTSLIB_LOG_LEVEL={py_s!r} is not a valid integer; using htslib default")
        level = HTSLIB_DEFAULT

    set_htslib_log_level(level)

_init_logging_from_env()


# overcome lack of __file__ in cython
import inspect
if not hasattr(sys.modules[__name__], '__file__'):
    __file__ = inspect.getfile(inspect.currentframe())


def par_relatedness(vcf_path, samples, ncpus, sites, min_depth=5, each=1):
    from multiprocessing import Pool
    p = Pool(ncpus)
    cdef np.ndarray aibs, an, ahets

    for i, fname in enumerate(p.imap_unordered(_par_relatedness, [
        (vcf_path, samples, min_depth, i, ncpus, each, sites) for i in range(ncpus)])):

        arrays = np.load(fname)
        os.unlink(fname)
        if i == 0:
            aibs, an, ahets = arrays['ibs'], arrays['n'], arrays['hets']
        else:
            aibs += arrays['ibs']
            an += arrays['n']
            ahets += arrays['hets']

    return VCF(vcf_path, samples=samples)._relatedness_finish(aibs, an, ahets)


def par_het(vcf_path, samples, ncpus, qsites, min_depth=8, percentiles=(10, 90),
                  int each=1, int offset=0):
    from multiprocessing import Pool
    p = Pool(ncpus)

    any_counts, sum_counts, het_counts = 0, 0, 0
    all_gt_types, mean_depths, sites = [], [], []
    maf_lists = defaultdict(list)
    for ret in p.imap_unordered(_par_het, [(vcf_path, samples, qsites, min_depth, i, ncpus, each) for i in range(ncpus)]):
        (mean_depths_, maf_lists_, het_counts_, sum_counts_,
                all_gt_types_, sites_, any_counts_) = ret
        mean_depths.extend(mean_depths_)
        any_counts += any_counts_
        sites.extend(sites_)
        all_gt_types.extend(all_gt_types_)
        sum_counts += sum_counts_ # an array
        het_counts += het_counts_ # an array
        for k in maf_lists_:
            li = maf_lists_[k]
            maf_lists[k].extend(li)
    mean_depths = np.array(mean_depths, dtype=np.int32).T
    vcf = VCF(vcf_path, samples=samples, gts012=True)
    return vcf._finish_het(mean_depths, maf_lists,
                           percentiles,
                           het_counts, sum_counts, all_gt_types,
                           sites, any_counts)


def _par_het(args):
    vcf_path, samples, sites, min_depth, offset, ncpus, each = args
    each *= ncpus
    vcf = VCF(vcf_path, samples=samples, gts012=True)
    return vcf.het_check(min_depth=min_depth, sites=sites, each=each, offset=offset, _finish=False)


def _par_relatedness(args):
    vcf_path, samples, min_depth, offset, ncpus, each, sites = args
    vcf = VCF(vcf_path, samples=samples, gts012=True)
    each = each * ncpus
    vibs, vn, vhet = vcf._site_relatedness(min_depth=min_depth, offset=offset, each=each, sites=sites)
    # to get around limits of multiprocessing size of transmitted data, we save
    # the arrays to disk and return the file
    fname = tempfile.mktemp(suffix=".npz")
    atexit.register(os.unlink, fname)
    np.savez_compressed(fname, ibs=np.asarray(vibs), hets=np.asarray(vhet), n=np.asarray(vn))
    return fname


cdef unicode xstr(s):
    if type(s) is unicode:
        # fast path for most common case(s)
        return <unicode>s
    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        # only accept byte strings in Python 2.x, not in Py3
        return (<bytes>s).decode('ascii')
    elif isinstance(s, unicode):
        # an evil cast to <unicode> might work here in some(!) cases,
        # depending on what the further processing does.  to be safe,
        # we can always create a copy instead
        return unicode(s)
    else:
        raise TypeError(...)


def r_(int32_t[::view.contiguous] a_gts, int32_t[::view.contiguous] b_gts, float f, int32_t n_samples):
    return r_unphased(&a_gts[0], &b_gts[0], f, n_samples)


cdef char* pystring_to_cstring(str py_string):
    # Convert Python string to byte string using UTF-8 encoding
    cdef bytes byte_string = py_string.encode('utf-8')  # Convert Python string to bytes
    # Convert Python bytes to a C-compatible pointer
    cdef const char* byte_string_ptr = byte_string

    # Allocate memory for C string with size equal to byte string length plus null terminator
    cdef size_t length = len(byte_string) + 1  # Plus 1 for the null terminator
    cdef char* c_string = <char*>stdlib.malloc(length)
    # return NULL if memory allocation fails
    if c_string == NULL:
        return c_string
    # Copy byte string to C string using memcpy and add null terminator
    memcpy(c_string, byte_string_ptr, length - 1)  # Use memcpy to copy bytes to C string
    c_string[length - 1] = b'\0'  # Add null terminator to the end of the C string
    return c_string


cdef set_constants(VCF v):
    v.HOM_REF = 0
    v.HET = 1
    if v.gts012:
        v.HOM_ALT = 2
        v.UNKNOWN = 3
    else:
        v.UNKNOWN = 2
        v.HOM_ALT = 3


cdef struct SchemaFieldStats:
    uint64_t records
    uint64_t total_values
    int min_cardinality
    int max_cardinality
    bint has_numeric
    double min_value
    double max_value


cdef struct SchemaIndexEntry:
    int32_t rid
    int64_t beg
    int64_t end
    uint64_t offset


cdef bytes _schema_vcf_index_metadata(list contig_names):
    cdef bytes names_blob
    names_blob = b"".join(name + b"\0" for name in contig_names)
    return struct.pack(
        "<7I",
        tbx_conf_vcf.preset,
        tbx_conf_vcf.sc,
        tbx_conf_vcf.bc,
        tbx_conf_vcf.ec,
        tbx_conf_vcf.meta_char,
        tbx_conf_vcf.line_skip,
        len(names_blob),
    ) + names_blob


cdef enum:
    SCHEMA_BCF_INVALID = -1
    SCHEMA_BCF_TRUNCATED = -2
    SCHEMA_BCF_MASK_INFO = 1
    SCHEMA_BCF_MASK_FLT = 2
    SCHEMA_BCF_MASK_FMT = 4

cdef int64_t SCHEMA_BCF_MAX_BLOCK = 1 << 30
cdef int SCHEMA_BCF_REQUIRED_RECORDS = 4
cdef int64_t SCHEMA_BCF_INITIAL_SEARCH = 1 << 20
cdef int64_t SCHEMA_BCF_MAX_SEARCH = 64 << 20


cdef inline int64_t _schema_raw_offset(int64_t offset) nogil:
    # Uncompressed BGZF streams address bytes as (64 KiB read << 16) | offset;
    # canonicalize so equal byte positions compare equal.
    return ((offset >> 16) + (offset & 0xFFFF)) << 16


cdef inline uint32_t _schema_u32(const uint8_t *data) nogil:
    return (<uint32_t>data[0] | (<uint32_t>data[1] << 8) |
            (<uint32_t>data[2] << 16) | (<uint32_t>data[3] << 24))


cdef inline int _schema_bcf_type_size(int value_type) nogil:
    if value_type == 0:
        return 0
    if value_type == 1 or value_type == 7:
        return 1
    if value_type == 2:
        return 2
    if value_type == 3 or value_type == 5:
        return 4
    return -1


cdef inline int _schema_bcf_int(
        const uint8_t *data, int64_t length, int64_t *cursor,
        int64_t *value) nogil:
    cdef int descriptor
    cdef int value_type
    cdef int size
    if cursor[0] >= length:
        return SCHEMA_BCF_TRUNCATED
    descriptor = data[cursor[0]]
    value_type = descriptor & 15
    if descriptor >> 4 != 1 or value_type < 1 or value_type > 3:
        return SCHEMA_BCF_INVALID
    size = _schema_bcf_type_size(value_type)
    if cursor[0] + 1 + size > length:
        return SCHEMA_BCF_TRUNCATED
    if value_type == 1:
        value[0] = <int8_t>data[cursor[0] + 1]
    elif value_type == 2:
        value[0] = <int16_t>(
            <uint16_t>data[cursor[0] + 1] | (<uint16_t>data[cursor[0] + 2] << 8))
    else:
        value[0] = <int32_t>_schema_u32(data + cursor[0] + 1)
    cursor[0] += 1 + size
    return 0


cdef inline int _schema_bcf_typed(
        const uint8_t *data, int64_t length, int64_t *cursor,
        int *value_type, int64_t *count) nogil:
    cdef int descriptor
    cdef int status
    if cursor[0] >= length:
        return SCHEMA_BCF_TRUNCATED
    descriptor = data[cursor[0]]
    value_type[0] = descriptor & 15
    count[0] = descriptor >> 4
    if _schema_bcf_type_size(value_type[0]) < 0:
        return SCHEMA_BCF_INVALID
    cursor[0] += 1
    if count[0] == 15:
        status = _schema_bcf_int(data, length, cursor, count)
        if status != 0:
            return status
        if count[0] < 15:
            return SCHEMA_BCF_INVALID
    if value_type[0] == 0 and count[0] != 0:
        return SCHEMA_BCF_INVALID
    return 0


cdef inline int _schema_bcf_skip_typed(
        const uint8_t *data, int64_t length, int64_t *cursor,
        int64_t multiplier, int *value_type) nogil:
    cdef int64_t count
    cdef int status = _schema_bcf_typed(
        data, length, cursor, value_type, &count)
    if status != 0:
        return status
    cursor[0] += count * multiplier * _schema_bcf_type_size(value_type[0])
    if cursor[0] > length:
        return SCHEMA_BCF_INVALID
    return 0


cdef int64_t _schema_bcf_check_record(
        const uint8_t *data, int64_t length, int64_t start,
        int n_contigs, int n_samples, int n_ids,
        const uint8_t *id_mask) nogil:
    """Validate the complete structure of one BCF record at ``start``.

    Returns the record length, SCHEMA_BCF_INVALID, or SCHEMA_BCF_TRUNCATED.
    """
    cdef int64_t shared_length
    cdef int64_t indiv_length
    cdef int64_t cursor
    cdef int64_t shared_end
    cdef int64_t indiv_end
    cdef int64_t key
    cdef int64_t count
    cdef int64_t i
    cdef int64_t j
    cdef int32_t rid
    cdef int32_t pos
    cdef int32_t rlen
    cdef uint32_t packed
    cdef int n_info
    cdef int n_allele
    cdef int n_fmt
    cdef int value_type
    cdef int status
    cdef int size

    if start + 32 > length:
        return SCHEMA_BCF_TRUNCATED
    shared_length = _schema_u32(data + start)
    indiv_length = _schema_u32(data + start + 4)
    if (shared_length < 24 or shared_length > SCHEMA_BCF_MAX_BLOCK or
            indiv_length > SCHEMA_BCF_MAX_BLOCK):
        return SCHEMA_BCF_INVALID
    cursor = start + 8
    rid = <int32_t>_schema_u32(data + cursor)
    pos = <int32_t>_schema_u32(data + cursor + 4)
    rlen = <int32_t>_schema_u32(data + cursor + 8)
    if rid < 0 or rid >= n_contigs or pos < -1 or rlen < 0:
        return SCHEMA_BCF_INVALID
    packed = _schema_u32(data + cursor + 16)
    n_info = packed & 0xffff
    n_allele = packed >> 16
    packed = _schema_u32(data + cursor + 20)
    n_fmt = packed >> 24
    if <int>(packed & 0xffffff) != n_samples:
        return SCHEMA_BCF_INVALID
    if (n_fmt == 0) != (indiv_length == 0):
        return SCHEMA_BCF_INVALID
    shared_end = start + 8 + shared_length
    indiv_end = shared_end + indiv_length
    if indiv_end > length:
        return SCHEMA_BCF_TRUNCATED
    cursor += 24

    status = _schema_bcf_skip_typed(data, shared_end, &cursor, 1, &value_type)
    if status != 0 or (value_type != 7 and value_type != 0):
        return SCHEMA_BCF_INVALID
    for i in range(n_allele):
        status = _schema_bcf_skip_typed(
            data, shared_end, &cursor, 1, &value_type)
        if status != 0 or (value_type != 7 and value_type != 0):
            return SCHEMA_BCF_INVALID
    status = _schema_bcf_typed(data, shared_end, &cursor, &value_type, &count)
    if status != 0 or value_type == 5 or value_type == 7:
        return SCHEMA_BCF_INVALID
    size = _schema_bcf_type_size(value_type)
    if cursor + count * size > shared_end:
        return SCHEMA_BCF_INVALID
    for j in range(count):
        if size == 1:
            key = <int8_t>data[cursor]
        elif size == 2:
            key = <int16_t>(<uint16_t>data[cursor] | (<uint16_t>data[cursor + 1] << 8))
        else:
            key = <int32_t>_schema_u32(data + cursor)
        if key < 0 or key >= n_ids or not (id_mask[key] & SCHEMA_BCF_MASK_FLT):
            return SCHEMA_BCF_INVALID
        cursor += size
    for i in range(n_info):
        if _schema_bcf_int(data, shared_end, &cursor, &key) != 0:
            return SCHEMA_BCF_INVALID
        if key < 0 or key >= n_ids or not (id_mask[key] & SCHEMA_BCF_MASK_INFO):
            return SCHEMA_BCF_INVALID
        if _schema_bcf_skip_typed(
                data, shared_end, &cursor, 1, &value_type) != 0:
            return SCHEMA_BCF_INVALID
    if cursor != shared_end:
        return SCHEMA_BCF_INVALID

    for i in range(n_fmt):
        if _schema_bcf_int(data, indiv_end, &cursor, &key) != 0:
            return SCHEMA_BCF_INVALID
        if key < 0 or key >= n_ids or not (id_mask[key] & SCHEMA_BCF_MASK_FMT):
            return SCHEMA_BCF_INVALID
        if _schema_bcf_skip_typed(
                data, indiv_end, &cursor, n_samples, &value_type) != 0:
            return SCHEMA_BCF_INVALID
    if cursor != indiv_end:
        return SCHEMA_BCF_INVALID
    return indiv_end - start


cdef int64_t _schema_bcf_find_record_start(
        const uint8_t *data, int64_t length, bint at_eof,
        int n_contigs, int n_samples, int n_ids,
        const uint8_t *id_mask) nogil:
    """Return the first offset that begins a chain of valid BCF records.

    Returns SCHEMA_BCF_INVALID if no candidate exists in ``data`` and
    SCHEMA_BCF_TRUNCATED if more decompressed data is needed to decide.
    """
    cdef int64_t candidate
    cdef int64_t cursor
    cdef int64_t record_length
    cdef int records

    for candidate in range(length):
        cursor = candidate
        records = 0
        while records < SCHEMA_BCF_REQUIRED_RECORDS:
            if at_eof and cursor == length:
                break
            record_length = _schema_bcf_check_record(
                data, length, cursor, n_contigs, n_samples, n_ids, id_mask)
            if record_length == SCHEMA_BCF_TRUNCATED and not at_eof:
                return SCHEMA_BCF_TRUNCATED
            if record_length < 0:
                break
            cursor += record_length
            records += 1
        if records == SCHEMA_BCF_REQUIRED_RECORDS or (
                records > 0 and at_eof and cursor == length):
            return candidate
    return SCHEMA_BCF_INVALID


cdef int64_t _schema_bcf_locate_start(
        BGZF *fp, bcf_hdr_t *hdr, int64_t block_start) except -2:
    """Find the virtual offset of the first BCF record starting in a block.

    Returns -1 when no record start could be identified within the search
    budget; the caller then relies on the previous chunk's verified handoff.
    """
    cdef int n_ids = hdr.n[BCF_DT_ID]
    cdef int n_contigs = hdr.n[BCF_DT_CTG]
    cdef int n_samples = bcf_hdr_nsamples(hdr)
    cdef uint8_t *id_mask = NULL
    cdef uint8_t *data = NULL
    cdef uint8_t *resized = NULL
    cdef int64_t capacity = 0
    cdef int64_t length = 0
    cdef int64_t candidate = SCHEMA_BCF_TRUNCATED
    cdef int64_t target
    cdef ssize_t got
    cdef bint at_eof = False
    cdef int i

    id_mask = <uint8_t *>stdlib.calloc(<size_t>max(n_ids, 1), 1)
    if id_mask == NULL:
        raise MemoryError()
    try:
        for i in range(n_ids):
            if bcf_hdr_idinfo_exists(hdr, BCF_HL_INFO, i):
                id_mask[i] |= SCHEMA_BCF_MASK_INFO
            if bcf_hdr_idinfo_exists(hdr, BCF_HL_FLT, i):
                id_mask[i] |= SCHEMA_BCF_MASK_FLT
            if bcf_hdr_idinfo_exists(hdr, BCF_HL_FMT, i):
                id_mask[i] |= SCHEMA_BCF_MASK_FMT
        if bgzf_seek(fp, block_start, 0) < 0:
            raise IOError("cannot seek to BGZF chunk")
        target = SCHEMA_BCF_INITIAL_SEARCH
        while candidate == SCHEMA_BCF_TRUNCATED:
            if target > capacity:
                resized = <uint8_t *>stdlib.realloc(data, <size_t>target)
                if resized == NULL:
                    raise MemoryError()
                data = resized
                capacity = target
            with nogil:
                got = bgzf_read(fp, data + length, <size_t>(target - length))
            if got < 0:
                raise IOError("error reading BCF chunk")
            length += got
            at_eof = length < target
            with nogil:
                candidate = _schema_bcf_find_record_start(
                    data, length, at_eof, n_contigs, n_samples, n_ids, id_mask)
            if candidate == SCHEMA_BCF_TRUNCATED:
                if target >= SCHEMA_BCF_MAX_SEARCH:
                    return -1
                target *= 2
        if candidate < 0:
            return -1
        if bgzf_seek(fp, block_start, 0) < 0:
            raise IOError("cannot seek to BGZF chunk")
        if candidate > 0:
            with nogil:
                got = bgzf_read(fp, data, <size_t>candidate)
            if got != candidate:
                raise IOError("error reading BCF chunk")
        return bgzf_tell(fp)
    finally:
        stdlib.free(id_mask)
        stdlib.free(data)


cdef void _schema_index_settings(
        const bcf_hdr_t *hdr, bint is_bcf, int index_format,
        int *min_shift, int *n_levels) except *:
    cdef int i
    cdef uint64_t max_length = 0
    cdef uint64_t max_position

    min_shift[0] = 14
    n_levels[0] = 5
    if index_format == HTS_FMT_TBI:
        return
    for i in range(hdr.n[BCF_DT_CTG]):
        if hdr.id[BCF_DT_CTG][i].val != NULL:
            max_length = max(max_length, hdr.id[BCF_DT_CTG][i].val.info[0])

    # Match HTSlib's bcf_index/tbx_index policies, including unknown lengths.
    if is_bcf:
        n_levels[0] = 0
        if max_length == 0:
            max_length = (<uint64_t>1 << 31) - 1
    else:
        n_levels[0] = 6
        if max_length == 0:
            n_levels[0] = 8
            return

    if max_length > (<uint64_t>1 << 62) - 256:
        raise ValueError("contig length exceeds the supported CSI coordinate range")
    max_length += 256
    max_position = <uint64_t>1 << (min_shift[0] + 3 * n_levels[0])
    while max_length > max_position:
        if n_levels[0] < 9:
            n_levels[0] += 1
            max_position <<= 3
        else:
            min_shift[0] += 1
            max_position <<= 1
    if min_shift[0] != 14:
        warnings.warn(
            "Adjusted CSI min_shift from 14 to %d due to the longest contig"
            % min_shift[0], RuntimeWarning)


cdef class _SchemaIndexBuilder:
    cdef hts_idx_t *index
    cdef bytes fname
    cdef bytes index_path
    cdef int *rid_map
    cdef int n_contigs
    cdef int next_rid
    cdef list header_contig_names
    cdef list contig_names
    cdef int index_format
    cdef bint is_bcf
    cdef bint finished

    def __cinit__(self, path, index_path):
        cdef htsFile *fp = NULL
        cdef bcf_hdr_t *hdr = NULL
        cdef int n_contigs = 0
        cdef int contig_index = 0
        cdef int min_shift = 14
        cdef int n_levels = 5
        cdef int64_t offset0 = 0
        cdef const char *contig_name = NULL

        self.index = NULL
        self.rid_map = NULL
        self.n_contigs = 0
        self.next_rid = 0
        self.header_contig_names = []
        self.contig_names = []
        self.finished = False
        self.fname = to_bytes(path)
        self.index_path = to_bytes(
            str(index_path) if isinstance(index_path, Path) else index_path)
        if self.fname == self.index_path:
            raise ValueError("index_path must differ from the input path")
        self.index_format = (
            HTS_FMT_TBI if self.index_path.endswith(b".tbi") else HTS_FMT_CSI)

        fp = hts_open(self.fname, b"r")
        if fp == NULL:
            raise IOError("Error opening %s" % path)
        try:
            if not fp.is_bgzf or fp.format.compression == no_compression:
                raise ValueError("index creation requires BGZF-compressed VCF/BCF")
            hdr = bcf_hdr_read(fp)
            if hdr == NULL:
                raise IOError("cannot parse VCF header")
            if self.index_format == HTS_FMT_TBI and fp.format.format != vcf:
                raise ValueError("TBI indexes can only be created for VCF files")
            n_contigs = hdr.n[BCF_DT_CTG]
            self.n_contigs = n_contigs
            self.is_bcf = fp.format.format == bcf
            self.rid_map = <int *>stdlib.malloc(
                <size_t>n_contigs * sizeof(int))
            if self.rid_map == NULL and n_contigs > 0:
                raise MemoryError("unable to allocate contig mapping")
            for contig_index in range(n_contigs):
                self.rid_map[contig_index] = -1
                contig_name = bcf_hdr_int2id(
                    hdr, BCF_DT_CTG, contig_index)
                if contig_name == NULL:
                    raise ValueError("unable to read VCF contig name")
                self.header_contig_names.append(<bytes>contig_name)
            offset0 = bgzf_tell(fp.fp.bgzf)
            _schema_index_settings(
                hdr, self.is_bcf, self.index_format, &min_shift, &n_levels)
            self.index = hts_idx_init(
                n_contigs if self.is_bcf else 0, self.index_format,
                <uint64_t>offset0, min_shift, n_levels)
            if self.index == NULL:
                raise MemoryError("unable to initialize index")
        finally:
            if hdr != NULL:
                bcf_hdr_destroy(hdr)
            if fp != NULL:
                hts_close(fp)

    def add_file(self, path):
        cdef SchemaIndexEntry *entries
        cdef SchemaIndexEntry entry
        cdef Py_ssize_t entry_count
        cdef Py_ssize_t entry_index
        cdef int mapped_rid
        cdef size_t chunk_size = (1 << 20) // sizeof(SchemaIndexEntry) * sizeof(SchemaIndexEntry)
        cdef object source = open(path, "rb")
        cdef bytes payload

        try:
            while True:
                payload = source.read(chunk_size)
                if not payload:
                    break
                if len(payload) % sizeof(SchemaIndexEntry) != 0:
                    raise IOError("invalid chunk index entry file")
                entries = <SchemaIndexEntry *>PyBytes_AS_STRING(payload)
                entry_count = len(payload) // sizeof(SchemaIndexEntry)
                for entry_index in range(entry_count):
                    entry = entries[entry_index]
                    if entry.rid < 0 or entry.rid >= self.n_contigs:
                        raise ValueError("invalid contig ID in index entries")
                    mapped_rid = entry.rid if self.is_bcf else self.rid_map[entry.rid]
                    if mapped_rid < 0:
                        mapped_rid = self.next_rid
                        self.rid_map[entry.rid] = mapped_rid
                        self.next_rid += 1
                        self.contig_names.append(
                            self.header_contig_names[entry.rid])
                    if hts_idx_push(
                            self.index, mapped_rid, entry.beg, entry.end,
                            entry.offset, 1) < 0:
                        raise ValueError(
                            "cannot build index: records must be coordinate-sorted")
        finally:
            source.close()

    def finish(self, int64_t final_offset):
        cdef int ret
        cdef bytes metadata

        if self.finished:
            raise ValueError("index builder is already finished")
        if final_offset < 0 or hts_idx_finish(
                self.index, <uint64_t>final_offset) != 0:
            raise ValueError("unable to finish index; input may not be coordinate-sorted")
        if not self.is_bcf:
            metadata = _schema_vcf_index_metadata(self.contig_names)
            if hts_idx_set_meta(
                    self.index, len(metadata),
                    <uint8_t *>PyBytes_AS_STRING(metadata), 1) < 0:
                raise MemoryError("unable to set tabix index metadata")
        ret = hts_idx_save_as(
            self.index, self.fname, self.index_path, self.index_format)
        if ret < 0:
            raise IOError("unable to save index to %s" % self.index_path)
        self.finished = True

    def __dealloc__(self):
        if self.index != NULL:
            hts_idx_destroy(self.index)
        if self.rid_map != NULL:
            stdlib.free(self.rid_map)


def _schema_bgzf_ranges(path, int workers):
    import bisect
    import struct

    file_size = os.path.getsize(path)
    block_offsets = []
    with open(path, "rb") as source:
        magic = source.read(4)
        if magic[:2] != b"\x1f\x8b":
            # Uncompressed BCF/VCF can be split at any byte; workers locate and
            # verify record starts exactly as for BGZF blocks.
            data_start = 0
            if magic != b"BCF\x02":
                source.seek(0)
                for line in source:
                    if not line.startswith(b"#"):
                        break
                    data_start += len(line)
            boundaries = sorted(set(
                data_start + (file_size - data_start) * worker // workers
                for worker in range(1, workers)))
            boundaries = [offset for offset in boundaries if data_start < offset < file_size]
            starts = [-1] + [offset << 16 for offset in boundaries]
            ends = [offset << 16 for offset in boundaries] + [(1 << 63) - 1]
            # Text workers seek one byte early to tell whether the boundary
            # already starts a line.
            return [
                (start, end, -1 if start < 0 else ((start >> 16) - 1) << 16)
                for start, end in zip(starts, ends)
            ]
        offset = 0
        while offset < file_size:
            source.seek(offset)
            fixed = source.read(12)
            if len(fixed) != 12 or fixed[:2] != b"\x1f\x8b" or not (fixed[3] & 4):
                raise ValueError("parallel schema scanning requires BGZF input")
            extra_length = struct.unpack("<H", fixed[10:12])[0]
            extra = source.read(extra_length)
            if len(extra) != extra_length:
                raise IOError("truncated BGZF extra header")
            block_size = None
            cursor = 0
            while cursor + 4 <= extra_length:
                subfield_length = struct.unpack("<H", extra[cursor + 2:cursor + 4])[0]
                subfield_end = cursor + 4 + subfield_length
                if subfield_end > extra_length:
                    raise IOError("invalid BGZF extra subfield")
                if extra[cursor:cursor + 2] == b"BC" and subfield_length == 2:
                    block_size = struct.unpack("<H", extra[cursor + 4:subfield_end])[0] + 1
                    break
                cursor = subfield_end
            if block_size is None or block_size < 18 or offset + block_size > file_size:
                raise IOError("invalid BGZF block header")
            block_offsets.append(offset)
            offset += block_size

    if not block_offsets:
        raise ValueError("empty BGZF input")
    target_offsets = [
        file_size * worker // workers
        for worker in range(1, workers)
    ]
    boundaries = []
    target_index = 0
    for block_offset in block_offsets:
        while target_index < len(target_offsets) and block_offset >= target_offsets[target_index]:
            if not boundaries or boundaries[-1] != block_offset:
                boundaries.append(block_offset)
            target_index += 1
    boundaries = [offset for offset in boundaries if offset < file_size]
    starts = [-1] + [offset << 16 for offset in boundaries]
    ends = [offset << 16 for offset in boundaries] + [(1 << 63) - 1]
    ranges = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        if start < 0:
            seek_offset = -1
        else:
            block_index = bisect.bisect_left(block_offsets, start >> 16)
            seek_offset = block_offsets[max(0, block_index - 1)] << 16
        ranges.append((start, end, seek_offset))
    return ranges


def _schema_scan_worker(args):
    (path, start, end, seek, threads, collect_index, index_entries_path,
     exact_start) = args
    reader = VCF(path)
    try:
        return reader._schema_scan_chunk(
            start, end, seek, threads, collect_index, index_entries_path,
            exact_start)
    finally:
        reader.close()


def _schema_index_is_available(path, index_path=None, is_bcf=False):
    cdef bytes fname = to_bytes(path)
    cdef bytes idxname
    cdef hts_idx_t *index = NULL
    cdef tbx_t *tbx = NULL
    cdef list candidates

    if index_path is not None:
        idxname = to_bytes(
            str(index_path) if isinstance(index_path, Path) else index_path)
        if fname == idxname:
            raise ValueError("index_path must differ from the input path")
        if not os.path.exists(index_path):
            return False
        if not is_bcf and idxname.endswith((b".tbi", b".csi")):
            tbx = tbx_index_load2(fname, idxname)
            if tbx != NULL:
                tbx_destroy(tbx)
                return True
        else:
            index = hts_idx_load2(fname, idxname)
            if index != NULL:
                hts_idx_destroy(index)
                return True
        raise OSError("unable to read existing index: %s" % index_path)

    candidates = [fname + b".csi"] if is_bcf else [fname + b".csi", fname + b".tbi"]
    for idxname in candidates:
        if not os.path.exists(idxname):
            continue
        if not is_bcf:
            tbx = tbx_index_load2(fname, idxname)
            if tbx != NULL:
                tbx_destroy(tbx)
                return True
        else:
            index = hts_idx_load2(fname, idxname)
            if index != NULL:
                hts_idx_destroy(index)
                return True
    return False


def _merge_schema_scan_summaries(summaries):
    merged = summaries[0]
    merged["records"] = sum(summary["records"] for summary in summaries)
    for summary in summaries[1:]:
        for key, count in summary["alleles_per_record"].items():
            merged["alleles_per_record"][key] = (
                merged["alleles_per_record"].get(key, 0) + count)
        merged["allele_counts"]["missing"] += summary["allele_counts"]["missing"]
        counts = merged["allele_counts"]["by_index"]
        for i, count in enumerate(summary["allele_counts"]["by_index"]):
            if i >= len(counts):
                counts.extend([0] * (i + 1 - len(counts)))
            counts[i] += count
        for ploidy, count in summary["ploidy"].items():
            merged["ploidy"][ploidy] = merged["ploidy"].get(ploidy, 0) + count
        for section in ("INFO", "FORMAT"):
            for name, field in summary["fields"][section].items():
                target = merged["fields"][section][name]
                target["records"] += field["records"]
                target["total_values"] += field["total_values"]
                if field["cardinality"]["min"] is not None:
                    current_min = target["cardinality"]["min"]
                    current_max = target["cardinality"]["max"]
                    target["cardinality"]["min"] = (
                        field["cardinality"]["min"] if current_min is None
                        else min(current_min, field["cardinality"]["min"]))
                    target["cardinality"]["max"] = (
                        field["cardinality"]["max"] if current_max is None
                        else max(current_max, field["cardinality"]["max"]))
                bounds = field["numeric_bounds"]
                if bounds is not None:
                    current = target["numeric_bounds"]
                    if current is None:
                        target["numeric_bounds"] = dict(bounds)
                    else:
                        current["min"] = min(current["min"], bounds["min"])
                        current["max"] = max(current["max"], bounds["max"])
    return merged


cdef class HTSFile:

    cdef htsFile *hts
    cdef bytes fname
    cdef bytes mode
    cdef bint from_path

    cdef _open_htsfile(self, fname, mode):
        """Opens an htsfile for reading or writing.

        Parameters
        ----------
        fname: str
            filename (str or Path), file descriptor (int), or file-like object (has fileno method).
        mode: str
            the mode to pass to hts_open.
        """
        cdef hFILE *hf
        self.mode = to_bytes(mode)
        reading = b"r" in self.mode
        writing = b"w" in self.mode
        if not reading and not writing:
            raise IOError("No 'r' or 'w' in mode %s" % str(self.mode))

        self.from_path = isinstance(fname, (basestring, Path))
        if self.from_path:
            self.fname = to_bytes(str(fname))
            if self.fname == b"-":
                self.fname = to_bytes(b"/dev/stdin") if reading else to_bytes(b"/dev/stdout")

            self.hts = hts_open(self.fname, self.mode)
        # from a file descriptor
        elif isinstance(fname, int):
            hf = hdopen(int(fname), self.mode)
            self.hts = hts_hopen(hf, "<file>", self.mode)
            self.fname = None
        # reading from a File object or other object with fileno
        elif hasattr(fname, "fileno"):
            if fname.closed:
                raise IOError('I/O operation on closed file')
            hf = hdopen(fname.fileno(), self.mode)
            self.hts = hts_hopen(hf, "<file>", self.mode)
            # .name can be TextIOWrapper
            try:
                self.fname = to_bytes(fname.name)
            except AttributeError:
                self.fname = None
        else:
            raise IOError("Cannot open '%s' for writing." % str(type(fname)))

        if self.hts == NULL:
            raise IOError("Error opening %s" % str(fname))
        if reading:
            if self.hts.format.format != vcf and self.hts.format.format != bcf:
                raise IOError(
                    "%s is not valid bcf or vcf (format: %s mode: %s)" % (fname, self.hts.format.format, mode)
                )
        else:
            if self.hts.format.format != text_format and self.hts.format.format != binary_format:
                raise IOError(
                    "%s is not valid text_format or binary_format (format: %s mode: %s)" % (fname, self.hts.format.format, mode)
                )

    cdef _c_close(self):
        if self.hts != NULL:
            if self.from_path:
                hts_close(self.hts)
            self.hts = NULL

    def close(self):
        self._c_close()

cdef class VCF(HTSFile):
    """
    VCF class holds methods to iterate over and query a VCF.

    Parameters
    ----------
    fname: str
        path to file
    gts012: bool
        if True, then gt_types will be 0=HOM_REF, 1=HET, 2=HOM_ALT, 3=UNKNOWN. If False, 3, 2 are flipped.
    lazy: bool
        if True, then don't unpack (parse) the underlying record until needed.
    strict_gt: bool
        if True, then any '.' present in a genotype will classify the corresponding element in the gt_types array as UNKNOWN.
    samples: list
        list of samples to extract from full set in file.
    threads: int
        the number of threads to use including this reader.


    Returns
    -------
    VCF object for iterating and querying.
    """

    cdef const bcf_hdr_t *hdr
    cdef tbx_t *idx
    cdef hts_idx_t *hidx
    cdef int n_samples
    cdef int PASS
    cdef bint gts012
    cdef bint lazy
    cdef bint strict_gt
    cdef object _schema_chunk
    cdef object _schema_chunk_index_path
    cdef bint _schema_collect_chunk_index
    cdef list _seqnames
    cdef list _seqlens
    # holds a lookup of format field -> type.
    cdef dict format_types

    #: The constant used to indicate the the genotype is HOM_REF.
    cdef readonly int HOM_REF
    #: The constant used to indicate the the genotype is HET.
    cdef readonly int HET
    #: The constant used to indicate the the genotype is HOM_ALT.
    cdef readonly int HOM_ALT
    #: The constant used to indicate the the genotype is UNKNOWN.
    cdef readonly int UNKNOWN

    def __init__(self, fname, mode="r", gts012=False, lazy=False, strict_gt=False, samples=None, threads=None):
        cdef bcf_hdr_t *hdr
        self._schema_chunk = None
        self._schema_chunk_index_path = None
        self._schema_collect_chunk_index = False
        self._open_htsfile(fname, mode)
        hdr = self.hdr = bcf_hdr_read(self.hts)
        if hdr == NULL:
            raise Exception("cannot parse VCF header")
        if samples is not None:
            self.set_samples(samples)
        self.n_samples = bcf_hdr_nsamples(self.hdr)
        self.PASS = -1
        self.gts012 = gts012
        self.lazy = lazy
        self.strict_gt = strict_gt
        self._seqnames = []
        self._seqlens = []
        set_constants(self)
        self.format_types = {}
        if threads is not None:
            self.set_threads(threads)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def set_threads(self, int n):
        v = hts_set_threads(self.hts, n)
        if v < 0:
            raise Exception("error setting number of threads: %d" % v)

    cdef get_type(self, fmt):
        fmt = from_bytes(fmt)
        if not fmt in self.format_types:
            s = self.get_header_type(fmt, order=[BCF_HL_FMT])
            self.format_types[fmt] = s["Type"]
        return from_bytes(self.format_types[fmt])

    def add_to_header(self, line):
        """Add a new line to the VCF header.

        Parameters
        ----------
        line: str
            full vcf header line.
        """
        ret = bcf_hdr_append(self.hdr, to_bytes(line))
        if ret != 0:
            raise Exception("couldn't add '%s' to header")
        ret = bcf_hdr_sync(self.hdr)
        if ret != 0:
            raise Exception("couldn't add '%s' to header")
        if line.startswith("##contig"):
            # need to trigger a refresh of seqnames
            self._seqnames = []
        return ret

    def add_info_to_header(self, adict):
        """Add a INFO line to the VCF header.

        Parameters
        ----------
        adict: dict
            dict containing keys for ID, Number, Type, Description.
        """
        return self.add_to_header("##INFO=<ID={ID},Number={Number},Type={Type},Description=\"{Description}\">".format(**adict))

    def add_format_to_header(self, adict):
        """Add a FORMAT line to the VCF header.

        Parameters
        ----------
        adict: dict
            dict containing keys for ID, Number, Type, Description.
        """
        return self.add_to_header("##FORMAT=<ID={ID},Number={Number},Type={Type},Description=\"{Description}\">".format(**adict))

    def add_filter_to_header(self, adict):
        """Add a FILTER line to the VCF header.

        Parameters
        ----------
        adict: dict
            dict containing keys for ID, Description.
        """
        return self.add_to_header("##FILTER=<ID={ID},Description=\"{Description}\">".format(**adict))

    def set_samples(self, samples):
        """Set the samples to be pulled from the VCF; this must be called before any iteration.

        Parameters
        ----------
        samples: list
            list of samples to extract.
        """
        cdef bint is_null = 0
        if samples is None:
            samples = "-".encode()
        if isinstance(samples, list):
            samples = to_bytes(",".join(samples))
            is_null = len(samples) == 0
        else:
            samples = to_bytes(samples)

        if is_null:
            ret = bcf_hdr_set_samples(self.hdr, NULL, 0)
        else:
            ret = bcf_hdr_set_samples(self.hdr, <const char *>samples, 0)
        assert ret >= 0, ("error setting samples", ret)
        self.n_samples = bcf_hdr_nsamples(self.hdr)
        if ret != 0 and samples != "-":
            s = from_bytes(samples).split(",")
            if self.n_samples < len(s):
                warnings.warn("not all requested samples found in VCF")


    def update(self, id, type, number, description):
        """Update the header with an INFO field of the given parameters.

        Parameters
        ----------
        id: str
            ID
        type: str
            valid VCF type
        number: str
             valid VCF number
        description: str
             description of added line.
        """
        ret = bcf_hdr_append(self.hdr, "##INFO=<ID={id},Number={number},Type={type},Description=\"{description}\">".format(id=id, type=type, number=number, description=description).encode())
        if ret != 0:
            raise Exception("unable to update to header: %d", ret)
        ret = bcf_hdr_sync(self.hdr)
        if ret != 0:
            raise Exception("unable to update to header")

    def set_index(self, index_path=""):
        # Clear any existing indexes
        if self.idx != NULL:
            tbx_destroy(self.idx)
            self.idx = NULL
        if self.hidx != NULL:
            hts_idx_destroy(self.hidx)
            self.hidx = NULL

        if index_path.endswith((".tbi", ".csi")):
            self.idx = tbx_index_load2(to_bytes(self.fname), to_bytes(index_path))
            if self.idx != NULL:
                return

        self.hidx = hts_idx_load2(to_bytes(self.fname), to_bytes(index_path))
        if self.hidx == NULL:
            self.idx = tbx_index_load2(to_bytes(self.fname), to_bytes(index_path))
        if self.hidx == NULL and self.idx == NULL:
          raise OSError("unable to open index:'%s' for '%s'" % (index_path, self.fname))

    def _schema_scan_chunk(
            self, start, end, seek, threads, collect_index, index_entries_path,
            exact_start=False):
        self._schema_chunk = (start, end, seek, exact_start)
        self._schema_collect_chunk_index = collect_index
        self._schema_chunk_index_path = index_entries_path
        try:
            return self.schema_scan(threads=threads)
        finally:
            self._schema_chunk = None
            self._schema_collect_chunk_index = False
            self._schema_chunk_index_path = None

    def schema_scan(self, index_path=None, threads=None, workers=None):
        """Scan the complete VCF/BCF and return compact field statistics.

        ``index_path`` may name an existing index to reuse, or a destination
        where an index will be built during this scan if it does not exist.
        No index is created when ``index_path`` is ``None``. Index creation
        requires a BGZF-compressed, coordinate-sorted input file.
        ``threads=0`` opts into automatic concurrency: half the available CPU
        budget (honoring affinity and Linux cgroups) is used for workers,
        each with one decompression thread. A one-CPU budget decodes
        synchronously. An explicit ``workers`` overrides automatic selection.
        Otherwise ``workers`` defaults to one and positive ``threads`` sets
        the number of HTSlib decompression threads per worker.
        ``workers`` parallelizes local BGZF-compressed or uncompressed VCF/BCF
        scans across independent file ranges (indexing requires BGZF).
        VCF workers start at the first line in their range; BCF workers locate
        their first record by
        structural validation; every range start is verified against the
        previous range's end and rescanned from the verified offset if needed. Requested index entries are merged in file
        order and written after all ranges have completed.
        """
        cdef htsFile *scan_hts = NULL
        cdef bcf_hdr_t *scan_hdr = NULL
        cdef bcf1_t *record = NULL
        cdef hts_idx_t *index = NULL
        cdef tbx_t *existing_tbx = NULL
        cdef hts_idx_t *existing_index = NULL
        cdef SchemaFieldStats *field_stats = NULL
        cdef SchemaIndexEntry *chunk_entries = NULL
        cdef int32_t *genotypes = NULL
        cdef int32_t *int_values = NULL
        cdef float *float_values = NULL
        cdef int *index_rid_map = NULL
        cdef uint64_t *allele_counts = NULL
        cdef uint64_t *alleles_per_record = NULL
        cdef uint64_t *ploidy_counts = NULL
        cdef int genotype_capacity = 0
        cdef int int_capacity = 0
        cdef int float_capacity = 0
        cdef int allele_capacity = 0
        cdef int alleles_per_record_capacity = 0
        cdef int ploidy_capacity = 0
        cdef size_t chunk_entry_capacity = 0
        cdef uint64_t *resized_counts = NULL
        cdef int n_fields = 0
        cdef int n_contigs = 0
        cdef int n_samples = 0
        cdef int ret = 0
        cdef int n_values = 0
        cdef int n_genotypes = 0
        cdef int stride = 0
        cdef int sample_index = 0
        cdef int value_index = 0
        cdef int field_index = 0
        cdef int field_type = 0
        cdef int cardinality = 0
        cdef int ploidy = 0
        cdef int allele_index = 0
        cdef int index_format = HTS_FMT_CSI
        cdef int index_status = 0
        cdef int indexed_rid = 0
        cdef int min_shift = 14
        cdef int n_levels = 5
        cdef uint64_t record_count = 0
        cdef uint64_t missing_genotype_alleles = 0
        cdef int64_t record_offset = 0
        cdef int64_t index_offset = 0
        cdef int64_t final_offset = 0
        cdef int64_t chunk_start = -1
        cdef int64_t chunk_final_offset = 0
        cdef size_t chunk_entry_count = 0
        cdef const char *tag = NULL
        cdef bytes index_path_bytes
        cdef bytes index_metadata
        cdef list index_contig_names = []
        cdef bint build_index = False
        cdef bint index_available = False
        cdef bint index_created = False
        cdef bint local_index_exists = False
        cdef bint is_info
        cdef double numeric_value
        cdef SchemaFieldStats *stats
        cdef kstring_t chunk_line
        cdef BGZF *chunk_fp = NULL
        cdef BGZF *text_fp = NULL
        cdef int64_t header_end = 0
        cdef bint chunk_start_found = False
        cdef bint chunked
        cdef bint chunked_vcf = False
        cdef bint raw_chunk = False
        cdef object chunk_actual_start = None
        cdef bint collect_chunk_index
        cdef object chunk_payload = None
        cdef object chunk_index_stream = None
        cdef object chunk_config

        chunked = self._schema_chunk is not None
        collect_chunk_index = self._schema_collect_chunk_index
        chunk_config = self._schema_chunk
        if chunked:
            _chunk_start, _chunk_end, _seek_offset, _exact_start = chunk_config
            _chunk_index_path = self._schema_chunk_index_path
        if workers is not None and (not isinstance(workers, int) or workers < 1):
            raise ValueError("workers must be a positive integer")
        if threads is not None and (not isinstance(threads, int) or threads < 0):
            raise ValueError("threads must be a non-negative integer")
        if threads == 0:
            from ._schema_scan import _available_cpu_count
            cpu_budget = _available_cpu_count()
            if workers is None:
                workers = (
                    max(1, cpu_budget // 2)
                    if self.fname is not None and b"://" not in self.fname
                    else 1
                )
            threads = 1 if cpu_budget >= 2 else None
        if workers is None:
            workers = 1
        if workers > 1 and not chunked:
            if self.fname is None:
                raise ValueError("parallel schema_scan requires a VCF filename")
            if b"://" in self.fname:
                raise ValueError("parallel schema_scan currently requires a local BGZF VCF/BCF")
            is_bcf = self.hts.format.format == bcf
            if (index_path is not None and
                    self.hts.format.compression == no_compression and
                    not _schema_index_is_available(self.fname, index_path, is_bcf)):
                raise ValueError("index creation requires BGZF-compressed VCF/BCF")
            index_available = _schema_index_is_available(
                self.fname, index_path, is_bcf)
            build_index = index_path is not None and not index_available
            source_path = os.fsdecode(self.fname)
            ranges = _schema_bgzf_ranges(
                source_path, workers)
            if is_bcf:
                ranges = [(start, end, start) for start, end, _ in ranges]
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context
            from tempfile import TemporaryDirectory
            chunk_summaries = []
            with TemporaryDirectory(prefix="cyvcf2-schema-scan-") as workdir:
                tasks = [
                    (
                        source_path, start, end, seek, threads,
                        build_index,
                        os.path.join(workdir, "chunk-%d.idx" % i)
                        if build_index else None,
                        False,
                    )
                    for i, (start, end, seek) in enumerate(ranges)
                ]
                builder = (
                    _SchemaIndexBuilder(self.fname, index_path)
                    if build_index else None
                )
                with ProcessPoolExecutor(
                        max_workers=len(tasks),
                        mp_context=get_context("spawn")) as pool:
                    futures = [
                        pool.submit(_schema_scan_worker, task)
                        for task in tasks
                    ]
                    expected_start = None
                    for i, future in enumerate(futures):
                        chunk_summary = future.result()
                        if (i > 0 and
                                chunk_summary["_actual_start"] != expected_start):
                            # The located start disagrees with where the
                            # previous range ended; rescan from the verified offset.
                            task = list(tasks[i])
                            task[1] = expected_start
                            task[3] = expected_start
                            task[7] = True
                            chunk_summary = pool.submit(
                                _schema_scan_worker, tuple(task)).result()
                        expected_start = chunk_summary.pop("_next_start")
                        chunk_summary.pop("_actual_start")
                        final_offset = chunk_summary.pop("_final_offset")
                        if build_index:
                            builder.add_file(
                                chunk_summary.pop("_index_entries_path"))
                        chunk_summaries.append(chunk_summary)
                if build_index:
                    builder.finish(final_offset)
            summary = _merge_schema_scan_summaries(chunk_summaries)
            if build_index:
                index_available = True
                index_created = True
            summary["index"] = {
                "available": bool(index_available),
                "built": bool(index_created),
                "path": str(index_path) if index_path is not None else None,
            }
            return summary

        chunk_line.s = NULL
        chunk_line.l = 0
        chunk_line.m = 0

        if self.fname is None:
            raise ValueError("schema_scan requires a VCF/BCF filename")

        scan_hts = hts_open(self.fname, b"r")
        if scan_hts == NULL:
            raise IOError("Error opening %s" % self.fname)
        try:
            if threads is not None:
                ret = hts_set_threads(scan_hts, threads)
                if ret < 0:
                    raise IOError("error setting number of threads: %d" % ret)

            scan_hdr = bcf_hdr_read(scan_hts)
            if scan_hdr == NULL:
                raise IOError("cannot parse VCF header")
            n_fields = scan_hdr.n[BCF_DT_ID]
            n_contigs = scan_hdr.n[BCF_DT_CTG]
            n_samples = bcf_hdr_nsamples(scan_hdr)
            if chunked:
                if scan_hts.format.format != vcf and scan_hts.format.format != bcf:
                    raise ValueError("parallel schema_scan requires a VCF/BCF")
                chunked_vcf = scan_hts.format.format == vcf
                raw_chunk = scan_hts.format.compression == no_compression
                if not scan_hts.is_bgzf and not (chunked_vcf and raw_chunk):
                    raise ValueError("parallel schema_scan requires a BGZF-compressed or uncompressed VCF/BCF")
                if scan_hts.is_bgzf:
                    chunk_fp = scan_hts.fp.bgzf
                    header_end = bgzf_tell(chunk_fp)
                else:
                    # Plain-text VCF: BGZF passes uncompressed data through
                    # with seekable (canonicalized) virtual offsets.
                    header_end = htell(scan_hts.fp.hfile) << 16
                    text_fp = bgzf_open(self.fname, b"r")
                    if text_fp == NULL:
                        raise IOError("Error opening %s" % self.fname)
                    chunk_fp = text_fp
                if raw_chunk:
                    header_end = _schema_raw_offset(header_end)
                chunk_start = (
                    header_end
                    if _chunk_start is None or _chunk_start < 0
                    else _chunk_start
                )
                if raw_chunk:
                    chunk_start = _schema_raw_offset(chunk_start)
                chunk_actual_start = chunk_start
                if not chunked_vcf and _chunk_start is not None and _chunk_start >= 0:
                    if not _exact_start:
                        chunk_start = _schema_bcf_locate_start(
                            chunk_fp, scan_hdr, _chunk_start)
                        if raw_chunk and chunk_start >= 0:
                            chunk_start = _schema_raw_offset(chunk_start)
                        chunk_actual_start = chunk_start if chunk_start >= 0 else None
                        if chunk_start < 0:
                            chunk_start = _chunk_end
                    if chunk_start < _chunk_end:
                        if bgzf_seek(chunk_fp, chunk_start, 0) < 0:
                            raise IOError("cannot seek to BCF record")
                elif chunked_vcf:
                    if chunk_start < header_end:
                        chunk_start = header_end
                    # The actual start is the first line at or after chunk_start.
                    chunk_actual_start = None
                    if _exact_start:
                        _seek_offset = chunk_start
                    if _seek_offset is not None and _seek_offset >= 0 and _seek_offset < chunk_start:
                        if bgzf_seek(chunk_fp, _seek_offset, 0) < 0:
                            raise IOError("cannot seek to VCF chunk")
                    elif ((_schema_raw_offset(bgzf_tell(chunk_fp)) if raw_chunk
                           else bgzf_tell(chunk_fp)) != chunk_start and
                          bgzf_seek(chunk_fp, chunk_start, 0) < 0):
                        raise IOError("cannot seek to VCF chunk")
                if collect_chunk_index:
                    if _chunk_index_path is None:
                        raise ValueError("parallel index scan requires an index-entry path")
                    chunk_index_stream = open(_chunk_index_path, "wb")
                    chunk_entry_capacity = 65536
                    chunk_entries = <SchemaIndexEntry *>stdlib.malloc(
                        chunk_entry_capacity * sizeof(SchemaIndexEntry))
                    if chunk_entries == NULL:
                        raise MemoryError()
            field_stats = <SchemaFieldStats *>stdlib.calloc(
                <size_t>(2 * n_fields), sizeof(SchemaFieldStats))
            if field_stats == NULL and n_fields > 0:
                raise MemoryError()

            if not chunked and index_path is not None:
                index_path_bytes = to_bytes(str(index_path) if isinstance(index_path, Path) else index_path)
                if index_path_bytes == self.fname:
                    raise ValueError("index_path must differ from the input path")
                if index_path_bytes.endswith(b".tbi"):
                    index_format = HTS_FMT_TBI
                local_index_exists = os.path.exists(index_path)
                if local_index_exists or b"://" in index_path_bytes:
                    existing_index = hts_idx_load2(self.fname, index_path_bytes)
                    if existing_index != NULL:
                        index_available = True
                        hts_idx_destroy(existing_index)
                        existing_index = NULL
                    elif local_index_exists:
                        raise OSError("unable to read existing index: %s" % index_path)
                    else:
                        build_index = True
                else:
                    build_index = True
            elif not chunked and scan_hts.format.format == bcf:
                if (b"://" in self.fname or
                        os.path.exists(self.fname + b".csi")):
                    existing_index = bcf_index_load(self.fname)
                    if existing_index != NULL:
                        index_available = True
                        hts_idx_destroy(existing_index)
                        existing_index = NULL
            elif not chunked:
                if (b"://" in self.fname or
                        os.path.exists(self.fname + b".csi") or
                        os.path.exists(self.fname + b".tbi")):
                    existing_tbx = tbx_index_load(self.fname)
                    if (existing_tbx == NULL and
                            os.path.exists(self.fname + b".csi")):
                        existing_tbx = tbx_index_load2(
                            self.fname, self.fname + b".csi")
                    if existing_tbx != NULL:
                        index_available = True
                        tbx_destroy(existing_tbx)
                        existing_tbx = NULL

            if build_index:
                if not scan_hts.is_bgzf or scan_hts.format.compression == no_compression:
                    raise ValueError("index creation requires BGZF-compressed VCF/BCF")
                if index_format == HTS_FMT_TBI and scan_hts.format.format != vcf:
                    raise ValueError("TBI indexes can only be created for VCF files")
                _schema_index_settings(
                    scan_hdr, scan_hts.format.format == bcf, index_format,
                    &min_shift, &n_levels)
                index = hts_idx_init(
                    0 if scan_hts.format.format == vcf else n_contigs,
                    index_format, <uint64_t>bgzf_tell(scan_hts.fp.bgzf),
                    min_shift, n_levels)
                if index == NULL:
                    raise MemoryError("unable to initialize index")
                if scan_hts.format.format == vcf:
                    index_rid_map = <int *>stdlib.malloc(
                        <size_t>n_contigs * sizeof(int))
                    if index_rid_map == NULL and n_contigs > 0:
                        raise MemoryError("unable to allocate contig mapping")
                    for field_index in range(n_contigs):
                        index_rid_map[field_index] = -1
            record = bcf_init()
            if record == NULL:
                raise MemoryError("unable to initialize VCF record")
            if chunk_fp == NULL and scan_hts.is_bgzf:
                chunk_fp = scan_hts.fp.bgzf
            while True:
                if build_index or chunked:
                    record_offset = bgzf_tell(chunk_fp)
                    if raw_chunk:
                        record_offset = _schema_raw_offset(record_offset)
                if chunked and not chunked_vcf and chunk_start >= _chunk_end:
                    record_offset = chunk_start
                    break
                if chunked_vcf and not chunk_start_found and record_offset >= chunk_start:
                    chunk_start_found = True
                    chunk_actual_start = record_offset
                if chunked and record_offset >= _chunk_end:
                    break
                if chunked_vcf:
                    with nogil:
                        ret = bgzf_getline(chunk_fp, 10, &chunk_line)
                    if ret < 0:
                        if ret == -1:
                            break
                        raise IOError("error reading VCF chunk")
                    if record_offset < chunk_start:
                        continue
                    with nogil:
                        ret = vcf_parse(&chunk_line, scan_hdr, record)
                    if ret != 0:
                        raise IOError("unable to parse VCF record in chunk")
                with nogil:
                    if not chunked_vcf:
                        ret = bcf_read(scan_hts, scan_hdr, record)
                if not chunked_vcf:
                    if ret == -1:
                        break
                    if ret < 0:
                        raise IOError("error reading VCF/BCF record (error-code: %d)" % ret)

                if build_index or collect_chunk_index:
                    index_offset = bgzf_tell(chunk_fp)
                    if index_offset < 0:
                        raise IOError("unable to read BGZF virtual offset")
                if build_index:
                    indexed_rid = record.rid
                    if indexed_rid < 0 or indexed_rid >= n_contigs:
                        raise ValueError(
                            "cannot index a record with an undefined contig")
                    if scan_hts.format.format == vcf:
                        if index_rid_map[indexed_rid] < 0:
                            index_rid_map[indexed_rid] = len(index_contig_names)
                            tag = bcf_hdr_int2id(
                                scan_hdr, BCF_DT_CTG, indexed_rid)
                            if tag == NULL:
                                raise ValueError("unable to read VCF contig name")
                            index_contig_names.append(<bytes>tag)
                        indexed_rid = index_rid_map[indexed_rid]
                    if hts_idx_push(
                            index, indexed_rid, record.pos,
                            record.pos + record.rlen, <uint64_t>index_offset, 1) < 0:
                        raise ValueError(
                            "cannot build index: records must have defined contigs "
                            "and be coordinate-sorted")
                if collect_chunk_index:
                    if record.rid < 0:
                        raise ValueError("cannot index a record with an undefined contig")
                    if chunk_entry_count == chunk_entry_capacity:
                        chunk_payload = PyBytes_FromStringAndSize(
                            <char *>chunk_entries,
                            <Py_ssize_t>(chunk_entry_count * sizeof(SchemaIndexEntry)))
                        chunk_index_stream.write(chunk_payload)
                        chunk_entry_count = 0
                    chunk_entries[chunk_entry_count].rid = record.rid
                    chunk_entries[chunk_entry_count].beg = record.pos
                    chunk_entries[chunk_entry_count].end = record.pos + record.rlen
                    chunk_entries[chunk_entry_count].offset = <uint64_t>index_offset
                    chunk_entry_count += 1

                if bcf_unpack(record, 15) != 0:
                    raise IOError("unable to unpack VCF/BCF record")
                record_count += 1

                if record.n_allele > alleles_per_record_capacity:
                    resized_counts = <uint64_t *>stdlib.realloc(
                        alleles_per_record, <size_t>record.n_allele * sizeof(uint64_t))
                    if resized_counts == NULL:
                        raise MemoryError()
                    alleles_per_record = resized_counts
                    for value_index in range(alleles_per_record_capacity, record.n_allele):
                        alleles_per_record[value_index] = 0
                    alleles_per_record_capacity = record.n_allele
                if record.n_allele > 0:
                    alleles_per_record[record.n_allele - 1] += 1

                for value_index in range(record.n_info):
                    field_index = record.d.info[value_index].key
                    if field_index < 0 or field_index >= n_fields:
                        continue
                    stats = &field_stats[field_index]
                    cardinality = record.d.info[value_index].len
                    stats.records += 1
                    stats.total_values += cardinality
                    if stats.records == 1 or cardinality < stats.min_cardinality:
                        stats.min_cardinality = cardinality
                    if stats.records == 1 or cardinality > stats.max_cardinality:
                        stats.max_cardinality = cardinality
                    field_type = bcf_hdr_id2type(scan_hdr, BCF_HL_INFO, field_index)
                    tag = bcf_hdr_int2id(scan_hdr, BCF_DT_ID, field_index)
                    if tag == NULL:
                        continue
                    if field_type == BCF_HT_INT:
                        n_values = bcf_get_info_values(
                            scan_hdr, record, tag, <void **>&int_values,
                            &int_capacity, BCF_HT_INT)
                        if n_values < 0:
                            raise IOError("unable to read INFO field values")
                        for value_index in range(n_values):
                            if int_values[value_index] == bcf_int32_missing or int_values[value_index] == bcf_int32_vector_end:
                                continue
                            numeric_value = int_values[value_index]
                            if not stats.has_numeric:
                                stats.min_value = numeric_value
                                stats.max_value = numeric_value
                                stats.has_numeric = True
                            else:
                                if numeric_value < stats.min_value:
                                    stats.min_value = numeric_value
                                if numeric_value > stats.max_value:
                                    stats.max_value = numeric_value
                    elif field_type == BCF_HT_REAL:
                        n_values = bcf_get_info_values(
                            scan_hdr, record, tag, <void **>&float_values,
                            &float_capacity, BCF_HT_REAL)
                        if n_values < 0:
                            raise IOError("unable to read INFO field values")
                        for value_index in range(n_values):
                            if bcf_float_is_missing(float_values[value_index]) or bcf_float_is_vector_end(float_values[value_index]):
                                continue
                            numeric_value = float_values[value_index]
                            if not stats.has_numeric:
                                stats.min_value = numeric_value
                                stats.max_value = numeric_value
                                stats.has_numeric = True
                            else:
                                if numeric_value < stats.min_value:
                                    stats.min_value = numeric_value
                                if numeric_value > stats.max_value:
                                    stats.max_value = numeric_value

                for value_index in range(record.n_fmt):
                    field_index = record.d.fmt[value_index].id
                    if field_index < 0 or field_index >= n_fields:
                        continue
                    stats = &field_stats[n_fields + field_index]
                    cardinality = record.d.fmt[value_index].n
                    stats.records += 1
                    stats.total_values += <uint64_t>cardinality * n_samples
                    if stats.records == 1 or cardinality < stats.min_cardinality:
                        stats.min_cardinality = cardinality
                    if stats.records == 1 or cardinality > stats.max_cardinality:
                        stats.max_cardinality = cardinality
                    field_type = bcf_hdr_id2type(scan_hdr, BCF_HL_FMT, field_index)
                    tag = bcf_hdr_int2id(scan_hdr, BCF_DT_ID, field_index)
                    if tag == NULL or tag == b"GT":
                        continue
                    if field_type == BCF_HT_INT:
                        n_values = bcf_get_format_values(
                            scan_hdr, record, tag, <void **>&int_values,
                            &int_capacity, BCF_HT_INT)
                        if n_values < 0:
                            raise IOError("unable to read FORMAT field values")
                        for value_index in range(n_values):
                            if int_values[value_index] == bcf_int32_missing or int_values[value_index] == bcf_int32_vector_end:
                                continue
                            numeric_value = int_values[value_index]
                            if not stats.has_numeric:
                                stats.min_value = numeric_value
                                stats.max_value = numeric_value
                                stats.has_numeric = True
                            else:
                                if numeric_value < stats.min_value:
                                    stats.min_value = numeric_value
                                if numeric_value > stats.max_value:
                                    stats.max_value = numeric_value
                    elif field_type == BCF_HT_REAL:
                        n_values = bcf_get_format_values(
                            scan_hdr, record, tag, <void **>&float_values,
                            &float_capacity, BCF_HT_REAL)
                        if n_values < 0:
                            raise IOError("unable to read FORMAT field values")
                        for value_index in range(n_values):
                            if bcf_float_is_missing(float_values[value_index]) or bcf_float_is_vector_end(float_values[value_index]):
                                continue
                            numeric_value = float_values[value_index]
                            if not stats.has_numeric:
                                stats.min_value = numeric_value
                                stats.max_value = numeric_value
                                stats.has_numeric = True
                            else:
                                if numeric_value < stats.min_value:
                                    stats.min_value = numeric_value
                                if numeric_value > stats.max_value:
                                    stats.max_value = numeric_value

                n_genotypes = bcf_get_genotypes(
                    scan_hdr, record, &genotypes, &genotype_capacity)
                if n_genotypes > 0 and n_samples > 0:
                    if n_genotypes % n_samples != 0:
                        raise IOError("invalid genotype vector cardinality")
                    stride = n_genotypes // n_samples
                    if stride > ploidy_capacity:
                        resized_counts = <uint64_t *>stdlib.realloc(
                            ploidy_counts, <size_t>(stride + 1) * sizeof(uint64_t))
                        if resized_counts == NULL:
                            raise MemoryError()
                        ploidy_counts = resized_counts
                        for value_index in range(ploidy_capacity, stride + 1):
                            ploidy_counts[value_index] = 0
                        ploidy_capacity = stride
                    for sample_index in range(n_samples):
                        ploidy = 0
                        for value_index in range(stride):
                            allele_index = genotypes[sample_index * stride + value_index]
                            if allele_index == bcf_int32_vector_end:
                                break
                            ploidy += 1
                            if bcf_gt_is_missing(allele_index):
                                missing_genotype_alleles += 1
                                continue
                            allele_index = bcf_gt_allele(allele_index)
                            if allele_index < 0:
                                missing_genotype_alleles += 1
                                continue
                            if allele_index >= allele_capacity:
                                resized_counts = <uint64_t *>stdlib.realloc(
                                    allele_counts, <size_t>(allele_index + 1) * sizeof(uint64_t))
                                if resized_counts == NULL:
                                    raise MemoryError()
                                allele_counts = resized_counts
                                for field_index in range(allele_capacity, allele_index + 1):
                                    allele_counts[field_index] = 0
                                allele_capacity = allele_index + 1
                            allele_counts[allele_index] += 1
                        if ploidy > ploidy_capacity:
                            resized_counts = <uint64_t *>stdlib.realloc(
                                ploidy_counts, <size_t>(ploidy + 1) * sizeof(uint64_t))
                            if resized_counts == NULL:
                                raise MemoryError()
                            ploidy_counts = resized_counts
                            for value_index in range(ploidy_capacity, ploidy + 1):
                                ploidy_counts[value_index] = 0
                            ploidy_capacity = ploidy
                        ploidy_counts[ploidy] += 1

            if chunked:
                chunk_final_offset = bgzf_tell(chunk_fp)
                if collect_chunk_index:
                    if chunk_entry_count:
                        chunk_payload = PyBytes_FromStringAndSize(
                            <char *>chunk_entries,
                            <Py_ssize_t>(chunk_entry_count * sizeof(SchemaIndexEntry)))
                        chunk_index_stream.write(chunk_payload)
                    chunk_index_stream.close()
                    chunk_index_stream = None
            if build_index:
                final_offset = bgzf_tell(scan_hts.fp.bgzf)
                if final_offset < 0 or hts_idx_finish(index, <uint64_t>final_offset) != 0:
                    raise ValueError("unable to finish index; input may not be coordinate-sorted")
                if scan_hts.format.format == vcf:
                    index_metadata = _schema_vcf_index_metadata(
                        index_contig_names)
                    if hts_idx_set_meta(
                            index, len(index_metadata),
                            <uint8_t *>PyBytes_AS_STRING(index_metadata), 1) < 0:
                        raise MemoryError("unable to set tabix index metadata")
                index_status = hts_idx_save_as(
                    index, self.fname, index_path_bytes, index_format)
                if index_status < 0:
                    raise IOError("unable to save index to %s" % index_path)
                index_available = True
                index_created = True

            fields = {"INFO": {}, "FORMAT": {}}
            for is_info in (True, False):
                for field_index in range(n_fields):
                    stats = &field_stats[field_index if is_info else n_fields + field_index]
                    tag = bcf_hdr_int2id(scan_hdr, BCF_DT_ID, field_index)
                    if tag == NULL:
                        continue
                    field_name = from_bytes(tag)
                    try:
                        header_values = self.get_header_type(
                            field_name,
                            order=[BCF_HL_INFO if is_info else BCF_HL_FMT])
                    except KeyError:
                        continue
                    numeric_bounds = None
                    if stats.has_numeric:
                        if header_values.get("Type") == "Integer":
                            numeric_bounds = {
                                "min": int(stats.min_value),
                                "max": int(stats.max_value),
                            }
                        else:
                            numeric_bounds = {
                                "min": stats.min_value,
                                "max": stats.max_value,
                            }
                    field_summary = {
                        "type": header_values.get("Type"),
                        "number": header_values.get("Number"),
                        "records": int(stats.records),
                        "total_values": int(stats.total_values),
                        "cardinality_unit": "per_record" if is_info else "per_sample",
                        "cardinality": {
                            "min": stats.min_cardinality if stats.records else None,
                            "max": stats.max_cardinality if stats.records else None,
                        },
                        "numeric_bounds": numeric_bounds,
                    }
                    fields["INFO" if is_info else "FORMAT"][field_name] = field_summary

            allele_count_summary = {
                "by_index": [int(allele_counts[value_index]) for value_index in range(allele_capacity)],
                "missing": int(missing_genotype_alleles),
            }
            summary = {
                "records": int(record_count),
                "samples": n_samples,
                "alleles_per_record": {
                    value_index + 1: int(alleles_per_record[value_index])
                    for value_index in range(alleles_per_record_capacity)
                    if alleles_per_record[value_index] > 0
                },
                "allele_counts": allele_count_summary,
                "ploidy": {
                    value_index: int(ploidy_counts[value_index])
                    for value_index in range(ploidy_capacity + 1)
                    if ploidy_counts != NULL and ploidy_counts[value_index] > 0
                },
                "fields": fields,
                "index": {
                    "available": bool(index_available),
                    "built": bool(index_created),
                    "path": from_bytes(index_path_bytes) if index_path is not None else None,
                },
            }
            if chunked:
                summary["_final_offset"] = chunk_final_offset
                summary["_actual_start"] = chunk_actual_start
                summary["_next_start"] = record_offset
                if collect_chunk_index:
                    summary["_index_entries_path"] = _chunk_index_path
            return summary
        finally:
            if record != NULL:
                bcf_destroy(record)
            if index != NULL:
                hts_idx_destroy(index)
            if existing_index != NULL:
                hts_idx_destroy(existing_index)
            if existing_tbx != NULL:
                tbx_destroy(existing_tbx)
            if scan_hdr != NULL:
                bcf_hdr_destroy(scan_hdr)
            if scan_hts != NULL:
                hts_close(scan_hts)
            if text_fp != NULL:
                bgzf_close(text_fp)
            stdlib.free(field_stats)
            stdlib.free(genotypes)
            stdlib.free(int_values)
            stdlib.free(float_values)
            stdlib.free(index_rid_map)
            stdlib.free(allele_counts)
            stdlib.free(alleles_per_record)
            stdlib.free(ploidy_counts)
            stdlib.free(chunk_entries)
            ks_free(&chunk_line)
            if chunk_index_stream is not None:
                chunk_index_stream.close()

    def _bcf_region(VCF self, region):
        if self.hidx == NULL:
            self.hidx = bcf_index_load(self.fname)
        assert self.hidx != NULL, ("error loading .csi index for %s" % self.fname)
        cdef bcf1_t *b
        cdef int ret
        cdef hts_itr_t *itr

        itr = bcf_itr_querys(self.hidx, self.hdr, to_bytes(region))
        if itr == NULL:
            warnings.warn("no intervals found for %s at %s" % (self.fname, region))
            return
        try:
            while True:
                b = bcf_init()
                ret = bcf_itr_next(self.hts, itr, b)
                if ret < 0:
                    bcf_destroy(b)
                    break
                ret = bcf_subset_format(self.hdr, b)
                assert ret == 0, ("could not subset variant", self.fname, region)
                yield newVariant(b, self)
        finally:
            if itr != NULL:
                hts_itr_destroy(itr)

    def __call__(VCF self, region=None):
        """
        Extract the region from the VCF.

        Parameters
        ----------
        region: str
           region string like chr1:1234-34566 or 'chr7

        Returns
        -------
        An Iterator over the requested region.

        Notes
        -----
        If `region` is empty, cyvcf2 uses `HTS_IDX_START` to iterate from the
        start of the index. This mode requires an index; without an index, the
        iterator is empty.
        """
        cdef hts_itr_t *itr
        cdef kstring_t s
        cdef bcf1_t *b
        cdef int slen = 1, ret = 0
        cdef bytes bregion
        cdef char *cregion

        if not region:
            self._open_index()
            if self.hidx == NULL and self.idx == NULL:
                return

            if self.hidx != NULL:
                itr = bcf_itr_queryi(self.hidx, HTS_IDX_START, 0, 0)
                if itr == NULL:
                    return
                try:
                    while True:
                        b = bcf_init()
                        ret = bcf_itr_next(self.hts, itr, b)
                        if ret < 0:
                            bcf_destroy(b)
                            break
                        ret = bcf_subset_format(self.hdr, b)
                        assert ret == 0, ("could not subset variant", self.fname, region)
                        yield newVariant(b, self)
                finally:
                    hts_itr_destroy(itr)
                return

            itr = tbx_itr_queryi(self.idx, HTS_IDX_START, 0, 0)
            if itr == NULL:
                # no iterator was created, so there is nothing to destroy
                return
            s.s, s.l, s.m = NULL, 0, 0
            try:
                while 1:
                    with nogil:
                        slen = tbx_itr_next(self.hts, self.idx, itr, &s)
                        if slen > 0:
                            b = bcf_init()
                            ret = vcf_parse(&s, self.hdr, b)
                    if slen <= 0:
                        break
                    if ret != 0:
                        bcf_destroy(b)
                        # s.s and itr are released by the finally block
                        raise Exception("error parsing: vcf_parse returned %d" % ret)
                    yield newVariant(b, self)
            finally:
                stdlib.free(s.s)
                hts_itr_destroy(itr)
            return

        if self.fname.decode(ENC).endswith(('.bcf', '.bcf.gz')):
            yield from self._bcf_region(region)
            return

        if self.hidx != NULL:
            yield from self._bcf_region(region)
            return

        if self.idx == NULL:
            self.idx = tbx_index_load(to_bytes(self.fname))
            if self.idx == NULL:
                csi_path = to_bytes(self.fname) + b".csi"
                self.idx = tbx_index_load2(to_bytes(self.fname), csi_path)
            assert self.idx != NULL, "error loading tabix index for %s" % self.fname

        bregion = to_bytes(region)
        cregion = bregion

        with nogil:
            itr = tbx_itr_querys(self.idx, cregion)

        if itr == NULL:
            warnings.warn("no intervals found for %s at %s" % (self.fname, region))
            return

        s.s, s.l, s.m = NULL, 0, 0
        try:
            while 1:
                with nogil:
                    slen = tbx_itr_next(self.hts, self.idx, itr, &s)
                    if slen > 0:
                            b = bcf_init()
                            ret = vcf_parse(&s, self.hdr, b)
                if slen <= 0:
                    break
                if ret != 0:
                    bcf_destroy(b)
                    # s.s and itr are released by the finally block
                    raise Exception("error parsing: vcf_parse returned %d" % ret)
                yield newVariant(b, self)
        finally:
            stdlib.free(s.s)
            hts_itr_destroy(itr)


    def header_iter(self):
        """
        Iterate over fields in the HEADER
        """
        cdef int i
        for i in range(self.hdr.nhrec):
            yield newHREC(self.hdr.hrec[i], self.hdr)

    def ibd(self, int nmax=-1):
        assert self.gts012
        import itertools

        cdef int i, rl, n_bins = 16

        samples = self.samples
        sample_to_idx = {s: samples.index(s) for s in samples}
        sample_pairs = list(itertools.combinations(samples, 2))
        # values of bins, run_length

        cdef int n = 0
        cdef float pi
        cdef int[:] b
        cdef int[:] gts
        cdef int idx0, idx1
        bins = np.zeros((len(sample_pairs), n_bins), dtype=np.int32)
        rls = np.zeros(len(sample_pairs), dtype=np.int32)

        for v in self:
            if n == nmax: break
            n += 1
            gts = v.gt_types
            pi = v.aaf
            for i, (s0, s1) in enumerate(sample_pairs):
                b = bins[i, :]
                idx0, idx1 = sample_to_idx[s0], sample_to_idx[s1]
                rls[i] = ibd(gts[idx0], gts[idx1], rls[i], pi, &b[0], n_bins)

        return {sample_pairs[i]: bins[i, :] for i in range(len(sample_pairs))}

    # pull something out of the HEADER, e.g. CSQ
    cpdef get_header_type(self, key, order=[BCF_HL_INFO, BCF_HL_FMT]):
        """Extract a field from the VCF header by id.

        Parameters
        ----------
        key: str
           ID to pull from the header.
        Returns
        -------
        rec: dict
            dictionary containing header information.
        """
        key = to_bytes(key)
        cdef bcf_hrec_t *b
        cdef int i
        for typ in order:
            b = bcf_hdr_get_hrec(self.hdr, typ, b"ID", key, NULL);
            if b != NULL:
                break
        if b == NULL:
            b = bcf_hdr_get_hrec(self.hdr, BCF_HL_GEN, key, NULL, NULL);
            if b == NULL:
                raise KeyError(key)
            d = {from_bytes(b.key): from_bytes(b.value)}
        else:
            d =  {from_bytes(b.keys[i]): from_bytes(b.vals[i]) for i in range(b.nkeys)}
        #bcf_hrec_destroy(b)
        return d

    cpdef remove_header(self, key, order=[BCF_HL_INFO, BCF_HL_FMT, BCF_HL_FLT]):
        """Remove a header record from the VCF by ID.

        Parameters
        ----------
        key: str
            ID to pull from the header.
        """
        key = to_bytes(key)
        cdef bcf_hrec_t *b

        for typ in order:
            b = bcf_hdr_get_hrec(self.hdr, typ, b"ID", key, NULL)
            if b != NULL:
                break

        if b == NULL:
            raise KeyError(key)

        bcf_hdr_remove(<bcf_hdr_t*>self.hdr, typ, <const char*>key)
        bcf_hdr_sync(<bcf_hdr_t*>self.hdr)

    def __getitem__(self, key):
        return self.get_header_type(key)

    def __contains__(self, key):
        """Check if the given ID is in the header."""
        try:
            self[key]
            return True
        except KeyError:
            return False

    contains = __contains__

    def __dealloc__(self):
        if self.hdr != NULL:
            bcf_hdr_destroy(self.hdr)
            self.hdr = NULL

        # Shouldn't call Python-space functions here
        self._c_close()
        if self.idx != NULL:
            tbx_destroy(self.idx)
        if self.hidx != NULL:
            hts_idx_destroy(self.hidx)

    def __iter__(self):
        return self

    def __next__(self):

        cdef bcf1_t *b
        cdef int ret
        if self.hts == NULL:
            raise Exception("attempt to iterate over closed/invalid VCF")
        with nogil:
            b = bcf_init()
            ret = bcf_read(self.hts, self.hdr, b)
        if ret >= 0 or b.errcode == BCF_ERR_CTG_UNDEF:
            return newVariant(b, self)
        else:
            bcf_destroy(b)
        if  ret == -1:  # end-of-file
            raise StopIteration
        else:
            raise Exception("error parsing variant with `htslib::bcf_read` error-code: %d and ret: %d" % (
                b.errcode, ret))


    property samples:
        "list of samples pulled from the VCF."
        def __get__(self):
            cdef int i
            return [str(self.hdr.samples[i].decode('utf-8')) for i in range(self.n_samples)]

    property raw_header:
        "string of the raw header from the VCF"
        def __get__(self):
            cdef kstring_t s
            s.s, s.l, s.m = NULL, 0, 0
            bcf_hdr_format(self.hdr, 0, &s)

            python_str = str(from_bytes(s.s))
            ks_free(&s)
            return python_str

    property seqlens:
        "list of chromosome lengths, if defined in the VCF header"
        def __get__(self):
            if len(self._seqlens) > 0: return self._seqlens
            cdef int32_t nseq;
            cdef int32_t* sls = bcf_hdr_seqlen(self.hdr, &nseq)
            if sls == NULL or nseq <= 0:
              raise AttributeError("no sequence lengths found in header")

            self._seqlens = [sls[i] for i in range(nseq)]
            stdlib.free(sls)
            return self._seqlens

    cdef _open_index(self):
        """
        Try to open an index, if not open already.
        """
        if self.hidx == NULL and self.idx == NULL:
            if self.fname.decode(ENC).endswith(('.bcf', '.bcf.gz')):
                self.hidx = bcf_index_load(self.fname)
            else:
                self.idx = tbx_index_load(to_bytes(self.fname))

    property seqnames:
        "list of chromosomes in the VCF"
        def __get__(self):
            if len(self._seqnames) > 0: return self._seqnames
            cdef const char **cnames
            cdef int i, n = 0
            cnames = bcf_hdr_seqnames(self.hdr, &n)
            if n == 0:
                self._open_index()
                if self.hidx != NULL:
                    cnames = bcf_index_seqnames(self.hidx, self.hdr, &n)
                if self.idx !=NULL:
                    cnames = tbx_seqnames(self.idx, &n)
            self._seqnames = [cnames[i].decode() for i in range(n)]
            stdlib.free(cnames)
            return self._seqnames

    cdef _num_records(self):
        cdef uint64_t total, records, v;
        cdef int ret, tid, nseq;
        cdef hts_idx_t *idx = NULL;

        self._open_index()
        if self.hidx != NULL:
            idx = self.hidx
            assert self.idx == NULL
        if self.idx != NULL:
            idx = self.idx.idx
            assert self.hidx == NULL

        if idx == NULL:
            raise ValueError(
                "File must be indexed to compute num_records (tip: use bcftools index)")

        nseq = hts_idx_nseq(idx)
        total = 0;
        for tid in range(nseq):
            # NOTE: the return value here doesn't seem to indicate an error
            # condition, and correct values are computed when it returns < 0.
            # bcftools index -n doesn't strictly check the output.
            hts_idx_get_stat(idx, tid, &records, &v);
            total += records
        return total

    property num_records:
        """
        The number of VCF records in the file, computed from the index.
        If the file is not indexed (or an index has not been set using
        ``set_index``) a ValueError is raised.

        Note that incorrect values may be returned if a mismatched
        index file (i.e., the index for a different VCF file) is used.
        This is not detected as an error condition.
        """
        def __get__(self):
            return self._num_records()

    def plot_relatedness(self, riter):
        import pandas as pd
        from matplotlib import pyplot as plt
        from matplotlib import gridspec
        import seaborn as sns
        sns.set_style("ticks")

        df = []
        for row in riter:
          row['jtags'] = '|'.join(row['tags'])
          df.append(row)

        df = pd.DataFrame(df)
        fig = plt.figure(figsize=(9, 9))

        gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1])

        ax0, ax1 = plt.subplot(gs[0]), plt.subplot(gs[1])

        if "error" in df.columns:
            # plot all gray except points that don't match our expectation.
            import matplotlib
            matplotlib.rcParams['pdf.fonttype'] = 42
            import matplotlib.colors as mc
            colors = [mc.hex2color(h) for h in ('#b6b6b6', '#ff3333')]
            for i, err in enumerate(("ok", "error")):
                subset = df[df.error == err]
                subset.plot(kind='scatter', x='rel', y='ibs0', c=colors[i],
                          edgecolor=colors[0],
                          label=err, ax=ax0, s=17 if i == 0 else 35)
            sub = df[df.error == "error"]
            for i, row in sub.iterrows():
                ax0.annotate(row['sample_a'] + "\n" + row['sample_b'],
                        (row['rel'], row['ibs0']), fontsize=8)
        else:
            # color by the relation derived from the genotypes.
            colors = sns.color_palette("Set1", len(set(df.jtags)))
            for i, tag in enumerate(set(df.jtags)):
                subset = df[df.jtags == tag]
                subset.plot(kind='scatter', x='rel', y='ibs0', c=colors[i],
                          label=tag, ax=ax0)

            ax0.legend()

        ax0.set_ylim(ymin=0)
        ax0.set_xlim(xmin=df.rel.min())

        ax1.set_xlim(*ax0.get_xlim())
        ax1.hist(df.rel, 40)
        ax1.set_yscale('log', nonposy='clip')
        return fig

    def gen_variants(self, sites,
                    offset=0, each=1, call_rate=0.8):

        seqnames = set(self.seqnames)

        if sites is not None:
            if isinstance(sites, basestring):
                isites = []
                for i in (x.strip().split(":") for x in open(sites)):
                    if not i[0] in seqnames and 'chr' + i[0] in seqnames:
                        i[0] = 'chr' + i[0]
                    i[1] = int(i[1])
                    isites.append(i)
            else:
                isites = sites
                for i in isites:
                    if not i[0] in seqnames and 'chr' + i[0] in seqnames:
                        i[0] = 'chr' + i[0]

        cdef Variant v
        cdef int k, last_pos
        if sites:
            isites = isites[offset::each]
            ref, alt = None, None
            j = 0
            for i, osite in enumerate(isites):
                if len(osite) >= 4:
                    chrom, pos, ref, alt = osite[:4]
                else:
                    chrom, pos = osite[:2]
                for v in self("%s:%s-%s" % (chrom, pos, pos)):
                    if len(v.ALT) != 1: continue
                    if ref is not None and v.REF != ref: continue
                    if alt is not None and v.ALT[0] != alt: continue
                    if v.call_rate < call_rate: continue
                    yield i, v
                    j += 1
                    break
        else:
            last_pos, k = -10000, 0
            for v in self:
                if abs(v.POS - last_pos) < 5000: continue
                if len(v.REF) != 1: continue
                if len(v.ALT) != 1: continue
                if v.call_rate < 0.5: continue
                if not 0.03 < v.aaf < 0.6: continue
                if np.mean(v.gt_depths > 7) < 0.5: continue
                last_pos = v.POS
                if k >= offset and k % each == 0:
                    yield k, v
                k += 1
                if k > 20000: break

    def het_check(self, min_depth=8, percentiles=(10, 90), _finish=True,
                  int each=1, int offset=0,
                  sites=None):

        cdef int i, k, n_samples = len(self.samples)
        cdef Variant v
        cdef np.ndarray het_counts = np.zeros((n_samples,), dtype=np.int32)

        cdef np.ndarray sum_depths = np.zeros((n_samples,), dtype=np.int32)
        cdef np.ndarray sum_counts = np.zeros((n_samples,), dtype=np.int32)
        cdef int any_counts = 0

        # keep the sites and gts that we used for PCA
        used_sites, all_gt_types = [], []

        mean_depths = []

        maf_lists = defaultdict(list)
        idxs = np.arange(n_samples)
        for i, v in self.gen_variants(sites, each=each, offset=offset):
            if v.CHROM in ('X', 'chrX'): break
            if v.aaf < 0.01: continue
            if v.call_rate < 0.5: continue
            used_sites.append("%s:%d:%s:%s" % (v.CHROM, v.start + 1, v.REF, v.ALT[0]))
            alts = v.gt_alt_depths
            assert len(alts) == n_samples
            depths = (alts + v.gt_ref_depths).astype(np.int32)
            sum_depths += depths
            sum_counts += (depths > min_depth)
            any_counts += 1
            mean_depths.append(depths)

            mafs = alts / depths.astype(float)
            gt_types = v.gt_types
            hets = gt_types == 1
            het_counts[hets] += 1
            for k in idxs[hets]:
                if depths[k] <= min_depth: continue
                maf_lists[k].append(mafs[k])
            all_gt_types.append(np.array(gt_types, dtype=np.uint8))

        if _finish:
            mean_depths = np.array(mean_depths, dtype=np.int32).T
            return self._finish_het(mean_depths, maf_lists,
                                    percentiles,
                                    het_counts, sum_counts, all_gt_types,
                                    used_sites, any_counts)
        return (mean_depths, maf_lists, het_counts, sum_counts,
                all_gt_types, used_sites, any_counts)


    def _finish_het(self, mean_depths, maf_lists, percentiles, het_counts,
            sum_counts, all_gt_types, sites, any_counts):

            sample_ranges = {}
            for i, sample in enumerate(self.samples):
                qs = np.asarray(np.percentile(maf_lists[i] or [0], percentiles))
                sample_ranges[sample] = dict(zip(['p' + str(p) for p in percentiles], qs))
                sample_ranges[sample]['range'] = qs.max() - qs.min()
                sample_ranges[sample]['het_ratio'] = het_counts[i] / float(any_counts)
                sample_ranges[sample]['het_count'] = het_counts[i]
                sample_ranges[sample]['sampled_sites'] = sum_counts[i]
                sample_ranges[sample]['mean_depth'] = np.mean(mean_depths[i])
                sample_ranges[sample]['median_depth'] = np.median(mean_depths[i])
            # used for the peddy paper.
            if os.environ.get('PEDDY_MAF_DUMP'):
                path = os.environ['PEDDY_MAF_DUMP']
                import cPickle
                cPickle.dump({s: maf_lists[i] for i, s in enumerate(self.samples)}, open(path, 'wb', -1))


            return sample_ranges, sites, np.transpose(all_gt_types)

    def site_relatedness(self, sites=None,
                         min_depth=5, each=1):

        vibs, vn, vhet = self._site_relatedness(sites=sites, min_depth=min_depth, each=each)
        return self._relatedness_finish(vibs, vn, vhet)

    cdef _site_relatedness(self, sites=None,
            int min_depth=5, int each=1, int offset=0):
        """
        sites must be an file of format: chrom:pos1:ref:alt where
        we match on all parts.
        it must have a matching file with a suffix of .bin.gz that is the binary
        genotype data. with 0 == hom_ref, 1 == het, 2 == hom_alt, 3 == unknown.
        min_depth applies per-sample
        """
        cdef int n_samples = len(self.samples)
        cdef int k, i
        assert each >= 0


        cdef int32_t[:, ::view.contiguous] ibs = np.zeros((n_samples, n_samples), np.int32)
        cdef int32_t[:, ::view.contiguous] n = np.zeros((n_samples, n_samples), np.int32)
        cdef int32_t[:] hets = np.zeros((n_samples, ), np.int32)
        cdef int32_t[:] gt_types = np.zeros((n_samples, ), np.int32)
        cdef int32_t[:] depths = np.zeros((n_samples, ), np.int32)
        cdef double[:] alt_freqs = np.zeros((n_samples,), np.double)

        cdef Variant v

        for j, (i, v) in enumerate(self.gen_variants(sites, offset=offset, each=each)):
            gt_types = v.gt_types
            alt_freqs = v.gt_alt_freqs
            krelated(&gt_types[0], &ibs[0, 0], &n[0, 0], &hets[0], n_samples,
                    &alt_freqs[0])

        return ibs, n, hets

    def relatedness(self, int n_variants=35000, int gap=30000, float min_af=0.04,
                    float max_af=0.8, float linkage_max=0.2, min_depth=8):
        cdef Variant v

        cdef int last = -gap, nv = 0, nvt=0
        cdef int32_t *last_gts = NULL
        samples = self.samples
        cdef int n_samples = len(samples)
        cdef float aaf = 0.0
        cdef int n_unlinked = 0

        cdef int32_t[:, ::view.contiguous] ibs = np.zeros((n_samples, n_samples), np.int32)
        cdef int32_t[:, ::view.contiguous] n = np.zeros((n_samples, n_samples), np.int32)
        cdef int32_t[:] hets = np.zeros((n_samples, ), np.int32)
        cdef int32_t[:] gt_types = np.zeros((n_samples, ), np.int32)

        for v in self:
            nvt += 1
            if last_gts == NULL:
                if v._gt_types == NULL:
                    v.gt_types
                last_gts = v._gt_types
            if v.POS - last < gap and v.POS > last:
                continue
            if v.call_rate < 0.5: continue
            # require half of the samples to meet the min depth
            if np.mean(v.gt_depths > min_depth) < 0.5: continue
            aaf = v.aaf
            if aaf < min_af: continue
            if aaf > max_af: continue
            if linkage_max < 1 and v.POS - last < 40000:
                if v._gt_types == NULL:
                    v.gt_types
                # require 5 unlinked variants
                if r_unphased(last_gts, v._gt_types, 1e-5, n_samples) > linkage_max:
                    continue
                n_unlinked += 1
                if n_unlinked < 5:
                    continue

            n_unlinked = 0

            if v._gt_types == NULL:
                gt_types = v.gt_types
            last, last_gts = v.POS, v._gt_types

            v.relatedness(ibs, n, hets)
            nv += 1
            if nv == n_variants:
                break
        warnings.warn("tested: %d variants out of %d" % (nv, nvt))
        return self._relatedness_finish(ibs, n, hets)

    cdef dict _relatedness_finish(self,
                                  int32_t[:, ::view.contiguous] _ibs,
                                  int32_t[:, ::view.contiguous] _n,
                                  int32_t[:] _hets):
        samples = self.samples
        cdef int sj, sk, ns = len(samples)

        res = {'sample_a': [], 'sample_b': [],
                'rel': array('f'),
                'hets_a': array('I'),
                'hets_b': array('I'),
               'shared_hets': array('I'),
               'ibs0': array('I'),
               'ibs2': array('I'),
               'n': array('I')}

        cdef float bot

        for sj in range(ns):
            sample_j = samples[sj]
            if _hets[sj] == 0:
                warnings.warn("peddy: no hets found for sample %s" % sample_j)
            for sk in range(sj, ns):
                if sj == sk: continue
                sample_k = samples[sk]

                # calculate relatedness. we use the geometric mean.
                #bot = math.exp(0.5 * (math.log(1 + _hets[sk]) + math.log(1 + _hets[sj])))
                #bot = (_hets[sk] + _hets[sj])/2.0
                bot = min(_hets[sk], _hets[sj])
                if bot == 0:
                    bot = max(_hets[sk], _hets[sj])
                    if bot == 0:
                        # set to negative value if we are unable to calculate it.
                        bot = -1

                phi = (_ibs[sk, sj] - 2.0 * _ibs[sj, sk]) / (bot)

                res['sample_a'].append(sample_j)
                res['sample_b'].append(sample_k)
                res['hets_a'].append(_hets[sj])
                res['hets_b'].append(_hets[sk])
                res['rel'].append(phi) # rel is 2*kinship
                res['ibs0'].append(_ibs[sj, sk])
                res['shared_hets'].append(_ibs[sk, sj])
                res['ibs2'].append(_n[sk, sj])
                res['n'].append(_n[sj, sk])
        return res

cdef class Allele(object):
    cdef int32_t *_raw
    cdef int i

    cdef int _value(self):
        if self._raw[self.i] < 0: return self._raw[self.i]
        return (self._raw[self.i] >> 1) - 1

    @property
    def phased(self):
        return self._raw[self.i] & 1 == 1

    @phased.setter
    def phased(self, bint ph):
        if ph:
            self._raw[self.i] = (self._value() + 1)<<1|1
        else:
            self._raw[self.i] = (self._value() + 1)<<1

    @property
    def value(self):
        if self._raw[self.i] < 0: return self._raw[self.i]
        return (self._raw[self.i] >> 1) - 1

    @value.setter
    def value(self, int value):
        if value < 0:
            self._raw[self.i] = value
            return
        if self.phased:
            self._raw[self.i] = (value + 1)<<1|1
        else:
            self._raw[self.i] = (value + 1)<<1

    def __repr__(self):
        if self.value < 0: return "."
        return str(self.value) + ("|" if self.phased else "/")

cdef inline Allele newAllele(int32_t *raw, int i):
    cdef Allele a = Allele.__new__(Allele)
    a._raw = raw
    a.i = i
    return a

cdef class Genotypes(object):
    cdef int32_t *_raw
    cdef readonly int n_samples
    cdef readonly int ploidy
    def __cinit__(self):
        self.ploidy = 0
        self.n_samples = 0
        self._raw = NULL
    def __dealloc__(self):
        if self._raw != NULL:
            stdlib.free(self._raw)

    def phased(self, int i):
        """
        a boolean indicating that the ith sample is phased.
        """
        return (self._raw[i * self.ploidy + 1] & 1) == 1

    def alleles(self, int i):
        cdef list result = []
        cdef int32_t v
        for j in range(self.ploidy):
            v = self._raw[i * self.ploidy + j]
            if v != bcf_int32_vector_end:
                result.append((v >> 1) - 1)
        return result

    def __repr__(self):
        return str(self.array())

    def array(Genotypes self, int fill=-2):
        """
        array returns an int16 numpy array  of shape n_samples, (ploidy + 1).
        The last column indicates phased (1 is phased, 0 is unphased).
        The other columns indicate the alleles, e.g. [0, 1, 1] is 0|1.
        Unknown alleles are represented by -1.
        If a mixture of ploidy levels are present then the array is padded
        with the `fill` value (default = -2) to indicate non-alleles.
        """
        cdef np.ndarray[np.int16_t, ndim=2] to_return = np.zeros((self.n_samples, self.ploidy + 1),
                dtype=np.int16)

        cdef int raw
        cdef int ind
        cdef int allele
        cdef int p = self.ploidy + 1

        for ind in range(self.n_samples):
            for allele in range(self.ploidy):
                raw = self._raw[ind * self.ploidy + allele]
                if raw == bcf_int32_vector_end:
                    to_return[ind, allele] = fill
                else:
                    to_return[ind, allele] = (raw >> 1) - 1
            to_return[ind, self.ploidy] = (self._raw[ind * self.ploidy + 1] & 1) == 1

        return to_return

    def __getitem__(self, int i):
        ## return the Allele objects for the i'th sample.
        cdef int k
        return [newAllele(self._raw, k) for k in range(i*self.ploidy,(i+1)*self.ploidy)]

cdef inline Genotypes newGenotypes(int32_t *raw, int ploidy, int n_samples):
    cdef Genotypes gs = Genotypes.__new__(Genotypes)
    gs._raw = raw
    gs.ploidy = ploidy
    gs.n_samples = n_samples
    return gs

cdef class Variant(object):
    """
    Variant represents a single VCF Record.

    It is created internally by iterating over a VCF.

    Attributes
    ----------

    INFO: `INFO`
       a dictionary-like field that provides access to the VCF INFO field.

    POS: the 1-based variant start.

    """
    cdef bcf1_t *b
    cdef VCF vcf
    cdef int32_t *_gt_types
    cdef int *_gt_ref_depths
    cdef int *_gt_alt_depths
    cdef float *_gt_alt_freqs
    cdef void *fmt_buffer
    cdef int *_gt_phased
    cdef float *_gt_quals
    cdef int *_int_gt_quals
    cdef int *_gt_idxs
    cdef int _gt_nper
    cdef int *_gt_pls
    cdef float *_gt_gls
    cdef readonly INFO INFO
    cdef int _ploidy
    cdef list _genotypes

    cdef readonly int POS

    def __init__(self, *args, **kwargs):
        raise TypeError("Variant object cannot be instantiated directly.")

    def __cinit__(self):
        self.b = NULL
        self._gt_types = NULL
        self._gt_phased = NULL
        self._gt_pls = NULL
        self._ploidy = -1

    def __repr__(self):
        return "Variant(%s:%d %s/%s)" % (self.CHROM, self.POS, self.REF, ",".join(self.ALT))

    def __str__(self):
        cdef kstring_t s
        s.s, s.l, s.m = NULL, 0, 0
        vcf_format(self.vcf.hdr, self.b, &s)
        try:
            return s.s[:s.l].decode()
        finally:
            stdlib.free(s.s)

    def __dealloc__(self):
        if self.b is not NULL:
            bcf_destroy(self.b)
            self.b = NULL
        if self._gt_types != NULL:
            stdlib.free(self._gt_types)
        if self._gt_ref_depths != NULL:
            stdlib.free(self._gt_ref_depths)
        if self._gt_alt_depths != NULL:
            stdlib.free(self._gt_alt_depths)
        if self._gt_alt_freqs != NULL:
            stdlib.free(self._gt_alt_freqs)
        if self._gt_phased != NULL:
            stdlib.free(self._gt_phased)
        if self._gt_quals != NULL:
            stdlib.free(self._gt_quals)
        if self._int_gt_quals != NULL:
            stdlib.free(self._int_gt_quals)
        if self._gt_idxs != NULL:
            stdlib.free(self._gt_idxs)
        if self._gt_pls != NULL:
            stdlib.free(self._gt_pls)
        if self._gt_gls != NULL:
            stdlib.free(self._gt_gls)

    property gt_bases:
        "numpy array indicating the alleles in each sample."
        def __get__(self):
            cdef np.ndarray gt_types = self.gt_types
            cdef int i, n = self.ploidy, j=-1, a, b
            cdef char **alleles = self.b.d.allele
            #cdef dict d = {i:alleles[i] for i in range(self.b.n_allele)}
            cdef list d = [from_bytes(alleles[i]) for i in range(self.b.n_allele)]
            d.append(".") # -1 gives .
            cdef list bases = ["./." for _ in range(self.vcf.n_samples)]
            cdef np.ndarray phased = (self.gt_phases).astype(int)
            cdef list lookup = ["/", "|"]
            cdef int unknown = 3 if self.vcf.gts012 else 2
            for i in range(0, n * self.vcf.n_samples, n):
                j += 1
                if n == 2:
                    if (gt_types[j] == unknown) and (not self.vcf.strict_gt):
                        continue
                    else:
                        a = self._gt_idxs[i]
                        b = self._gt_idxs[i + 1]
                        if a >= -1 and b >= -1:
                          bases[j] = d[a] + lookup[phased[j]] + d[b]
                        else:
                          bases[j] = d[max(-1, a)]
                elif n == 1:
                    bases[j] = d[self._gt_idxs[i]]
                else:
                    raise Exception("gt_bases not implemented for ploidy > 2")

            return np.array(bases, str)

    def relatedness(self,
                    int32_t[:, ::view.contiguous] ibs,
                    int32_t[:, ::view.contiguous] n,
                    int32_t[:] hets):
        if not self.vcf.gts012:
            raise Exception("must call relatedness with gts012")
        if self._gt_types == NULL:
            self.gt_types
        cdef double[:] alt_freqs = self.gt_alt_freqs
        return krelated(<int32_t *>self._gt_types, &ibs[0, 0], &n[0, 0],
                &hets[0], self.vcf.n_samples, &alt_freqs[0])

    property num_called:
        "number of samples that were not UNKNOWN."
        def __get__(self):
            if self._gt_types == NULL:
                self.gt_types
            cdef int n = 0, i = 0
            if self.vcf.gts012:
                for i in range(self.vcf.n_samples):
                    if self._gt_types[i] != 3:
                        n+=1
            else:
                for i in range(self.vcf.n_samples):
                    if self._gt_types[i] != 2:
                        n+=1
            return n

    property call_rate:
        "proportion of samples that were not UNKNOWN."
        def __get__(self):
            if self.vcf.n_samples > 0:
                return float(self.num_called) / self.vcf.n_samples

    property aaf:
        "alternate allele frequency across samples in this VCF."
        def __get__(self):
            num_chroms = 2.0 * self.num_called
            if num_chroms == 0.0:
                return 0.0
            return float(self.num_het + 2 * self.num_hom_alt) / num_chroms

    property nucl_diversity:
        def __get__(self):
            num_chroms = 2.0 * self.num_called
            p = self.aaf
            return (num_chroms / (num_chroms - 1.0)) * 2 * p * (1 - p)

    property num_hom_ref:
        "number homozygous reference samples at this variant."
        def __get__(self):
            if self._gt_types == NULL:
                self.gt_types
            cdef int n = 0, i = 0
            for i in range(self.vcf.n_samples):
                if self._gt_types[i] == 0:
                    n+=1
            return n

    property num_het:
        "number heterozygous samples at this variant."
        def __get__(self):
            if self._gt_types == NULL:
                self.gt_types
            cdef int n = 0, i = 0
            for i in range(self.vcf.n_samples):
                if self._gt_types[i] == 1:
                    n+=1
            return n

    property num_hom_alt:
        "number homozygous alternate samples at this variant."
        def __get__(self):
            if self._gt_types == NULL:
                self.gt_types
            cdef int n = 0, i = 0
            if self.vcf.gts012:
                for i in range(self.vcf.n_samples):
                    if self._gt_types[i] == 2:
                        n+=1
            else:
                for i in range(self.vcf.n_samples):
                    if self._gt_types[i] == 3:
                        n+=1
            return n

    property num_unknown:
        "number unknown samples at this variant."
        def __get__(self):
            if self._gt_types == NULL:
                self.gt_types
            cdef int n = 0, i = 0
            cdef int unknown = 3 if self.vcf.gts012 else 2
            for i in range(self.vcf.n_samples):
                n += (self._gt_types[i] == unknown)
            return n

    property FORMAT:
        "VCF FORMAT field for this variant."
        def __get__(self):
            cdef int i
            cdef bcf_fmt_t fmt
            cdef char *key
            keys = []
            for i in range(self.b.n_fmt):
                fmt = self.b.d.fmt[i];
                key = bcf_hdr_int2id(self.vcf.hdr, BCF_DT_ID, fmt.id)
                keys.append(from_bytes(key))
            return keys

    def format(self, field, vtype=None):
        """format returns a numpy array for the requested field.

        The numpy array shape will match the requested field. E.g. if the fields
        has number=3, then the shape will be (n_samples, 3).

        Parameters
        ----------
        field: str
            FORMAT field to get the values.

        Returns
        -------
        numpy array.
        """
        if vtype is None:
            vtype = self.vcf.get_type(field)

        cdef bytes tag = to_bytes(field)
        cdef bcf_fmt_t *fmt = bcf_get_fmt(self.vcf.hdr, self.b, tag)
        cdef int n = 0, nret
        cdef void *buf = NULL;
        cdef int typenum = 0
        if vtype == "Integer" or vtype == int:
            nret = bcf_get_format_int32(self.vcf.hdr, self.b, tag, <int **>&buf, &n)
            typenum = np.NPY_INT32
        elif vtype == "Float" or vtype == float:
            nret = bcf_get_format_float(self.vcf.hdr, self.b, tag, <float **>&buf, &n)
            typenum = np.NPY_FLOAT32
        elif vtype == "String" or vtype == str or vtype == "Character":
            vtype = str
            nret = bcf_get_format_string(self.vcf.hdr, self.b, tag, <char ***>&buf, &n)
            typenum = np.NPY_STRING
        else:
            raise Exception("type %s not supported to format()" % vtype)
        if nret < 0:
            return None

        cdef char **dst
        cdef int i
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.vcf.n_samples
        shape[1] = fmt.n # values per sample

        if vtype == str:
            dst = <char **>buf
            v = [dst[i] for i in range(self.vcf.n_samples)]
            xret = np.array(v, dtype=str)
            stdlib.free(dst[0])
            stdlib.free(dst)
            return xret

        iv = np.PyArray_SimpleNewFromData(2, shape, typenum, buf)
        iret = np.array(iv)
        stdlib.free(buf)
        return iret

    @property
    def genotype(self):
        if self.vcf.n_samples == 0: return None
        cdef int32_t *gts = NULL
        cdef int ndst = 0
        cdef int nret = bcf_get_genotypes(self.vcf.hdr, self.b, &gts, &ndst)
        if nret < 0:
            raise Exception("couldn't get genotypes for variant")
        return newGenotypes(gts, int(nret/self.vcf.n_samples), self.vcf.n_samples)

    @genotype.setter
    def genotype(self, Genotypes g):
        cdef int ret = bcf_update_genotypes(self.vcf.hdr, self.b, g._raw, self.vcf.n_samples * g.ploidy)
        if ret < 0:
            raise Exception("error setting genotypes with: %s" % g)

    property genotypes:
        """genotypes returns a list for each sample Indicating the allele and phasing.

        e.g. [0, 1, True] corresponds to 0|1
        while [1, 2, False] corresponds to 1/2
        """
        def __get__(self):
            if self.vcf.n_samples == 0: return None
            if self._genotypes is not None:
              return self._genotypes
            cdef int32_t *gts = NULL
            cdef int i, j, nret, off = 0, ndst = 0, k = 0
            cdef int n_samples = self.vcf.n_samples
            #self._genotypes = []
            nret = bcf_get_genotypes(self.vcf.hdr, self.b, &gts, &ndst)
            if nret < 0:
                raise Exception("error parsing genotypes")
            nret /= n_samples
            self._genotypes = [[] for _ in range(n_samples)]

            for i in range(n_samples):
              k = i * nret
              for j in range(nret):
                  #assert k + j < ndst
                  if bcf_gt_is_missing(gts[k + j]):
                      self._genotypes[i].append(-1)
                      continue
                  if gts[k + j] == bcf_int32_vector_end:
                      break
                  self._genotypes[i].append(bcf_gt_allele(gts[k + j]))
              self._genotypes[i].append(
                    bool(bcf_gt_is_phased(gts[k+1 if k+1 < ndst else k])))

            stdlib.free(gts)
            return self._genotypes

        def __set__(self, gts):
            cdef int n_samples = self.vcf.n_samples
            if len(gts) != n_samples:
                raise Exception("genotypes: must set with a number of gts equal the number of samples in the vcf")
            elif len(gts) == 0:
                nret = 0
            else:
                nret = max(len(gt)-1 for gt in gts)
            cdef int * cgts = <int *>stdlib.malloc(sizeof(int) * nret * n_samples)
            cdef int i, j, k
            self._genotypes = None

            for i in range(n_samples):
                k = i * nret
                for j in range(nret):
                    if j == len(gts[i]) - 1:
                        cgts[k + j] = bcf_int32_vector_end #bcf_gt_phased(-1)
                        break
                    else:
                        cgts[k + j] = bcf_gt_phased(gts[i][j]) if gts[i][-1] else bcf_gt_unphased(gts[i][j])
            ret = bcf_update_genotypes(self.vcf.hdr, self.b, cgts, n_samples * nret)
            if ret < 0:
                raise Exception("error setting genotypes with: %s" % gts)
            stdlib.free(cgts)

    def set_pos(self, int pos0):
        """
        set the POS to the given 0-based position
        """
        self.b.pos = pos0
        self.POS = self.b.pos + 1

    def set_format(self, name, np.ndarray data not None):
        """
        set the format field given by name..
        data must be a numpy array of type float, int or string (fixed length ASCII np.bytes_)
        """
        cdef int n_samples = self.vcf.n_samples
        if len(data) % n_samples != 0:
            raise Exception("format: len(data) must be a multiple of number of samples in vcf.")

        cdef np.ndarray[np.float32_t, mode="c"] afloat
        cdef np.ndarray[np.int32_t, mode="c"] aint
        cdef char *bytesp
        cdef size_t i

        cdef int size
        cdef int ret
        if np.issubdtype(data.dtype, np.signedinteger) or np.issubdtype(data.dtype, np.unsignedinteger):
            size = data.shape[0]
            if len((<object>data).shape) > 1:
                size *= data.shape[1]
            aint = data.astype(np.int32).reshape((size,))
            ret = bcf_update_format_int32(self.vcf.hdr, self.b, to_bytes(name), &aint[0], size)
        elif np.issubdtype(data.dtype, np.floating):
            size = data.shape[0]
            if len((<object>data).shape) > 1:
                size *= data.shape[1]
            afloat = data.astype(np.float32).reshape((size,))
            isnan = np.isnan(afloat)
            for i in range(size):
                if isnan[i]:
                    bcf_float_set(&afloat[i], bcf_float_missing)
            ret = bcf_update_format_float(self.vcf.hdr, self.b, to_bytes(name), &afloat[0], size)
        elif np.issubdtype(data.dtype, np.bytes_):
            if len((<object>data).shape) > 1:
                raise Exception("Setting string type format fields with number>1 are currently not supported")
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)
            size = data.nbytes
            bytesp = <char *>data.data
            ret = bcf_update_format(self.vcf.hdr, self.b, to_bytes(name), bytesp, size, BCF_HT_STR)
        else:
            raise Exception("format: currently only float, int and string (fixed length ASCII np.bytes_) numpy arrays are supported. got %s", data.dtype)
        if ret < 0:
            raise Exception("error (%d) setting format with: %s" % (ret, data[:100]))

    property gt_types:
        """gt_types returns a numpy array indicating the type of each sample.

        HOM_REF=0, HET=1. For `gts012=True` HOM_ALT=2, UNKNOWN=3
        """
        def __get__(self):
            cdef int ndst = 0, ngts, n, i, nper, j = 0, k = 0
            cdef int a
            if self.vcf.n_samples == 0:
                return np.array([])
            if self._gt_types == NULL:
                self._gt_phased = <int *>stdlib.malloc(sizeof(int) * self.vcf.n_samples)
                ngts = bcf_get_genotypes(self.vcf.hdr, self.b, &self._gt_types, &ndst)
                if ngts < 0:
                    raise Exception("error parsing genotypes; they may be absent from this record")
                nper = int(ngts / self.vcf.n_samples)
                self._ploidy = nper
                self._gt_idxs = <int *>stdlib.malloc(sizeof(int) * self.vcf.n_samples * nper)
                if ngts == 0 or nper == 0:
                    return np.array([])
                for i in range(0, ngts, nper):
                    for k in range(i, i + nper):
                        a = self._gt_types[k]
                        if a >= 0:
                            self._gt_idxs[k] = bcf_gt_allele(a)
                        else:
                            self._gt_idxs[k] = a

                    self._gt_phased[j] = self._gt_types[i] > 0 and <int>bcf_gt_is_phased(self._gt_types[i+1])
                    j += 1

                if self.vcf.gts012:
                    n = as_gts(self._gt_types, self.vcf.n_samples, nper, self.vcf.strict_gt, 2, 3)
                else:
                    n = as_gts(self._gt_types, self.vcf.n_samples, nper, self.vcf.strict_gt, 3, 2)
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.vcf.n_samples
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, self._gt_types)

    property ploidy:
        """get the ploidy of each sample for the given record."""
        def __get__(self):
            if self._ploidy == -1:
                self.gt_types
            return self._ploidy

    property gt_phred_ll_homref:
        """get the PL of Hom ref for each sample as a numpy array."""
        def __get__(self):
            if self.vcf.n_samples == 0:
                return []
            cdef int ndst = 0, nret=0, n, i, j, nper

            cdef int imax = np.iinfo(np.int32(0)).max

            if self._gt_pls == NULL and self._gt_gls == NULL:
                nret = bcf_get_format_int32(self.vcf.hdr, self.b, "PL", &self._gt_pls, &ndst)
                if nret < 0:
                    nret = bcf_get_format_float(self.vcf.hdr, self.b, "GL", &self._gt_gls, &ndst)
                    if nret < 0:
                        return []
                    else:
                        for i in range(nret):
                            if self._gt_gls[i] <= -2147483646:
                                # this gets translated on conversion to PL
                                self._gt_gls[i] = imax / -10.0
                else:
                    for i in range(nret):
                        if self._gt_pls[i] < 0:
                            self._gt_pls[i] = imax

                self._gt_nper = int(nret / self.vcf.n_samples)
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self._gt_nper * self.vcf.n_samples
            if self._gt_pls != NULL:
                pls = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32,
                        self._gt_pls)[::self._gt_nper]
                return pls
            else:
                gls = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32,
                        self._gt_gls)[::self._gt_nper]
                gls = (-10 * gls).round().astype(np.int32)
                return gls

    property gt_phred_ll_het:
        """get the PL of het for each sample as a numpy array."""
        def __get__(self):
            if self.vcf.n_samples == 0:
                return []
            if self._gt_pls == NULL and self._gt_gls == NULL:
                # NOTE: the missing values for all homref, het, homalt are set
                # by this call.
                self.gt_phred_ll_homref
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self._gt_nper * self.vcf.n_samples
            if self._gt_pls != NULL:
                if self._gt_nper > 1:
                    ret = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, self._gt_pls)[1::self._gt_nper]
                    return ret

                return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, self._gt_pls)
            else:
                if self._gt_nper > 1:
                    gls = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32,
                            self._gt_gls)[1::self._gt_nper]
                else:
                    gls = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32, self._gt_gls)
                gls = (-10 * gls).round().astype(np.int32)
                return gls

    property gt_phred_ll_homalt:
        """get the PL of hom_alt for each sample as a numpy array."""
        def __get__(self):
            if self.vcf.n_samples == 0:
                return []
            if self._gt_pls == NULL and self._gt_gls == NULL:
                self.gt_phred_ll_homref
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self._gt_nper * self.vcf.n_samples
            if self._gt_pls != NULL:
                if self._gt_nper > 1:
                    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32,
                            self._gt_pls)[2::self._gt_nper]
                return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, self._gt_pls)
            else:
                if self._gt_nper > 1:
                    gls = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32,
                            self._gt_gls)[2::self._gt_nper]
                else:
                    gls = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32,
                            self._gt_gls)
                gls = (-10 * gls).round().astype(np.int32)
                return gls

    property gt_ref_depths:
        """get the count of reference reads for a diploid sample as a numpy array."""
        def __get__(self):
            cdef int ndst, nret = 0, n, i, j = 0, nper = 0
            if self.vcf.n_samples == 0:
                return []
            if self._gt_ref_depths == NULL:
                ndst = 0
                # GATK
                nret = bcf_get_format_int32(self.vcf.hdr, self.b, "AD", &self._gt_ref_depths, &ndst)
                if nret > 0:
                    nper = int(nret / self.vcf.n_samples)
                    if nper == 1:
                        stdlib.free(self._gt_ref_depths); self._gt_ref_depths = NULL
                        return -1 + np.zeros(self.vcf.n_samples, np.int32)

                    for i in range(0, nret, nper):
                        self._gt_ref_depths[j] = self._gt_ref_depths[i]
                        j += 1
                elif nret == -1:
                    # Freebayes
                    # RO has to be 1:1
                    nret = bcf_get_format_int32(self.vcf.hdr, self.b, "RO", &self._gt_ref_depths, &ndst)
                    if nret < 0:
                        stdlib.free(self._gt_ref_depths); self._gt_ref_depths = NULL
                        return -1 + np.zeros(self.vcf.n_samples, np.int32)
                # TODO: add new vcf standard.
                else:
                    stdlib.free(self._gt_ref_depths); self._gt_ref_depths = NULL
                    return -1 + np.zeros(self.vcf.n_samples, np.int32)

                for i in range(self.vcf.n_samples):
                    if self._gt_ref_depths[i] < 0:
                        self._gt_ref_depths[i] = -1
            else:
                pass

            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.vcf.n_samples
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, self._gt_ref_depths)

    property gt_alt_depths:
        """get the count of alternate reads for a diploid sample as a numpy array."""
        def __get__(self):
            cdef int ndst, nret = 0, n, i, j = 0, k = 0, nper = 0
            if self.vcf.n_samples == 0:
                return []
            if self._gt_alt_depths == NULL:
                ndst = 0
                # GATK
                nret = bcf_get_format_int32(self.vcf.hdr, self.b, "AD", &self._gt_alt_depths, &ndst)
                if nret > 0:
                    nper = int(nret / self.vcf.n_samples)
                    if nper == 1:
                        stdlib.free(self._gt_alt_depths); self._gt_alt_depths = NULL
                        return (-1 + np.zeros(self.vcf.n_samples, np.int32))

                    for i in range(0, nret, nper):
                        self._gt_alt_depths[j] = self._gt_alt_depths[i+1]
                        # add up all the alt alleles
                        for k in range(2, nper):
                            self._gt_alt_depths[j] += self._gt_alt_depths[i+k]
                        j += 1

                elif nret == -1:
                    # Freebayes
                    nret = bcf_get_format_int32(self.vcf.hdr, self.b, "AO", &self._gt_alt_depths, &ndst)
                    nper = int(nret / self.vcf.n_samples)
                    if nret < 0:
                        stdlib.free(self._gt_alt_depths); self._gt_alt_depths = NULL
                        return -1 + np.zeros(self.vcf.n_samples, np.int32)
                    for i in range(0, nret, nper):
                        self._gt_alt_depths[j] = self._gt_alt_depths[i]
                        for k in range(1, nper):
                            self._gt_alt_depths[j] += self._gt_alt_depths[i+k]
                        j += 1
                else:
                    stdlib.free(self._gt_alt_depths); self._gt_alt_depths = NULL
                    return -1 + np.zeros(self.vcf.n_samples, np.int32)

                # TODO: add new vcf standard.
            for i in range(self.vcf.n_samples):
                if self._gt_alt_depths[i] < 0:
                    self._gt_alt_depths[i] = -1

            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.vcf.n_samples
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, self._gt_alt_depths)

    property gt_alt_freqs:
        """get the freq of alternate reads for a diploid sample as a numpy array."""
        def __get__(self):
            if self.vcf.n_samples == 0:
                return []
            t = np.array(self.gt_depths, float)
            a = np.array(self.gt_alt_depths, float)

            # for which samples are the alt or total depths unknown?
            tU = t < 0
            aU = a < 0
            # for which samples is the total depth 0?
            t0 = t == 0

            ## initialize
            alt_freq = t.astype(float)

            # default
            alt_freq[t0] = 0
            alt_freq[aU] = 0
            alt_freq[tU] = -1

            # compute the alt_freq when not unknown and no div0 error
            clean = ~tU & ~aU & ~t0
            alt_freq[clean] = (a[clean] / t[clean])

            return alt_freq

    property gt_quals:
        """get the GQ for each sample as a numpy array."""
        def __get__(self):
            if self.vcf.n_samples == 0:
                return []
            cdef int ndst = 0, nret, n, i
            cdef int *gq
            cdef np.ndarray[np.float32_t, ndim=1] a
            if self._gt_quals == NULL and self._int_gt_quals == NULL:
                nret = bcf_get_format_int32(self.vcf.hdr, self.b, "GQ", &self._int_gt_quals, &ndst)
                if nret == -2: # defined as int
                    ndst = 0
                    nret = bcf_get_format_float(self.vcf.hdr, self.b, "GQ", &self._gt_quals, &ndst)
                if nret < 0 and nret != -2:
                    return -1.0 + np.zeros(self.vcf.n_samples, np.float32)
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.vcf.n_samples
            if self._int_gt_quals != NULL:
                a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, self._int_gt_quals).astype(np.float32)
                a[a < 0] = -1
            else:
                a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32, self._gt_quals)
                # this take up 10% of the total vcf parsing time. fix!!
                a[np.isnan(a)] = -1
            return a

    property gt_depths:
        """get the read-depth for diploid samples as a numpy array."""
        def __get__(self):
            if self.vcf.n_samples == 0:
                return []
            # unfortunately need to create a new array here since we're modifying.
            r = np.array(self.gt_ref_depths, np.int32)
            a = np.array(self.gt_alt_depths, np.int32)
            # keep the -1 for empty.
            rl0 = r < 0
            al0 = a < 0
            r[rl0] = 0
            a[al0] = 0
            depth = r + a
            depth[rl0 & al0] = -1
            return depth

    property gt_phases:
        """get a boolean indicating whether each sample is phased as a numpy array."""
        def __get__(self):
            # run for side-effect
            if self._gt_phased == NULL:
                self.gt_types
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.vcf.n_samples

            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, self._gt_phased).astype(bool)

    property REF:
        "the reference allele."
        def __get__(self):
            return self.b.d.allele[0].decode()

        def __set__(self, ref):
          # int bcf_update_alleles_str(const bcf_hdr_t *hdr, bcf1_t *line, const char *alleles_string);
          alleles = (ref + "," + ",".join(self.ALT)).encode()
          if bcf_update_alleles_str(self.vcf.hdr, self.b, alleles) != 0:
            raise ValueError("couldn't set reference to:" + str(ref))

    property ALT:
        "the list of alternate alleles."
        def __get__(self):
            cdef int i
            return [self.b.d.allele[i].decode() for i in range(1, self.b.n_allele)]

        def __set__(self, alts):
          # int bcf_update_alleles_str(const bcf_hdr_t *hdr, bcf1_t *line, const char *alleles_string);
          if not isinstance(alts, list):
            alts = [alts]
          alleles = (self.REF + "," + ",".join(alts)).encode()
          if bcf_update_alleles_str(self.vcf.hdr, self.b, alleles) != 0:
            raise ValueError("couldn't set alternates to:" + str(alts))

    property is_snp:
        "boolean indicating if the variant is a SNP."
        def __get__(self):
            cdef int i
            if len(self.b.d.allele[0]) > 1: return False
            for i in range(1, self.b.n_allele):
                if not self.b.d.allele[i] in (b"A", b"C", b"G", b"T"):
                    return False
            return self.b.n_allele > 1

    property is_mnp:
        "boolean indicating if the variant is a MNP."
        def __get__(self):
            cdef int i
            is_sv = self.is_sv
            if len(self.b.d.allele[0]) == 1 and not is_sv: return False

            if len(self.REF) == 1 and not is_sv: return False

            if all([len(x)==len(self.REF) for x in self.ALT]):
                if not is_sv:
                    return True
            return False

    property is_indel:
        "boolean indicating if the variant is an indel."
        def __get__(self):
            cdef int i
            if self.is_sv: return False
            if len(self.b.d.allele[0]) > 1 and self.b.d.allele[0] != b'<': return True

            if len(self.REF) > 1: return True

            for i in range(1, self.b.n_allele):
                alt = self.b.d.allele[i]
                if alt[0] == b'<': continue
                if alt == b".":
                    return True
                if len(alt) > 1:
                    return True
            return False

    property is_transition:
        "boolean indicating if the variant is a transition."
        def __get__(self):
            if len(self.ALT) != 1: return False

            if not self.is_snp: return False
            ref = self.REF
                # just one alt allele
            alt_allele = self.ALT[0]
            if ((ref == 'A' and alt_allele == 'G') or
                (ref == 'G' and alt_allele == 'A') or
                (ref == 'C' and alt_allele == 'T') or
                (ref == 'T' and alt_allele == 'C')):
                    return True
            return False

    property is_deletion:
        "boolean indicating if the variant is a deletion."
        def __get__(self):
            if len(self.ALT) > 1: return False

            if not self.is_indel: return False
            if len(self.ALT) == 0:
                return True
            alt = self.ALT[0]
            if alt is None or alt == ".":
                return True

            if len(self.REF) > len(alt):
                return True
            return False

    property is_sv:
        "boolean indicating if the variant is an SV."
        def __get__(self):
            return self.INFO.get(b'SVTYPE') is not None

    property CHROM:
        """Chromosome of the variant."""
        def __get__(self):
            return bcf_hdr_id2name(self.vcf.hdr, self.b.rid).decode()

        def __set__(self, new_chrom):
            new_rid = bcf_hdr_id2int(self.vcf.hdr, BCF_DT_CTG, new_chrom.encode())
            if new_rid < 0:
                self.vcf.add_to_header("##contig=<ID={}>".format(new_chrom))
                new_rid = bcf_hdr_id2int(self.vcf.hdr, BCF_DT_CTG, new_chrom.encode())
                if new_rid < 0:
                    raise ValueError("Unable to add {} to CHROM".format(new_chrom))
                warnings.warn("added new contig {} to header".format(new_chrom))
            self.b.rid = new_rid

    property var_type:
        "type of variant (snp/indel/sv)"
        def __get__(self):
           if self.is_snp:
               return "snp"
           if self.is_mnp:
               return "mnp"
           elif self.is_indel:
               return "indel"
           elif self.is_sv:
               return "sv"
           else:
               return "unknown"

    property var_subtype:
        def __get__(self):
            if self.is_snp:
                if self.is_transition:
                    return "ts"
                if len(self.ALT) == 1:
                    return "tv"
                return "unknown"

            elif self.is_indel:
                if self.is_deletion:
                    return "del"
                if len(self.ALT) == 1:
                    return "ins"
                else:
                    return "unknown"

            svt = self.INFO.get("SVTYPE")
            if svt is None:
                return "unknown"
            if svt == "BND":
                return "complex"
            if self.INFO.get('IMPRECISE') is None:
                return svt
            return self.ALT[0].strip('<>')

    property start:
        "0-based start of the variant."
        def __get__(self):
            return self.b.pos

    property end:
        "end of the variant. the INFO field is parsed for SVs."
        def __get__(self):
            return self.b.pos + self.b.rlen

    property ID:
        "the value of ID from the VCF field."
        def __get__(self):
            cdef char *id = self.b.d.id
            if id == b".": return None
            return from_bytes(id)

        def __set__(self, value):
            sanitized = str(value) if value is not None else '.'
            ret = bcf_update_id(self.vcf.hdr, self.b, to_bytes(sanitized))
            if ret != 0:
                raise Exception("not able to set ID: %s", value)

    property FILTER:
        """the value of FILTER from the VCF field.

        a value of PASS or '.' in the VCF will give None for this function
        """
        def __get__(self):
            cdef int i
            cdef int n = self.b.d.n_flt
            if n == 1:
                if self.vcf.PASS != -1:
                    if self.b.d.flt[0] == self.vcf.PASS:
                        return None
                else:
                    v = bcf_hdr_int2id(self.vcf.hdr, BCF_DT_ID, self.b.d.flt[0])
                    if v == b"PASS":
                        self.vcf.PASS = self.b.d.flt[0]
                        return None
                    return from_bytes(v)
            if n == 0:
                return None
            return from_bytes(b';'.join(bcf_hdr_int2id(self.vcf.hdr, BCF_DT_ID, self.b.d.flt[i]) for i in range(n)))

        def __set__(self, filters):
            if isinstance(filters, basestring):
                filters = filters.split(";")
            cdef bcf_hdr_t *h = self.vcf.hdr
            cdef int *flt_ids = <int *>stdlib.malloc(sizeof(int) * len(filters))
            for i, fname in enumerate(filters):
                flt_ids[i] = bcf_hdr_id2int(h, BCF_DT_ID, to_bytes(fname))
            ret = bcf_update_filter(h, self.b, flt_ids, len(filters))
            stdlib.free(flt_ids)
            if ret != 0:
                raise Exception("not able to set filter: %s", filters)

    property FILTERS:
        """the FILTER values as a list from the VCF field.

        a value '.' in the VCF will return an empty list for this property
        """
        def __get__(self):
            cdef int i
            cdef int n = self.b.d.n_flt
            return [from_bytes(bcf_hdr_int2id(self.vcf.hdr, BCF_DT_ID, self.b.d.flt[i])) for i in range(n)]

    property QUAL:
        "the float value of QUAL from the VCF field."
        def __get__(self):
            cdef float q = self.b.qual
            if bcf_float_is_missing(q):
                return None
            return q

        def __set__(self, value):
            if value is None:
                bcf_float_set(&self.b.qual, bcf_float_missing)
            else:
                self.b.qual = value


cdef inline HREC newHREC(bcf_hrec_t *hrec, bcf_hdr_t *hdr):
    cdef HREC h = HREC.__new__(HREC)
    h.hdr = hdr
    h.hrec = hrec
    return h


cdef class HREC(object):
    cdef bcf_hdr_t *hdr
    cdef bcf_hrec_t *hrec

    def __cinit__(HREC self):
        pass

    def __dealloc__(self):
        #bcf_hrec_destroy(self.hrec)
        self.hrec = NULL
        self.hdr = NULL

    @property
    def type(self):
        return ["FILTER", "INFO", "FORMAT", "CONTIG", "STR", "GENERIC"][self.hrec.type]

    def __getitem__(self, key):
        key = from_bytes(key)
        if key == "HeaderType":
            return self.type
        cdef int i
        for i in range(self.hrec.nkeys):
            if from_bytes(self.hrec.keys[i]) == key:
                return from_bytes(self.hrec.vals[i])
        raise KeyError

    def info(self, extra=False):
        """
        return a dict with commonly used stuffs
        """
        d = {}
        for k in ('Type', 'Number', 'ID', 'Description'):
            try:
                d[k] = self[k]
            except KeyError:
                continue
        d['HeaderType'] = self.type
        if extra:
            for i in range(self.hrec.nkeys):
                k = self.hrec.keys[i]
                if k in d: continue
                d[k] = self.hrec.vals[i]
        return d

    def __repr__(self):
        return str(self.info())


cdef class INFO(object):
    """
    INFO is create internally by accessing `Variant.INFO`

    is acts like a dictionary where keys are expected to be in the INFO field of the Variant
    and values are typed according to what is specified in the VCF header

    Items can be deleted with del v.INFO[key] and accessed with v.INFO[key] or v.INFO.get(key)
    """
    cdef bcf_hdr_t *hdr
    cdef bcf1_t *b
    cdef int _i

    def __cinit__(INFO self):
        self._i = 0

    def __delitem__(self, okey):
        okey = to_bytes(okey)
        cdef char *key = okey
        cdef bcf_info_t *info = bcf_get_info(self.hdr, self.b, key)
        if info == NULL:
            raise KeyError(key)
        cdef int htype = bcf_hdr_id2type(self.hdr, BCF_HL_INFO, info.key)
        cdef int ret = bcf_update_info(self.hdr, self.b, key,NULL,0,htype)
        if ret != 0:
            raise Exception("error deleting %s" % key)

    def __setitem__(self, key, value):
        # only support strings for now.
        if value is True or value is False:

            ret = bcf_update_info_flag(self.hdr, self.b, to_bytes(key), b"", int(value))
            if ret != 0:
                raise Exception("not able to set: %s -> %s (%d)" % (key, value, ret))

            return
        cdef int32_t iint
        cdef float ifloat
        if isinstance(value, int):
            iint = value
            ret = bcf_update_info_int32(self.hdr, self.b, to_bytes(key), &iint, 1)
        elif isinstance(value, float):
            ifloat = value
            ret = bcf_update_info_float(self.hdr, self.b, to_bytes(key), &ifloat, 1)
        else:
            ret = bcf_update_info_string(self.hdr, self.b, to_bytes(key), to_bytes(value))
        if ret != 0:
            raise Exception("not able to set: %s -> %s (%d)", key, value, ret)

    cdef _getval(INFO self, bcf_info_t * info, char *key):

        if info.len == 1:
            if info.type == BCF_BT_INT8:
                if info.v1.i == INT8_MIN:
                    return None
                return <int>(info.v1.i)

            if info.type == BCF_BT_INT16:
                if info.v1.i == INT16_MIN:
                    return None
                return <int>(info.v1.i)

            if info.type == BCF_BT_INT32:
                if info.v1.i == INT32_MIN:
                    return None
                return <int>(info.v1.i)

            if info.type == BCF_BT_FLOAT:
                if bcf_float_is_missing(info.v1.f):
                    return None
                return info.v1.f

        if info.type == BCF_BT_CHAR:
            v = info.vptr[:info.vptr_len]
            if len(v) > 0 and v[0] == 0x7:
                return None
            return from_bytes(v)

        # FLAG.
        if info.len == 0:
            return bcf_hdr_id2type(self.hdr, BCF_HL_INFO, info.key) == BCF_HT_FLAG

        return bcf_array_to_object(info.vptr, info.type, info.len)

    def __getitem__(self, okey):
        okey = to_bytes(okey)
        cdef char *key = okey
        cdef bcf_info_t *info = bcf_get_info(self.hdr, self.b, key)
        if info == NULL:
            raise KeyError(key)
        return self._getval(info, key)

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        cdef bcf_info_t *info = NULL
        cdef char *name
        while info == NULL:
            if self._i >= self.b.n_info:
                raise StopIteration
            info = &(self.b.d.info[self._i])
            self._i += 1
        name = bcf_hdr_int2id(self.hdr, BCF_DT_ID, info.key)
        return name.decode(), self._getval(info, name)


# this function is copied verbatim from pysam/cbcf.pyx
cdef bcf_array_to_object(void *data, int type, int n, int scalar=0):
    cdef char    *datac
    cdef int8_t  *data8
    cdef int16_t *data16
    cdef int32_t *data32
    cdef float   *dataf
    cdef int      i

    if not data or n <= 0:
        return None

    if type == BCF_BT_CHAR:
        datac = <char *>data
        value = datac[:n].decode() if datac[0] != bcf_str_missing else None
    else:
        value = []
        if type == BCF_BT_INT8:
            data8 = <int8_t *>data
            for i in range(n):
                if data8[i] == bcf_int8_vector_end:
                    break
                value.append(data8[i] if data8[i] != bcf_int8_missing else None)
        elif type == BCF_BT_INT16:
            data16 = <int16_t *>data
            for i in range(n):
                if data16[i] == bcf_int16_vector_end:
                    break
                value.append(data16[i] if data16[i] != bcf_int16_missing else None)
        elif type == BCF_BT_INT32:
            data32 = <int32_t *>data
            for i in range(n):
                if data32[i] == bcf_int32_vector_end:
                    break
                value.append(data32[i] if data32[i] != bcf_int32_missing else None)
        elif type == BCF_BT_FLOAT:
            dataf = <float *>data
            for i in range(n):
                if bcf_float_is_vector_end(dataf[i]):
                    break
                value.append(dataf[i] if not bcf_float_is_missing(dataf[i]) else None)
        else:
            raise TypeError('unsupported info type code')

        if not value:
            value = None
        elif scalar and len(value) == 1:
            value = value[0]
        else:
            value = tuple(value)

    return value


cdef inline Variant newVariant(bcf1_t *b, VCF vcf):
    cdef Variant v = Variant.__new__(Variant)
    v.b = b
    if not vcf.lazy:
        with nogil:
            bcf_unpack(v.b, 15)
    else:
        with nogil:
            bcf_unpack(v.b, 1|2|4)

    v.vcf = vcf
    v.POS = v.b.pos + 1
    cdef INFO i = INFO.__new__(INFO)
    i.b = b
    i.hdr = vcf.hdr
    v.INFO = i
    return v


cdef to_bytes(s, enc=ENC):
    if not isinstance(s, bytes):
        return s.encode(enc)
    return s


cdef from_bytes(s):
    if isinstance(s, bytes):
        try:
            return s.decode(ENC)
        except UnicodeDecodeError:
            return s.decode('utf8', errors='replace')
    return s


# TODO: make Writer extend HTSFile not VCF by moving common methods into HTSFile
cdef class Writer(VCF):
    """
    Writer class makes a VCF Writer.

    Parameters
    ----------
    fname: str
        path to file
    tmpl: VCF
        a template to use to create the output header.
    mode: str
        | Mode to use for writing the file. If ``None`` (default) is given, the mode is
          inferred from the filename extension. If stdout (``"-"``) is provided for ``fname``
          and ``mode`` is left at default, uncompressed VCF will be produced.
        | Valid values are:
        |  - ``"wbu"``: uncompressed BCF
        |  - ``"wb"``: compressed BCF
        |  - ``"wz"``: compressed VCF
        |  - ``"w"``: uncompressed VCF
        | Compression level can also be indicated by adding a single integer to one of
          the compressed modes (e.g. ``"wz4"`` for VCF with compressions level 4).

    Note
    ----
    File extensions ``.bcf`` and ``.bcf.gz`` will both return compressed BCF. If you
    want uncompressed BCF you must explicitly provide the appropriate ``mode``.

    Returns
    -------
    VCF object for iterating and querying.
    """
    #cdef htsFile *hts
    #cdef bcf_hdr_t *hdr
    cdef public bytes name
    cdef bint header_written
    cdef const bcf_hdr_t *ohdr

    def __init__(Writer self, fname, VCF tmpl, mode=None):
        mode = self._infer_file_mode(fname, mode)
        self._open_htsfile(fname, mode)
        bcf_hdr_sync(tmpl.hdr)
        self.ohdr = tmpl.hdr
        self.hdr = bcf_hdr_dup(tmpl.hdr)
        bcf_hdr_sync(self.hdr)
        self.header_written = False

    @staticmethod
    def _infer_file_mode(fname, mode=None):
        if mode is not None:
            return mode

        from_path = isinstance(fname, (basestring, Path))
        if not from_path:
            return "w"

        fname = str(fname)
        is_compressed = fname.endswith(".gz")
        fmt_idx = -2 if is_compressed else -1
        file_fmt = fname.split(".")[fmt_idx]
        # bcftools output write mode chars - https://github.com/samtools/bcftools/blob/76392b3014de70b7fa5c6b5c9d5bc47361951770/version.c#L64-L70
        inferred_mode = "w"
        if file_fmt == "bcf":
            inferred_mode += "b"
        if is_compressed and file_fmt == "vcf":
            inferred_mode += "z"

        return inferred_mode

    @classmethod
    def from_string(Writer cls, fname, header_string, mode="w"):
        cdef Writer self = Writer.__new__(Writer)
        cdef char *hmode = "w"
        self._open_htsfile(fname, mode)
        self.hdr = bcf_hdr_init(hmode)
        self.ohdr = bcf_hdr_dup(self.hdr)
        if bcf_hdr_parse(self.hdr, to_bytes(header_string)) != 0:
            raise Exception("error parsing header:" + header_string)
        if bcf_hdr_sync(self.hdr) != 0:
            raise Exception("error syncing header:" + header_string)
        self.header_written = False
        self.n_samples = bcf_hdr_nsamples(self.hdr)
        return self

    def variant_from_string(self, variant_string):
        cdef bcf1_t *b = bcf_init()
        cdef kstring_t s

        # vcf_parse may realloc s.s, so the memory must be created by malloc
        s.s = pystring_to_cstring(variant_string)
        if s.s == NULL:
            bcf_destroy(b)
            raise MemoryError("Failed to allocate memory")

        # plus 1 for '\0' of cstring
        s.l = len(variant_string) + 1
        s.m = len(variant_string) + 1

        ret = vcf_parse(&s, self.hdr, b)
        if ret != 0:
            bcf_destroy(b)
            ks_free(&s)
            raise Exception("error parsing: %s (vcf_parse returned %d)" % (variant_string, ret))

        var = newVariant(b, self)
        if var.b.errcode == BCF_ERR_CTG_UNDEF:
            self.add_to_header("##contig=<ID=%s>" % var.CHROM)
            var.b.errcode = 0

        ks_free(&s)
        return var

    def write_header(Writer self):
        bcf_hdr_write(self.hts, self.hdr)
        self.header_written = True

    def write_record(Writer self, Variant var):
        "Write the variant to the writer."
        cdef bcf_hrec_t *h
        if not self.header_written:
            self.write_header()
        if var.b.errcode == BCF_ERR_CTG_UNDEF:
            h = bcf_hdr_id2hrec(bcf_hdr_dup(self.ohdr), BCF_DT_CTG, 0, var.b.rid)
            if h == NULL:
                raise Exception("contig %d unknown and not found in header" % var.b.rid)
            if bcf_hdr_add_hrec(self.hdr, h) < 0:
                raise Exception("error adding contig %d to header" % var.b.rid)
            bcf_hdr_sync(self.hdr)
        elif var.b.errcode != 0:
            raise Exception("variant to be written has errorcode: %d" % var.b.errcode)
        return bcf_write(self.hts, self.hdr, var.b)

    def close(Writer self):
        if not self.header_written:
            self.write_header()
        super().close()
