.. cyvcf2 documentation master file, created by
   sphinx-quickstart on Mon Nov 14 08:40:54 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cyvcf2
======

cyvcf2 is a fast VCF parser for python (2 and 3) released under the MIT license.
It is a `cython` wrapper for `htslib <https://github.com/samtools/htslib/>`_ developed in 
the `Quinlan lab <http://quinlanlab.org/>`_.

See the :ref:`api` for detailed documentation, but the most common usage is summarized in the snippet below:

.. code-block:: python

    from cyvcf2 import VCF

    for variant in VCF('some.vcf.gz'): # or VCF('some.bcf')
        variant.REF, variant.ALT # e.g. REF='A', ALT=['C', 'T']

        variant.CHROM, variant.start, variant.end, variant.ID, \
                    variant.FILTER, variant.QUAL

        # numpy arrays of specific things we pull from the sample fields.
        # gt_types is array of 0,1,2,3==HOM_REF, HET, UNKNOWN, HOM_ALT
        variant.gt_types, variant.gt_ref_depths, variant.gt_alt_depths # numpy arrays
        variant.gt_phases, variant.gt_quals, variant.gt_bases # numpy array


        ## INFO Field.
        ## extract from the info field by it's name:
        variant.INFO.get('DP') # int
        variant.INFO.get('FS') # float
        variant.INFO.get('AC') # float

        # convert back to a string.
        str(variant)


        ## per-sample info...
        # Get a numpy array of the depth per sample:
        dp = variant.format('DP')
        # or of any other format field:
        sb = variant.format('SB')
        assert sb.shape == (n_samples, 4) # 4-values per

    # to do a region-query:

    vcf = VCF('some.vcf.gz')
    for v in vcf('11:435345-556565'):
        if v.INFO["AF"] > 0.1: continue
        print(str(v))

    # to query "all records" via __call__:
    # this uses the index (HTS_IDX_START), so an index is required.
    # if no index is available, this yields zero records.
    all_vars = list(vcf())

        # single sample of 0|1 in vcf becomes [[0, 1, True]]
        # 2 samples of 0/0 and 1|1 would be [[0, 0, False], [1, 1, True]]
        print v.genotypes 

Schema scanning and index creation
==================================

For index creation alone, use ``build_index`` instead of ``schema_scan``:

.. code-block:: python

    from cyvcf2 import build_index

    index_path = build_index("variants.vcf.gz", threads=1)
    build_index("variants.vcf.gz", index_path="variants.vcf.gz.tbi", threads=4)
    build_index("variants.vcf.gz", index_path="parallel.csi", workers=4, threads=1)

This calls HTSlib's native index builder with the GIL released. VCF uses the
same coordinate/interval parser as tabix, without decoding sample genotypes or
collecting schema statistics. BCF uses HTSlib's BCF index builder. Inputs must
be coordinate-sorted, BGZF-compressed VCF/BCF files. Local input defaults to
``path + ".csi"``; explicit destinations must end in ``.csi`` or ``.tbi``.
BCF supports CSI only. CSI uses a minimum bin size of 16,384 bases; TBI has
the usual 512 Mi-base coordinate limit.

``workers`` is a positive integer; the default, ``1``, keeps the direct native
builder. With ``workers>1``, native parsing threads read disjoint VCF ranges,
calling the same HTSlib tabix coordinate parser without genotype/statistics
decoding. Coordinate batches are fed in file order to one native index builder,
preserving contig ordering, interval overlaps, record counts, and offsets.
Record-boundary handoffs are checked before an index can be published.

Parallel coordinate indexing currently supports BGZF VCF and bundled HTSlib
only: tabix's line parser is an internal interface, not an external-library ABI.
BCF and external HTSlib builds retain the one-worker path. Parallel indexing
reads the VCF header once; it is not a headerless generic tabix interface.

``threads`` is a non-negative integer specifying HTSlib decompression threads
**per worker**; the default, ``0``, disables extra decompression threads, not
parsing workers (unlike ``schema_scan(threads=0)``). Positive values also create
HTSlib I/O helper threads. Worker and decompression counts are independent and
explicit; there is no automatic CPU-budget selection in ``build_index``.

Scheduling uses roughly 4 MiB compressed ranges and at most one outstanding
native batch per worker, each capped at 65,536 coordinate entries (2 MiB).
Split discovery reads small windows near compressed targets, not a whole-file
BGZF-offset list. There is no per-record temporary stream. Memory includes the
final HTSlib index, bounded coordinate batches, per-worker decompression buffers,
contig maps, and reusable line buffers; a single exceptionally long VCF record
can still require a large buffer. This is not a hard RSS cap. Chunk scheduling
is lazy, so neither the number of chunks nor their offsets are retained for the
entire input. Random-access split candidates are CRC-checked and adjacent native
readers must agree exactly on the resulting record boundary; a mismatch fails
explicitly instead of publishing a gapped or duplicated index.

The returned value is the destination path as a filesystem string.
Existing destinations, even corrupt indexes or dangling symlinks, raise
``FileExistsError`` rather than being reused or overwritten. A same-directory
temporary file and atomic hard link also prevent overwriting a destination
created concurrently. Native indexing and publication failures raise exceptions;
temporary files are cleaned up.

**Indexing is not full VCF validation.** Like tabix, HTSlib's VCF index parser
can log and skip unparsable text lines; a successful index does not certify
every record, sample genotype, or schema field. Use ``schema_scan`` when full
schema and genotype summaries are required. Its behavior is unchanged.

S3 and HTTP input
-----------------

Remote inputs are read directly through HTSlib, without downloading a local
copy of the VCF/BCF. Supply an explicit **local** index destination:

.. code-block:: python

    build_index(
        "s3://bucket/variants.vcf.gz", index_path="variants.csi",
        workers=8, threads=1,
    )

Parallel splitting uses HTSlib range reads and requires a seekable server
with a known object size. Network latency, request counts and available
bandwidth can limit scaling; the local-file benchmark is not an S3 performance
claim. HTSlib must have its S3/libcurl backend enabled and uses its own AWS
credential configuration. For S3-compatible endpoints, it also supports
``HTS_S3_HOST``, ``HTS_S3_ADDRESS_STYLE`` and ``s3+http://`` / ``s3+https://``.
URLs containing embedded credentials, query strings or fragments are rejected;
configure credentials outside the URL. The source object must remain unchanged
during indexing; cyvcf2 does not pin all worker reads to an object version.

The core package has **no AWS SDK, UPath or fsspec dependency**. For S3 output,
the separate ``scripts/build_index_s3.py`` adapter uses optional UPath/s3fs
tooling installed only in the environment running that script:

.. code-block:: bash

    uv run --with 'universal_pathlib>=0.2.6' --with 's3fs>=2026.9.0' \
        python scripts/build_index_s3.py \
        s3://bucket/variants.vcf.gz s3://bucket/variants.vcf.gz.csi \
        --workers 8 --threads 1

The adapter streams the index through a pipe into an s3fs transactional
multipart upload, with **no local input or index staging file**. It completes
the upload only after indexing succeeds, using conditional creation to prevent
overwriting an existing or concurrently created object. This requires an S3
provider implementing conditional multipart completion; there is no
unconditional-upload fallback. A failed operation attempts to abort the
multipart upload; a provider-side lifecycle policy can clean up uploads if
the process or network fails before the abort.

The streaming adapter currently supports BGZF VCF, not BCF, and requires
bundled HTSlib and POSIX ``/dev/fd``. It reads the source twice: once for
contig metadata and once for the index, so the source must remain unchanged
through both passes. Coordinate workers operate on bounded row batches. Index
construction retains at most one chunk per genomic bin for one contig at a
time, not the full index; at most 37,449 bins plus 32,768 seek offsets
(about 841 KiB in total) are allocated.
Contig names are limited to 100,000 and 16 MiB of name bytes. Memory also
includes the header, longest record, worker batches and decompression buffers,
Python objects and the S3 multipart buffer (at least 5 MiB); it is not bounded
independently of worker count or record/header size.

The streamed index can be less selective than HTSlib's standard index: CSI
uses coarser bins when necessary to bound memory, and each bin's records are
covered by a single range. Some region queries may read more data. The
regular local ``build_index`` path still uses HTSlib's in-memory index and
local atomic staging; use the adapter for bounded streaming to S3. HTSlib's
bundled S3 writer does not provide this conditional publication guarantee, so
the core API does not silently route S3 output through it. The adapter uses the
filesystem backend's credentials separately from HTSlib's read credentials.
For custom endpoints, configure both backends (e.g. HTSlib's host settings and
the filesystem's ``AWS_ENDPOINT_URL_S3``). Core local-destination publication
continues to use the same atomic hard-link mechanism.

Full schema scans
-----------------

``VCF.schema_scan()`` reads the complete file through HTSlib and returns
record, allele, genotype-ploidy, INFO, and FORMAT summaries without creating
Python objects for individual records. It scans an independent input stream,
so the reader's current iteration position is preserved.

.. code-block:: python

    vcf = VCF("variants.vcf.gz")
    summary = vcf.schema_scan(threads=4)
    summary["fields"]["INFO"]["DP"]["numeric_bounds"]

Each header-declared field is included in ``fields``, even if it is absent
from all records. Its summary includes VCF ``Type`` and ``Number``, the number
of records containing the field, total value cardinality, minimum and maximum
cardinality, and numeric bounds when available. INFO cardinality is per
record; FORMAT cardinality is the encoded width per sample. GT ploidy is
summarized separately. ``allele_counts["by_index"]`` contains called
genotype-allele counts by zero-based VCF allele index, with missing genotype
alleles reported separately.

String and Character fields also include ``logical_cardinality`` with observed
``min`` and ``max`` element counts: per record for INFO, per sample for FORMAT.
This follows the VCF comma-separated value syntax, not the byte width or the
header-declared ``Number``. For example, samples ``longword`` and ``x,y`` give
logical cardinality ``{"min": 1, "max": 2}``, while their encoded FORMAT
cardinality remains ``{"min": 9, "max": 9}``. Scalar strings count as one
element; missing placeholders count as slots (``.`` is one, ``.,.`` is two).
Absent fields have ``None`` extrema. These are not counts of nonmissing values.

Delimiters are counted before percent decoding, so ``a%2Cb`` is one element,
not two; UTF-8 multibyte characters do not change element counts. BCF string
NUL terminators/padding are excluded. A zero-length encoded string has zero
elements. GT uses its special allele-vector encoding and does not have
``logical_cardinality``. Statistics are accumulated during the same native
scan and merged across workers; no extra input pass or Python list allocation
is needed. See the `VCF specification <https://samtools.github.io/hts-specs/VCFv4.3.pdf>`_
for value delimiters and percent escaping.

Pass ``index_path`` to reuse an existing index at that path or create one
during the scan when it does not exist. Index creation supports CSI for VCF or
BCF and TBI for BGZF-compressed VCF. It requires coordinate-sorted input.
Without ``index_path``, an existing default index is detected but no index is
created.

Existing local indexes are read-only: an unreadable, truncated, incompatible,
or detectably stale index raises an error rather than being ignored or rebuilt.
To replace one, remove or rebuild it explicitly outside ``schema_scan``.
New indexes are written to a temporary file and published without replacement;
a destination created concurrently causes ``FileExistsError``. Index creation
requires a local destination so that no-overwrite publication can be guaranteed.

For a local BGZF-compressed VCF or BCF, pass ``workers`` to scan independent
BGZF block ranges in parallel. Each worker returns compact summaries and, when an
index is requested, streams its record offsets to a temporary file. The
offsets are fed in input order to one in-memory HTSlib index builder; the
finished CSI/TBI is written after all ranges are processed. ``threads``
continues to control HTSlib decompression threads per scan.

When an existing local CSI/TBI is available, its chunk start/end virtual
offsets supply split hints, avoiding the BGZF block-discovery pass. The scanner
still processes the entire header-to-EOF record stream, including variants
spanning boundaries, exactly once; it does not concatenate coordinate-region
queries. Boundary handoffs and index record counts (when present) are checked;
failures raise an error, without falling back or changing the index. Sparse
indexes may provide fewer useful splits than the requested worker count.
Without an index the byte/block splitter remains available.

CSI/TBI does not contain a source fingerprint. These checks detect some stale
indexes but cannot certify all their coordinate-query metadata. Split hints
never replace full-file parsing or determine which records are included.

Use ``threads=0`` for automatic concurrency. For local files, the default is
``max(1, available_cpus // 2)`` workers, each with one decompression thread
alongside its parsing thread. CPU availability accounts for process affinity,
Linux cgroup v1/v2 CPU quotas (including visible ancestor limits), and cpusets;
fractional CPU quotas are rounded down, with a minimum of one CPU. HTSlib also
creates an I/O helper thread, so this is a CPU-budget heuristic rather than a
strict OS-thread limit. Uncompressed inputs need no decompression threads.
With a one-CPU budget, decoding is synchronous. Remote inputs retain a single
worker. Explicit ``workers`` overrides automatic selection, even if it exceeds
the detected budget. Without ``threads=0``, the default remains a serial scan;
positive ``threads`` values retain their per-worker decompression meaning.

CSI indexes use HTSlib's format-specific depth selection from the declared
contig lengths, with the same fallback when lengths are unknown. The smallest
bins span 16,384 bases by default; exceptionally long contigs require larger
bins. TBI retains its fixed 512 Mi-base coordinate limit per contig.

BCF has no record sync marker, so each BCF worker locates its first record by
validating the full binary structure of several consecutive records. Every
located start is then checked against the exact offset where the previous
worker stopped; a range whose start does not match is rescanned from the
verified offset, so results never depend on the heuristic. VCF workers start at
the first line in their range and use the same handoff check. Uncompressed
VCF and BCF are split at arbitrary byte positions the same way; they cannot be
indexed.

.. code-block:: python

    summary = vcf.schema_scan(
        index_path="variants.vcf.gz.csi",
        workers=4,
        threads=1,
    )

Modifying Existing Records
==========================

.. this example is also in the writing.rst page

`cyvcf2` is optimized for fast reading and extraction from existing files.
However, it also offers some means of modifying existing VCFs. Here, we
show an example of how to modify the INFO field at a locus to annotate
variants with the genes that they overlap.

.. code-block:: python

    from cyvcf2 import VCF, Writer
    vcf = VCF(VCF_PATH)
    # adjust the header to contain the new field
    # the keys 'ID', 'Description', 'Type', and 'Number' are required.
    vcf.add_info_to_header({'ID': 'gene', 'Description': 'overlapping gene',
        'Type':'Character', 'Number': '1'})

    # create a new vcf Writer using the input vcf as a template.
    fname = "out.vcf"
    w = Writer(fname, vcf)

    for v in vcf:
        # The get_gene_intersections function is not shown.
        # This could be any operation to find intersections
        # or any manipulation required by the user.
        genes = get_gene_intersections(v)
        if genes is not None:
            v.INFO["gene"] = ",".join(genes)
        w.write_record(v)

    w.close(); vcf.close()

More info on writing vcfs can be found :doc:`here <writing>`

Setting Genotyping Strictness
=============================

By default, genotypes containing a single missing allele (e.g. genotypes such
as ``0/.``, ``./0``, ``1/.``, or ``./1``) are classified as heterozygous ("HET"
or id: 1) instead of "UNKNOWN" (or id: 2).  A case can be made for either
classification on these partially missing genotypes depending on the situation.

A better way to explicitly control the genotype classification for these cases,
is to use the "strict genotype", or ``strict_gt``, feature.  Here's an example
usage:

.. code-block:: python

    from cyvcf2 import VCF
    vcf = VCF("/path/to/vcf/file", strict_gt=True)
    for variant in vcf:
        # do something

When the ``strict_gt`` flag is enabled, `cyvcf2` will treat any genotype
containing a missing allele (containing a '.') as an UNKNOWN genotype;
otherwise, genotypes like ``0/.``, ``./0``, ``1/.``, or ``./1`` will be
classified as heterozygous ("HET").

The "strict genotype" feature is not enabled by default (i.e. ``strict_gt`` is
set to `False` by default) to preserve backwards compatibility.

Installation
============

.. code-block:: bash

    pip install cyvcf2

or via bioconda.

Testing
=======

Install `pytest`, then tests can be run with:

.. code-block:: bash

    pytest

Known Limitations
=================
* `cyvcf2` currently does not support reading VCFs encoded with UTF-8 with non-ASCII characters in the contents of string-typed FORMAT fields.
* Genotype-level phasing flags cannot fully represent partially phased
  polyploid calls. For example, ``0|1/2`` reports ``True`` and ``0/1|2``
  reports ``False`` because the flag follows the separator before the second
  allele, not every separator in the call. See :ref:`genotype-phasing` for
  the affected accessors and how to inspect individual allele phase bits.

For limitations on writing VCFs, see :ref:`here <Limitations with writing>`

See Also
========

Pysam also `has a cython wrapper to htslib <https://github.com/pysam-developers/pysam/blob/master/pysam/cbcf.pyx>`_ and one block of code here is taken directly from that library. But, the optimizations that we want for gemini are very specific so we have chosen to create a separate project.

.. toctree::
   :maxdepth: 1

   docstrings
   writing
   internal-release
