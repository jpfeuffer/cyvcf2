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

Pass ``index_path`` to reuse an existing index at that path or create one
during the scan when it does not exist. Index creation supports CSI for VCF or
BCF and TBI for BGZF-compressed VCF. It requires coordinate-sorted input.
Without ``index_path``, an existing default index is detected but no index is
created.

For a local BGZF-compressed VCF or BCF, pass ``workers`` to scan independent
BGZF block ranges in parallel. Each worker returns compact summaries and, when an
index is requested, streams its record offsets to a temporary file. The
offsets are fed in input order to one in-memory HTSlib index builder; the
finished CSI/TBI is written after all ranges are processed. ``threads``
continues to control HTSlib decompression threads per scan.

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

For limitations on writing VCFs, see :ref:`here <Limitations with writing>`

See Also
========

Pysam also `has a cython wrapper to htslib <https://github.com/pysam-developers/pysam/blob/master/pysam/cbcf.pyx>`_ and one block of code here is taken directly from that library. But, the optimizations that we want for gemini are very specific so we have chosen to create a separate project.

.. toctree::
   :maxdepth: 1

   docstrings
   writing
