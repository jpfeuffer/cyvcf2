# Experimental native batches

This fork adds an opt-in, sequential `VCF.read_batch` prototype. The existing
`Variant` interface is unchanged. This is a materialization experiment, not a
replacement for arbitrary VCF field access, schema discovery, or indexed queries.

```python
from cyvcf2 import VCF

with VCF("input.bcf", samples=["sample1", "sample2"]) as reader:
    while True:
        batch = reader.read_batch(
            ["FORMAT/GT", "FORMAT/DP", "FORMAT/PL", "INFO/AC"],
            max_rows=1024,
            max_bytes=64 * 1024 * 1024,
        )
        if batch["num_rows"] == 0:
            break
        dp = batch["fields"]["FORMAT/DP"]
        for row, width in enumerate(dp["widths"]):
            if width < 0:
                continue  # absent, not a present all-missing field
            start, stop = dp["offsets"][row:row + 2]
            values = dp["values"][start:stop].reshape(batch["num_samples"], width)
```

Only requested numeric INFO/FORMAT fields and raw FORMAT/GT are decoded into
columns. Header-declared Integer and Float fields may have fixed or variable
cardinality. Unknown fields, duplicate requests, and unsupported String,
Character and Flag fields fail explicitly before reading records. GT is the one
String-header exception: its BCF representation is numeric. No new dependency
or HTSlib modification is required.

## Result layout

All arrays are read-only and have independent native storage owners. They remain
valid after another batch, reader close, or reader destruction. No output array
borrows a record, header, or reusable HTSlib decoding buffer.

| Key | Representation |
| --- | --- |
| `num_rows`, `num_samples` | Python integers; zero rows means EOF |
| `nbytes` | Sum of all returned NumPy arrays' `nbytes`, including offsets |
| `POS` | int64, one-based positions in input order, including duplicates |
| `rid` | int32, header contig IDs (use `reader.seqnames` while open for names) |
| `rlen` | int64, HTSlib record length, equivalent to `variant.end - variant.start`, including symbolic/END records |
| `QUAL` | float32, unchanged HTSlib bits, including the missing sentinel |
| `strings` | uint8, concatenated packed basic-record strings |
| `string_offsets` | int64 byte offsets, `num_rows + 1` entries |
| `fields[name]["values"]` | Flattened int32 or float32 values |
| `fields[name]["offsets"]` | int64 **element** offsets, `num_rows + 1` entries |
| `fields[name]["widths"]` | int32 per-record FORMAT width per sample, or INFO element count; -1 means absent |

Each packed basic-record string is tab-delimited:
`CHROM TAB ID TAB REF [TAB ALT ...] TAB FILTERS`.
The first two and last components have fixed roles; the middle components are
the alleles, REF first. A record without ALT has only REF in this section.
ID `.` means missing, not an empty ID. FILTERS is the ordered semicolon-delimited
list returned by `Variant.FILTERS`: missing is empty, PASS is `PASS`. VCF fields
cannot contain literal tab delimiters. Bytes are preserved, not percent-decoded.
This intentionally avoids a Python string/list object for every output cell.

FORMAT row values reshape to `(num_samples, stored_width)`. Vector-end padding
remains **per sample**. Width -1 distinguishes absence from present all-missing
values; offsets then repeat. Width 0 denotes a present empty vector, if emitted
by HTSlib. When there are zero selected samples, FORMAT columns are excluded
(width -1); basic and INFO columns still contain every record. In particular,
the prototype does not call HTSlib's FORMAT unpacker on zero-sample BCF records
whose individual data were removed by sample subsetting.

Integer missing and vector-end values remain `INT32_MIN` and `INT32_MIN + 1`.
Float missing/vector-end/ordinary NaN bit patterns are copied without
normalization. A missing INFO value can therefore be present with width 1,
unlike an absent field. No downstream fill, dtype narrowing, schema padding, or
phase policy is applied.

GT values use HTSlib's raw genotype encoding: allele index is `(value >> 1) - 1`,
missing allele encodings are 0/1, vector end is `INT32_MIN + 1`, and the low bit
retains each allele's phase flag. Stop at vector end for sample ploidy; missing
alleles still occupy slots. First-allele implicit phase bits can vary by HTSlib
version. The reader never collapses these values to `gt_types` or chooses one
phase flag for a whole call. Converting to `variant.genotype.array()` requires
a separate compatibility adapter; that legacy API has a known haploid phase
lookahead issue, deliberately not changed here.

## Bounds, errors and ownership

`max_rows` and `max_bytes` must be positive. The byte limit covers all output
array allocations: numeric payloads, packed strings, and all offsets/widths.
Native output buffers request exactly their used byte counts, with no geometric
spare capacity. A batch never intentionally exceeds that limit.

**This is not a hard RSS or total working-memory limit.** Python objects,
NumPy headers, allocator size-class rounding and transient `realloc` copying,
codec buffers, header dictionaries, one reusable native record, per-requested-
field HTSlib decoding scratch, and temporary packed strings are outside the
output budget. A record must first be parsed and decoded to determine its size,
so even an oversized record can cause native high-water allocations before its
output is rejected. Previously returned batches also remain allocated as long
as the caller retains them. Release a previous batch before requesting another
if overlapping output owners are undesirable.

When the next row does not fit after at least one row, it is retained as a single
native lookahead record and the current batch is returned. If a single row
cannot fit, `ValueError` is raised; that record is retained, not dropped. Retry
with a larger cap or consume it through `next(reader)`. There is no empty-batch
retry loop that silently stalls on a large row. Changing samples or starting an
indexed region query while a lookahead is pending raises explicitly. Close
releases lookahead; all returned arrays survive.

Reading and decoding errors propagate; a failed multi-row call is not
transactional, so earlier records consumed during that failed call are not
returned. Do not retry a parse failure as if it were a byte-cap refusal.
This sequential API does not expose a region argument or batch an existing
region generator. Do not interleave active indexed iterators with sequential
reading on the same HTS handle; use separate readers.

The implementation reuses a native record and decoding scratch inside each
batch, copies into column buffers, and creates NumPy views once per batch,
without constructing `Variant` instances or per-row NumPy arrays.
It still uses HTSlib's existing VCF/BCF parser and numeric decoders; it is not a
new parser, file cache, scan-only API, or ISA-specific implementation.
