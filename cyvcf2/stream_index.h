#ifndef CYVCF2_STREAM_INDEX_H
#define CYVCF2_STREAM_INDEX_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "htslib/bgzf.h"
#include "htslib/hts.h"

typedef struct {
    uint64_t start, end;
} cyvcf2_stream_bin;

typedef struct {
    cyvcf2_stream_bin *bins;
    uint64_t *linear;
    BGZF *output;
    uint32_t nbins, used, n_linear;
    int shift, levels, format, tid;
    int64_t last_pos;
    uint64_t last_offset, ref_start, count;
    uint64_t fingerprint, total;
} cyvcf2_stream_index;

static int cyvcf2_stream_allocate(cyvcf2_stream_index *s) {
    s->bins = calloc(s->nbins, sizeof(*s->bins));
    s->linear = calloc((size_t)1 << (3 * s->levels), sizeof(*s->linear));
    return s->bins && s->linear ? 0 : -1;
}

static int cyvcf2_stream_write(BGZF *f, uint64_t value, int n) {
    unsigned char b[8];
    int i;
    for (i = 0; i < n; i++) b[i] = (unsigned char)(value >> (8 * i));
    return bgzf_write(f, b, n) == n ? 0 : -1;
}

static int cyvcf2_stream_reference(cyvcf2_stream_index *s) {
    uint32_t i;
    for (i = s->n_linear; i > 1; i--)
        if (!s->linear[i - 2]) s->linear[i - 2] = s->linear[i - 1];
    if (cyvcf2_stream_write(s->output, s->used + 1, 4) < 0) return -1;
    /* One conservative chunk per bin bounds memory independently of rows. */
    for (i = 0; i < s->nbins; i++) {
        cyvcf2_stream_bin *b = &s->bins[i];
        uint32_t bottom;
        if (!b->end) continue;
        bottom = hts_bin_bot(i, s->levels);
        if (cyvcf2_stream_write(s->output, i, 4) < 0 ||
            (s->format == HTS_FMT_CSI &&
             cyvcf2_stream_write(s->output,
                 bottom < s->n_linear ? s->linear[bottom] : 0, 8) < 0) ||
            cyvcf2_stream_write(s->output, 1, 4) < 0 ||
            cyvcf2_stream_write(s->output, b->start, 8) < 0 ||
            cyvcf2_stream_write(s->output, b->end, 8) < 0) return -1;
    }
    if (cyvcf2_stream_write(s->output, s->nbins + 1, 4) < 0 ||
        (s->format == HTS_FMT_CSI && cyvcf2_stream_write(s->output, 0, 8) < 0) ||
        cyvcf2_stream_write(s->output, 2, 4) < 0 ||
        cyvcf2_stream_write(s->output, s->ref_start, 8) < 0 ||
        cyvcf2_stream_write(s->output, s->last_offset, 8) < 0 ||
        cyvcf2_stream_write(s->output, s->count, 8) < 0 ||
        cyvcf2_stream_write(s->output, 0, 8) < 0)
        return -1;
    if (s->format == HTS_FMT_TBI) {
        if (cyvcf2_stream_write(s->output, s->n_linear, 4) < 0) return -1;
        for (i = 0; i < s->n_linear; i++)
            if (cyvcf2_stream_write(s->output, s->linear[i], 8) < 0) return -1;
    }
    memset(s->bins, 0, s->nbins * sizeof(*s->bins));
    memset(s->linear, 0, ((size_t)1 << (3 * s->levels)) * sizeof(*s->linear));
    s->used = 0;
    s->n_linear = 0;
    return 0;
}

static int cyvcf2_stream_push(cyvcf2_stream_index *s, int tid,
                              int64_t beg, int64_t end, uint64_t offset) {
    int bin;
    cyvcf2_stream_bin *entry;
    if (end < beg || beg > ((int64_t)1 << (s->shift + 3 * s->levels)) ||
        end > ((int64_t)1 << (s->shift + 3 * s->levels)) ||
        (tid == s->tid && beg < s->last_pos) ||
        (tid != s->tid && tid != s->tid + 1))
        return -1;
    if (tid != s->tid) {
        if (s->output && s->tid >= 0 && cyvcf2_stream_reference(s) < 0) return -2;
        s->tid = tid;
        s->ref_start = s->last_offset;
        s->count = 0;
    }
    s->last_pos = beg < 0 ? 0 : beg;
    if (beg < 0) beg = 0;
    if (end <= 0) end = 1;
    if (s->output) {
        uint32_t first, last, i;
        bin = hts_reg2bin(beg, end, s->shift, s->levels);
        if (bin < 0 || (uint32_t)bin >= s->nbins) return -1;
        first = beg >> s->shift;
        last = (end - 1) >> s->shift;
        if (first >= (1U << (3 * s->levels)) ||
            last >= (1U << (3 * s->levels))) return -1;
        for (i = first; i <= last; i++)
            if (!s->linear[i]) s->linear[i] = s->last_offset;
        if (s->n_linear < last + 1) s->n_linear = last + 1;
        entry = &s->bins[bin];
        if (!entry->end) {
            entry->start = s->last_offset;
            s->used++;
        }
        entry->end = offset;
    }
    s->last_offset = offset;
    s->count++;
    s->total++;
    s->fingerprint = (s->fingerprint ^ (uint64_t)tid) * UINT64_C(1099511628211);
    s->fingerprint = (s->fingerprint ^ (uint64_t)beg) * UINT64_C(1099511628211);
    s->fingerprint = (s->fingerprint ^ (uint64_t)end) * UINT64_C(1099511628211);
    s->fingerprint = (s->fingerprint ^ offset) * UINT64_C(1099511628211);
    return 0;
}

#endif
