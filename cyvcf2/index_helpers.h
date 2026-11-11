#ifndef CYVCF2_INDEX_HELPERS_H
#define CYVCF2_INDEX_HELPERS_H

#include <stdint.h>
#include "htslib/tbx.h"

/* tbx_parse1 is internal to bundled HTSlib, not an external-library ABI. */
typedef struct {
    int64_t beg, end;
    char *ss, *se;
    int tid;
} cyvcf2_tbx_interval;

#ifdef CYVCF2_BUNDLED_HTSLIB
extern int tbx_parse1(const tbx_conf_t *, size_t, char *, cyvcf2_tbx_interval *);
#define CYVCF2_COORDINATE_WORKERS 1
static inline int cyvcf2_parse_interval(size_t length, char *line,
                                      cyvcf2_tbx_interval *interval) {
    return tbx_parse1(&tbx_conf_vcf, length, line, interval);
}
#else
#define CYVCF2_COORDINATE_WORKERS 0
static inline int cyvcf2_parse_interval(size_t length, char *line,
                                      cyvcf2_tbx_interval *interval) {
    return -1;
}
#endif

#endif
