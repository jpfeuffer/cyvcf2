"""Optional UPath adapter, requiring universal_pathlib>=0.2.6 and s3fs>=2026.9.0."""

import argparse
import sys
from upath import UPath

from cyvcf2 import build_index_stream


def build_s3_index(path, index_path, *, workers=1, threads=0, progress=None):
    destination = UPath(index_path, skip_instance_cache=True)
    if destination.protocol != "s3":
        raise ValueError("this adapter requires an s3:// index destination")
    suffix = destination.suffix
    if suffix not in (".csi", ".tbi"):
        raise ValueError("index destination must end in .csi or .tbi")
    # Defer publication until the native producer has completed successfully.
    with destination.fs.transaction:
        with destination.open("xb", block_size=5 << 20) as output:
            build_index_stream(
                path, output, index_format=suffix[1:], workers=workers,
                threads=threads, progress=progress,
            )
    return str(destination)


def _log_progress(event):
    eta = event["eta_seconds"]
    remaining = "unknown" if eta is None else f"{eta:.0f}s"
    print(
        f'{event["phase"]}: {event["fraction"]:.1%} '
        f'(pass {event["pass"]}/{event["passes"]}), '
        f'{event["bytes_per_second"] / (1 << 20):.1f} MiB/s, ETA {remaining}',
        file=sys.stderr, flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Stream a bounded-memory VCF index to S3 without overwriting.")
    parser.add_argument("input", help="BGZF VCF local file or HTSlib-supported S3 URL")
    parser.add_argument("index", help="s3://bucket/key.csi or .tbi")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0, help="Decompression threads per worker")
    parser.add_argument("--progress", action="store_true", help="Log approximate progress and ETA to stderr")
    args = parser.parse_args()
    build_s3_index(
        args.input, args.index, workers=args.workers, threads=args.threads,
        progress=_log_progress if args.progress else None,
    )
    if args.progress:
        print("Index published successfully.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
