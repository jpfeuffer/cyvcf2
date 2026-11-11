"""Optional UPath adapter, requiring universal_pathlib>=0.2.6 and s3fs>=2026.9.0."""

import argparse
from upath import UPath

from cyvcf2 import build_index_stream


def build_s3_index(path, index_path, *, workers=1, threads=0):
    destination = UPath(index_path, skip_instance_cache=True)
    if destination.protocol != "s3":
        raise ValueError("this adapter requires an s3:// index destination")
    suffix = destination.suffix
    if suffix not in (".csi", ".tbi"):
        raise ValueError("index destination must end in .csi or .tbi")
    # Defer publication until the native producer has completed successfully.
    with destination.fs.transaction:
        with destination.open("xb", block_size=5 << 20) as output:
            build_index_stream(path, output, index_format=suffix[1:], workers=workers, threads=threads)
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description="Stream a bounded-memory VCF index to S3 without overwriting.")
    parser.add_argument("input", help="BGZF VCF local file or HTSlib-supported S3 URL")
    parser.add_argument("index", help="s3://bucket/key.csi or .tbi")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0, help="Decompression threads per worker")
    args = parser.parse_args()
    build_s3_index(args.input, args.index, workers=args.workers, threads=args.threads)


if __name__ == "__main__":
    main()
