"""Optional UPath adapter, requiring universal_pathlib>=0.2.6 and s3fs>=2026.9.0."""

import argparse
from pathlib import Path
import tempfile

from upath import UPath

from cyvcf2 import build_index


def build_s3_index(path, index_path, *, workers=1, threads=0):
    destination = UPath(index_path)
    if destination.protocol != "s3":
        raise ValueError("this adapter requires an s3:// index destination")
    suffix = destination.suffix
    if suffix not in (".csi", ".tbi"):
        raise ValueError("index destination must end in .csi or .tbi")
    with tempfile.TemporaryDirectory(prefix="cyvcf2-s3-index-") as directory:
        staging = Path(directory) / ("index" + suffix)
        build_index(path, staging, workers=workers, threads=threads)
        # s3fs implements create with conditional PUT / multipart completion.
        destination.fs.put_file(str(staging), destination.path, mode="create")
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description="Index through HTSlib and publish to S3 without overwriting.")
    parser.add_argument("input", help="Local file or HTSlib-supported S3 URL")
    parser.add_argument("index", help="s3://bucket/key.csi or .tbi")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0, help="Decompression threads per worker")
    args = parser.parse_args()
    build_s3_index(args.input, args.index, workers=args.workers, threads=args.threads)


if __name__ == "__main__":
    main()
