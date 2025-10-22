#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import string
from pathlib import Path
from typing import List, Optional, Dict, Any
from typing import List, Optional, Dict, Any

def random_name(prefix: str = "") -> str:
    letters = string.ascii_letters + string.digits + "-_"
    return prefix + "".join(random.choice(letters) for _ in range(12))


def ensure_dirs(root: Path) -> List[Path]:
    # Build a moderately complex tree
    subdirs = [
        "projects/app/src",
        "projects/app/build",
        "projects/lib/include",
        "backup/2021/12",
        "backup/2022/01",
        "media/images/raw",
        "media/images/jpg",
        "media/videos",
        "home/user/docs",
        "home/user/docs/archive/2020",
        "home/user/docs/archive/2021",
        "var/log/app",
        "tmp/chunks",
        "tmp/cache",
    ]
    paths = []
    for s in subdirs:
        p = root / s
        p.mkdir(parents=True, exist_ok=True)
        paths.append(p)
    return paths


def write_random_file(path: Path, size_bytes: int, chunk_size: int = 1024 * 1024) -> None:
    remaining = size_bytes
    with open(path, "wb") as f:
        while remaining > 0:
            n = min(remaining, chunk_size)
            f.write(os.urandom(n))
            remaining -= n


def write_zero_file(path: Path, size_bytes: int, chunk_size: int = 1024 * 1024) -> None:
    remaining = size_bytes
    zero = b"\x00" * chunk_size
    with open(path, "wb") as f:
        while remaining > 0:
            n = min(remaining, chunk_size)
            f.write(zero[:n])
            remaining -= n


def create_duplicate_group(dest_dirs: List[Path], base_content: Optional[bytes], size_bytes: int, copies: int) -> List[Path]:
    files = []
    for i in range(copies):
        d = random.choice(dest_dirs)
        name = random_name("dup_") + f"_{i}.bin"
        p = d / name
        if base_content is None:
            write_zero_file(p, size_bytes)
        else:
            with open(p, "wb") as f:
                # repeat base_content to reach size
                remaining = size_bytes
                while remaining > 0:
                    chunk = base_content[: min(len(base_content), remaining)]
                    f.write(chunk)
                    remaining -= len(chunk)
        files.append(p)
    return files


def populate_dataset(root: Path, total_mb: int, dupe_groups: int, dupe_copies: int, min_large: int, max_large: int) -> Dict[str, Any]:
    random.seed(1337)
    root.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(root)

    bytes_target = total_mb * 1024 * 1024
    bytes_written = 0
    created = []

    # 1) Small files (< min-bytes) — should be skipped by the grader
    small_sizes = [13, 99, 257, 511, 777, 1023]
    for _ in range(200):
        d = random.choice(dirs)
        p = d / (random_name("small_") + ".txt")
        write_random_file(p, random.choice(small_sizes))
        created.append(p)

    # 2) Edge-case exactly 1024 bytes
    for _ in range(20):
        d = random.choice(dirs)
        p = d / (random_name("edge_") + ".bin")
        write_random_file(p, 1024)
        created.append(p)
        bytes_written += 1024

    # 3) Duplicate groups (mix zeros and random patterns)
    large_sizes = [
        random.randint(min_large, max_large) for _ in range(dupe_groups // 2)
    ] + [
        random.randint(min_large, max_large) for _ in range(dupe_groups - dupe_groups // 2)
    ]
    for i, sz in enumerate(large_sizes):
        if i % 2 == 0:
            base = os.urandom(64 * 1024)  # random pattern
        else:
            base = None  # zeros
        files = create_duplicate_group(dirs, base, sz, dupe_copies)
        created.extend(files)
        bytes_written += sz * dupe_copies
        if bytes_written >= bytes_target:
            break

    # 4) Unique large-ish files to fill up to target size
    while bytes_written < bytes_target:
        d = random.choice(dirs)
        size = random.randint(min_large, max_large)
        p = d / (random_name("uniq_") + ".dat")
        write_random_file(p, size)
        created.append(p)
        bytes_written += size

    # 5) Some nested deep paths and long filenames
    deep = root / "a" / "b" / "c" / "d" / "e" / "f"
    deep.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        long_name = ("very_" * 10) + random_name() + ".log"
        p = deep / long_name
        write_zero_file(p, 2 * 1024 * 1024)
        created.append(p)
        bytes_written += 2 * 1024 * 1024

    return {
        "root": str(root),
        "files": len(created),
        "bytes": bytes_written,
        "target_bytes": bytes_target,
        "dupe_groups": dupe_groups,
        "dupe_copies": dupe_copies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a heavy dataset for dirhash-fast benchmark.")
    parser.add_argument("--root", default="/data", help="Destination dataset root (will be created)")
    parser.add_argument("--total-mb", type=int, default=256, help="Approx total size in MB (default: 256)")
    parser.add_argument("--dupe-groups", type=int, default=24, help="Number of duplicate groups (default: 24)")
    parser.add_argument("--dupe-copies", type=int, default=5, help="Copies per duplicate group (default: 5)")
    parser.add_argument("--min-large", type=int, default=1 * 1024 * 1024, help="Min size for large files (bytes)")
    parser.add_argument("--max-large", type=int, default=8 * 1024 * 1024, help="Max size for large files (bytes)")
    parser.add_argument("--reset", action="store_true", help="Delete root if exists before creating")
    args = parser.parse_args()

    root = Path(args.root)
    if root.exists() and args.reset:
        shutil.rmtree(root)

    meta = populate_dataset(
        root=root,
        total_mb=args.total_mb,
        dupe_groups=args.dupe_groups,
        dupe_copies=args.dupe_copies,
        min_large=args.min_large,
        max_large=args.max_large,
    )
    print(
        f"Created dataset at {meta['root']} — files={meta['files']} bytes={meta['bytes']:,} (target={meta['target_bytes']:,}), dupes={meta['dupe_groups']}x{meta['dupe_copies']}"
    )


if __name__ == "__main__":
    main()


