"""Download and unpack GoPro Large locally.

The original `load_gopro.sh` uses a lab-specific `/mnt/ssd2/...` path. This
script keeps everything under this repository:

    datasets/downloads/GOPRO_Large.zip
    datasets/GoPro/{train,test}/...
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


URL = "https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip"


def download(url: str, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if start else "wb"

    print(f"[download] {url}")
    print(f"[download] {archive} (resume from {start} bytes)")
    with urllib.request.urlopen(req, timeout=60) as response, partial.open(mode) as handle:
        length = response.headers.get("Content-Length")
        total = int(length) + start if length else None
        downloaded = start
        last_report = start
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded - last_report >= 256 * 1024 * 1024:
                if total:
                    print(f"[download] {downloaded/1e9:.2f}/{total/1e9:.2f} GB ({downloaded/total*100:.1f}%)", flush=True)
                else:
                    print(f"[download] {downloaded/1e9:.2f} GB", flush=True)
                last_report = downloaded
    partial.rename(archive)
    print(f"[download] complete: {archive}")


def unpack(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".gopro_unpacked"
    if marker.exists() and (target / "train").exists() and (target / "test").exists():
        print(f"[unpack] already prepared: {target}")
        return

    tmp = target.parent / "_GoPro_unpack_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    print(f"[unpack] extracting {archive} -> {tmp}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(tmp)

    candidates = [tmp, tmp / "GOPRO_Large", tmp / "GoPro", tmp / "GOPRO"]
    root = next((p for p in candidates if (p / "train").exists() and (p / "test").exists()), None)
    if root is None:
        matches = [p for p in tmp.rglob("train") if p.is_dir() and (p.parent / "test").exists()]
        if matches:
            root = matches[0].parent
    if root is None:
        raise RuntimeError(f"Could not find train/test folders under {tmp}")

    if target.exists():
        for child in target.iterdir():
            if child.name != "downloads":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    for child in root.iterdir():
        shutil.move(str(child), str(target / child.name))
    marker.write_text("ok\n", encoding="utf-8")
    shutil.rmtree(tmp)
    print(f"[unpack] ready: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("datasets/downloads/GOPRO_Large.zip"))
    parser.add_argument("--target", type=Path, default=Path("datasets/GoPro"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-unpack", action="store_true")
    args = parser.parse_args()

    if not args.skip_download and not args.archive.exists():
        download(URL, args.archive)
    if not args.archive.exists():
        raise FileNotFoundError(args.archive)
    if not args.skip_unpack:
        unpack(args.archive, args.target)


if __name__ == "__main__":
    main()
