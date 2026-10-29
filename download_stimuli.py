"""
Download the 64 experimental stimuli of Haupt et al. (2025) from OSF into stimuli/.

The images are not redistributed in this repository. They are part of the Haupt et
al. OSF project (https://osf.io/xeukw/, file data/stimuli/stimuli.zip), where they are
numbered 1.jpg to 64.jpg in blocks of 16: faces, animals, places, objects. This script
downloads that zip, checks every image against the SHA-256 hashes in
stimuli/MANIFEST.csv, and saves it under the category name the analysis code expects
(e.g. "17.jpg" -> "animal1.jpg"), because image categories are read from file names.

Existing files are kept if their hash matches and the script stops if one differs,
so it never silently replaces a stimulus.

Usage:
    python download_stimuli.py
    python download_stimuli.py --zip-path Downloads/stimuli.zip   # use a manually downloaded zip
"""

import argparse
import csv
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

OSF_STIMULI_URL = "https://osf.io/download/3mwje/"
STIMULI_DIR = Path("stimuli")
MANIFEST_PATH = STIMULI_DIR / "MANIFEST.csv"


def sha256_of_bytes(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """
    Read the stimulus manifest.

    Parameters:
        manifest_path (Path): CSV with columns osf_file, file_name, category, sha256.

    Returns:
        list[dict[str, str]]: One row per stimulus (64 rows).
    """
    with open(manifest_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 64:
        raise ValueError(f"{manifest_path} should list 64 stimuli; found {len(rows)}.")
    return rows


def load_zip_bytes(zip_path: Path | None) -> bytes:
    """Return the stimuli zip as bytes, downloading it from OSF unless a local path is given."""
    if zip_path is not None:
        return zip_path.read_bytes()
    print(f"Downloading {OSF_STIMULI_URL} ...")
    with urllib.request.urlopen(OSF_STIMULI_URL) as response:
        return response.read()


def extract_stimuli(zip_bytes: bytes, manifest: list[dict[str, str]], output_dir: Path) -> None:
    """
    Verify each stimulus in the OSF zip and write it under its category file name.

    Parameters:
        zip_bytes (bytes): Contents of the OSF stimuli.zip.
        manifest (list[dict[str, str]]): Rows from read_manifest().
        output_dir (Path): Destination folder, normally stimuli/.

    Returns:
        None
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        # The OSF zip stores images as stimuli/<n>.jpg
        members = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
        for row in manifest:
            if row["osf_file"] not in members:
                raise FileNotFoundError(f"{row['osf_file']} is missing from the OSF zip.")
            data = archive.read(members[row["osf_file"]])
            if sha256_of_bytes(data) != row["sha256"]:
                raise ValueError(f"{row['osf_file']} does not match the expected SHA-256 hash.")

            target = output_dir / row["file_name"]
            if target.exists():
                if sha256_of_bytes(target.read_bytes()) != row["sha256"]:
                    raise FileExistsError(f"{target} exists but differs from {row['osf_file']}; remove it first.")
                continue
            target.write_bytes(data)
    print(f"{len(manifest)} stimuli verified in {output_dir}/")


def main() -> None:
    """Download, verify, and rename the experimental stimuli."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Use a stimuli.zip already downloaded from OSF instead of downloading it.",
    )
    args = parser.parse_args()

    manifest = read_manifest(MANIFEST_PATH)
    extract_stimuli(load_zip_bytes(args.zip_path), manifest, STIMULI_DIR)


if __name__ == "__main__":
    main()
