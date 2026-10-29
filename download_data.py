"""
Download the data release from Zenodo (https://doi.org/10.5281/zenodo.22864335) into this repository.

The release (CC-BY 4.0) has two kinds of zip, listed with their SHA-256 hashes and the
figures they serve in MANIFEST.tsv:

- dedifferentiation_core.zip: per-simulation differentiation tables, IT->V1 gradient
  arrays, the unit-selectivity tables used for the simulations, and the statistics
  outputs. Enough to redraw every figure and table without rerunning the simulations.
- rdms_<model_dir>_<damage_type>_top<fraction>.zip (optional, up to 2 GB each): the
  category-selective RDMs, needed only to rebuild the differentiation tables with
  compute_differentiation.py.

Each zip is checked against MANIFEST.tsv and unpacked into the repository root, where the
scripts read it (data/, unit_selectivity/, stats/). Downloads are kept in downloads/ so a
rerun does not fetch them again. Existing files are kept if identical and the script stops
if one differs, so it never silently replaces a file.

Usage:
    python download_data.py                         # core only
    python download_data.py --list                  # show every zip and the figures it serves
    python download_data.py --rdms rdms_cornet_rt5_c_noise_top0.05.zip
    python download_data.py --rdms all --skip-core  # every RDM part (~27 GB)
    python download_data.py --from-dir Downloads/zenodo   # zips and MANIFEST.tsv downloaded by hand
"""

import argparse
import csv
import hashlib
import shutil
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath

ZENODO_RECORD_URL = "https://zenodo.org/records/22864336/files"
MANIFEST_NAME = "MANIFEST.tsv"
CORE_ZIP_NAME = "dedifferentiation_core.zip"
DOWNLOAD_DIR = Path("downloads")
OUTPUT_ROOT = Path(".")
# Top-level folders a release zip may write to
ALLOWED_TOP_DIRS = ("data", "unit_selectivity", "stats")
CHUNK_BYTES = 16 * 1024 * 1024


def sha256_of_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def crc32_of_file(path: Path) -> int:
    """Return the CRC-32 of a file, as stored for zip members."""
    crc = 0
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            crc = zlib.crc32(chunk, crc)
    return crc


def download(url: str, target: Path) -> None:
    """Download a URL to a file, writing to a temporary name until it is complete."""
    partial = target.with_name(target.name + ".partial")
    print(f"Downloading {url} ...", flush=True)
    with urllib.request.urlopen(url) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f, CHUNK_BYTES)
    partial.replace(target)


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """
    Read the release manifest.

    Parameters:
        manifest_path (Path): Tab-separated file with columns file, size_bytes, sha256, contents, figures.

    Returns:
        list[dict[str, str]]: One row per zip in the release.
    """
    with open(manifest_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    required = {"file", "size_bytes", "sha256", "contents", "figures"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"{manifest_path} should have rows with columns {sorted(required)}.")
    return rows


def select_files(manifest: list[dict[str, str]], rdm_parts: list[str], skip_core: bool) -> list[dict[str, str]]:
    """
    Choose the manifest rows to download.

    Parameters:
        manifest (list[dict[str, str]]): Rows from read_manifest().
        rdm_parts (list[str]): RDM zip names, or ["all"] for every RDM part.
        skip_core (bool): Leave out the core zip.

    Returns:
        list[dict[str, str]]: Selected rows, core first, in manifest order.

    Example:
        rdm_parts=["rdms_cornet_rt5_c_noise_top0.05.zip"] -> core zip + that part
    """
    available = [row["file"] for row in manifest if row["file"] != CORE_ZIP_NAME]
    wanted = set(available) if rdm_parts == ["all"] else set(rdm_parts)
    unknown = sorted(wanted - set(available))
    if unknown:
        raise ValueError(f"Unknown RDM parts {unknown}; choose from {available} or 'all'.")
    if not skip_core:
        wanted.add(CORE_ZIP_NAME)
    if not wanted:
        raise ValueError("Nothing to download: --skip-core was given without --rdms.")
    return [row for row in manifest if row["file"] in wanted]


def fetch_verified(row: dict[str, str], download_dir: Path, from_dir: Path | None) -> Path:
    """
    Return a local copy of one release zip whose SHA-256 matches the manifest.

    Parameters:
        row (dict[str, str]): Manifest row of the zip.
        download_dir (Path): Where downloads are kept (unused with from_dir).
        from_dir (Path | None): Folder holding zips downloaded by hand, or None to download from Zenodo.

    Returns:
        Path: The verified zip.
    """
    path = (from_dir if from_dir is not None else download_dir) / row["file"]
    if path.exists() and sha256_of_file(path) == row["sha256"]:
        return path
    if from_dir is not None:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; download it from {ZENODO_RECORD_URL}.")
    else:
        download_dir.mkdir(parents=True, exist_ok=True)
        download(f"{ZENODO_RECORD_URL}/{urllib.parse.quote(row['file'])}?download=1", path)
        if sha256_of_file(path) == row["sha256"]:
            return path
    raise ValueError(f"{path} does not match the SHA-256 hash in {MANIFEST_NAME}; delete it and retry.")


def check_member_name(name: str) -> None:
    """Raise if a zip member would be written outside the allowed top-level folders."""
    parts = PurePosixPath(name).parts
    if name.startswith("/") or ".." in parts or not parts or parts[0] not in ALLOWED_TOP_DIRS:
        raise ValueError(f"Unexpected zip member {name!r}; members must lie under {ALLOWED_TOP_DIRS}.")


def extract_release_zip(zip_path: Path, output_root: Path) -> None:
    """
    Unpack a release zip into the repository, keeping identical existing files.

    Parameters:
        zip_path (Path): Verified release zip.
        output_root (Path): Repository root.

    Returns:
        None
    """
    with zipfile.ZipFile(zip_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        for info in members:
            check_member_name(info.filename)
        n_written = 0
        for info in members:
            target = output_root / info.filename
            if target.exists():
                if target.stat().st_size != info.file_size or crc32_of_file(target) != info.CRC:
                    raise FileExistsError(f"{target} exists but differs from {zip_path.name}; remove it first.")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with archive.open(info) as src, open(partial, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK_BYTES)
            partial.replace(target)
            n_written += 1
    print(f"{zip_path.name}: {len(members)} files verified, {n_written} written")


def main() -> None:
    """Download, verify, and unpack the selected parts of the data release."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rdms", nargs="+", default=[],
                        help="RDM zips to download as well (names from --list), or 'all'.")
    parser.add_argument("--skip-core", action="store_true", help="Do not download the core zip.")
    parser.add_argument("--list", action="store_true", help="Print the zips in the release and exit.")
    parser.add_argument("--from-dir", type=Path, default=None,
                        help="Use MANIFEST.tsv and zips already downloaded by hand into this folder.")
    parser.add_argument("--download-dir", type=Path, default=DOWNLOAD_DIR,
                        help=f"Where downloaded zips are kept (default: {DOWNLOAD_DIR}).")
    args = parser.parse_args()

    if args.from_dir is not None:
        manifest_path = args.from_dir / MANIFEST_NAME
    else:
        args.download_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.download_dir / MANIFEST_NAME
        download(f"{ZENODO_RECORD_URL}/{MANIFEST_NAME}?download=1", manifest_path)
    manifest = read_manifest(manifest_path)

    if args.list:
        for row in manifest:
            print(f"{row['file']}  ({int(row['size_bytes']) / 1e6:,.0f} MB)  {row['figures']}")
        return
    for row in select_files(manifest, args.rdms, args.skip_core):
        extract_release_zip(fetch_verified(row, args.download_dir, args.from_dir), OUTPUT_ROOT)


if __name__ == "__main__":
    main()
