"""Tests for download_data.py using small fake release zips (no network access)."""

import csv
import zipfile

import pytest

import download_data

PART_NAME = "rdms_cornet_rt5_c+b_units_top0.05.zip"


def write_release(release_dir, members_by_zip):
    """Write fake release zips and their MANIFEST.tsv; return the manifest rows."""
    release_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for zip_name, members in members_by_zip.items():
        path = release_dir / zip_name
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        rows.append({"file": zip_name, "size_bytes": str(path.stat().st_size),
                     "sha256": download_data.sha256_of_file(path), "contents": "test", "figures": "Fig 1b"})
    with open(release_dir / download_data.MANIFEST_NAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


@pytest.fixture
def release(tmp_path):
    """Return (release folder, manifest rows) of a core zip and one RDM part."""
    rows = write_release(tmp_path / "release", {
        download_data.CORE_ZIP_NAME: {"data/differentiation/cornet_rt5_c.csv.gz": b"table",
                                      "stats/layer_choice/cornet_rt5_layer_scores.csv": b"scores"},
        PART_NAME: {"data/simulations/cornet_rt5_c+b/units/V1/x.npz": b"rdm"},
    })
    return tmp_path / "release", rows


def test_read_manifest_round_trip(release):
    release_dir, rows = release
    assert download_data.read_manifest(release_dir / download_data.MANIFEST_NAME) == rows


def test_select_files_core_by_default_and_parts_on_request(release):
    _, rows = release
    assert [r["file"] for r in download_data.select_files(rows, [], False)] == [download_data.CORE_ZIP_NAME]
    assert [r["file"] for r in download_data.select_files(rows, ["all"], True)] == [PART_NAME]
    assert len(download_data.select_files(rows, [PART_NAME], False)) == 2


def test_select_files_rejects_unknown_part_and_empty_selection(release):
    _, rows = release
    with pytest.raises(ValueError, match="Unknown RDM parts"):
        download_data.select_files(rows, ["rdms_missing.zip"], False)
    with pytest.raises(ValueError, match="Nothing to download"):
        download_data.select_files(rows, [], True)


def test_fetch_verified_rejects_wrong_hash(release, tmp_path):
    release_dir, rows = release
    rows[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        download_data.fetch_verified(rows[0], tmp_path / "downloads", release_dir)


def test_extract_writes_into_repository_layout_and_is_idempotent(release, tmp_path):
    release_dir, rows = release
    root = tmp_path / "repo"
    for row in rows:
        zip_path = download_data.fetch_verified(row, tmp_path / "downloads", release_dir)
        download_data.extract_release_zip(zip_path, root)
        download_data.extract_release_zip(zip_path, root)
    assert (root / "data/differentiation/cornet_rt5_c.csv.gz").read_bytes() == b"table"
    assert (root / "data/simulations/cornet_rt5_c+b/units/V1/x.npz").read_bytes() == b"rdm"


def test_extract_never_overwrites_a_different_file(release, tmp_path):
    release_dir, _ = release
    target = tmp_path / "repo" / "data/differentiation/cornet_rt5_c.csv.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"other")
    with pytest.raises(FileExistsError):
        download_data.extract_release_zip(release_dir / download_data.CORE_ZIP_NAME, tmp_path / "repo")


@pytest.mark.parametrize("name", ["../outside.txt", "configs/main.yaml", "/abs/file.txt", "data/../../x"])
def test_extract_rejects_members_outside_allowed_folders(tmp_path, name):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(name, b"x")
    with pytest.raises(ValueError, match="Unexpected zip member"):
        download_data.extract_release_zip(zip_path, tmp_path / "repo")
