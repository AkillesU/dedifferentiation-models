"""Tests for download_stimuli.py using a small fake OSF zip (no network access)."""

import io
import zipfile

import pytest

import download_stimuli


def fake_zip_and_manifest(content=b"image"):
    """Return (zip bytes, manifest rows) for 64 identical fake images."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for i in range(1, 65):
            archive.writestr(f"stimuli/{i}.jpg", content)
    digest = download_stimuli.sha256_of_bytes(content)
    manifest = [{"osf_file": f"{i}.jpg", "file_name": f"face{i}.jpg", "category": "face", "sha256": digest} for i in range(1, 65)]
    return buffer.getvalue(), manifest


def test_extract_stimuli_writes_and_is_idempotent(tmp_path):
    zip_bytes, manifest = fake_zip_and_manifest()
    download_stimuli.extract_stimuli(zip_bytes, manifest, tmp_path)
    download_stimuli.extract_stimuli(zip_bytes, manifest, tmp_path)
    assert len(list(tmp_path.glob("*.jpg"))) == 64


def test_extract_stimuli_rejects_wrong_hash(tmp_path):
    zip_bytes, manifest = fake_zip_and_manifest()
    manifest[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        download_stimuli.extract_stimuli(zip_bytes, manifest, tmp_path)


def test_extract_stimuli_never_overwrites_a_different_file(tmp_path):
    zip_bytes, manifest = fake_zip_and_manifest()
    (tmp_path / "face1.jpg").write_bytes(b"other")
    with pytest.raises(FileExistsError):
        download_stimuli.extract_stimuli(zip_bytes, manifest, tmp_path)


def test_repository_manifest_lists_16_images_per_category():
    rows = download_stimuli.read_manifest(download_stimuli.MANIFEST_PATH)
    assert sorted({r["category"] for r in rows}) == ["animal", "face", "object", "place"]
    assert all(sum(r["category"] == c for r in rows) == 16 for c in ("animal", "face", "object", "place"))
