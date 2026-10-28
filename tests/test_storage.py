"""Publishing, immutable artifacts, verified downloads, snapshots, and offline cache."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _support import (
    COMMIT,
    DownloadRepository,
    FakeS3,
    SnapshotRepository,
)
from huggingface_hub import errors

from hf_hub_s3_mirror import _artifacts, _cache, mirror, publish


def _unsupported_symlink(*args: object, **kwargs: object) -> None:
    raise OSError(errno.EPERM, "Operation not permitted")


def _set_symlink_support(monkeypatch: pytest.MonkeyPatch, supported: bool) -> None:
    if not supported:
        monkeypatch.setattr(os, "symlink", _unsupported_symlink)


def _snapshot_file(cache_dir: Path, commit: str) -> Path:
    return cache_dir / "models--example--model" / "snapshots" / commit / "config.json"


@pytest.mark.parametrize("symlinks_supported", [True, False], ids=["symlink", "copy"])
def test_hub_download_does_not_write_through_a_planted_dangling_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, symlinks_supported: bool
) -> None:
    # Given
    payload = b"a verified configuration"
    commit = "a" * 40
    cache_dir = tmp_path / "cache"
    outside_target = tmp_path / "outside.json"
    snapshot_file = _snapshot_file(cache_dir, commit)
    snapshot_file.parent.mkdir(parents=True)
    snapshot_file.symlink_to(outside_target)
    monkeypatch.setattr(mirror, "_BACKEND", DownloadRepository(payload, commit))
    _set_symlink_support(monkeypatch, symlinks_supported)
    expected_bytes = payload
    expected_cache_root = cache_dir.resolve()

    # When
    path = mirror.hf_hub_download(
        "example/model", "config.json", revision=commit, cache_dir=cache_dir
    )

    # Then
    assert not outside_target.exists()
    assert Path(path).read_bytes() == expected_bytes
    assert Path(path).resolve().is_relative_to(expected_cache_root)


@pytest.mark.parametrize("symlinks_supported", [True, False], ids=["symlink", "copy"])
def test_hub_download_replaces_a_planted_snapshot_file_with_the_verified_blob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, symlinks_supported: bool
) -> None:
    # Given
    payload = b"GOOD"
    commit = "a" * 40
    cache_dir = tmp_path / "cache"
    snapshot_file = _snapshot_file(cache_dir, commit)
    snapshot_file.parent.mkdir(parents=True)
    snapshot_file.write_bytes(b"EVIL")
    monkeypatch.setattr(mirror, "_BACKEND", DownloadRepository(payload, commit))
    _set_symlink_support(monkeypatch, symlinks_supported)
    expected_bytes = b"GOOD"

    # When
    path = mirror.hf_hub_download(
        "example/model", "config.json", revision=commit, cache_dir=cache_dir
    )

    # Then
    assert Path(path).read_bytes() == expected_bytes


@pytest.mark.parametrize("symlinks_supported", [True, False], ids=["symlink", "copy"])
def test_hub_download_replaces_a_planted_symlink_to_matching_outside_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, symlinks_supported: bool
) -> None:
    # Given
    payload = b"a verified configuration"
    commit = "a" * 40
    cache_dir = tmp_path / "cache"
    outside_target = tmp_path / "outside.json"
    outside_target.write_bytes(payload)
    snapshot_file = _snapshot_file(cache_dir, commit)
    snapshot_file.parent.mkdir(parents=True)
    snapshot_file.symlink_to(outside_target)
    monkeypatch.setattr(mirror, "_BACKEND", DownloadRepository(payload, commit))
    _set_symlink_support(monkeypatch, symlinks_supported)
    expected_cache_root = cache_dir.resolve()

    # When
    path = mirror.hf_hub_download(
        "example/model", "config.json", revision=commit, cache_dir=cache_dir
    )

    # Then
    assert Path(path).resolve().is_relative_to(expected_cache_root)


@pytest.mark.parametrize("symlinks_supported", [True, False], ids=["symlink", "copy"])
def test_repeated_hub_download_keeps_the_existing_snapshot_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, symlinks_supported: bool
) -> None:
    # Given
    payload = b"a verified configuration"
    commit = "a" * 40
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(mirror, "_BACKEND", DownloadRepository(payload, commit))
    _set_symlink_support(monkeypatch, symlinks_supported)
    first_path = mirror.hf_hub_download(
        "example/model", "config.json", revision=commit, cache_dir=cache_dir
    )
    expected_inode = os.lstat(first_path).st_ino

    # When
    second_path = mirror.hf_hub_download(
        "example/model", "config.json", revision=commit, cache_dir=cache_dir
    )

    # Then
    assert os.lstat(second_path).st_ino == expected_inode


@pytest.mark.parametrize("symlinks_supported", [True, False], ids=["symlink", "copy"])
def test_failed_snapshot_replacement_leaves_no_temporary_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, symlinks_supported: bool
) -> None:
    # Given
    commit = "a" * 40
    cache_dir = tmp_path / "cache"
    snapshot_file = _snapshot_file(cache_dir, commit)
    snapshot_file.mkdir(parents=True)
    monkeypatch.setattr(mirror, "_BACKEND", DownloadRepository(b"payload", commit))
    _set_symlink_support(monkeypatch, symlinks_supported)
    expected_entries = ["config.json"]

    # When
    with pytest.raises(IsADirectoryError):
        mirror.hf_hub_download(
            "example/model", "config.json", revision=commit, cache_dir=cache_dir
        )

    # Then
    assert [path.name for path in snapshot_file.parent.iterdir()] == expected_entries


def _download_file_offline(cache_dir: Path, local_dir: Path) -> str:
    return mirror.hf_hub_download(
        "example/model",
        "config.json",
        revision="main",
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=True,
    )


def _download_snapshot_offline(cache_dir: Path, local_dir: Path) -> str:
    return mirror.snapshot_download(
        "example/model",
        revision="main",
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=True,
    )


@pytest.mark.parametrize(
    "download_offline",
    [_download_file_offline, _download_snapshot_offline],
    ids=["hf_hub_download", "snapshot_download"],
)
def test_local_download_rejects_a_ref_that_is_not_a_commit(
    tmp_path: Path, download_offline: Callable[[Path, Path], str]
) -> None:
    # Given
    cache_dir = tmp_path / "cache"
    local_dir = tmp_path / "local"
    repo_root = cache_dir / "models--example--model"
    (repo_root / "snapshots").mkdir(parents=True)
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("../../../outside")
    outside_file = tmp_path / "outside" / "config.json"
    outside_file.parent.mkdir()
    outside_file.write_bytes(b"outside the cache")

    # When
    with pytest.raises(errors.LocalEntryNotFoundError):
        download_offline(cache_dir, local_dir)

    # Then
    assert not (local_dir / "config.json").exists()


@pytest.mark.parametrize(
    "digest",
    [
        "A" * 64,
        "../" + "a" * 61,
    ],
)
def test_validate_manifest_rejects_non_lowercase_sha256_digests(digest: str) -> None:
    # Given
    commit = "a" * 40
    manifest = {
        "repo_type": "model",
        "repo_id": "example/model",
        "commit": commit,
        "files": {"config.json": {"blob_sha256": digest, "size": 1}},
    }
    expected_error = "Invalid manifest entry: config.json"

    # When
    try:
        _artifacts._validate_manifest(manifest, "model", "example/model", commit)
    except ValueError as error:
        actual_error = str(error)
    else:
        actual_error = None

    # Then
    assert actual_error == expected_error


def test_validate_manifest_accepts_lowercase_sha256_digest() -> None:
    # Given
    commit = "a" * 40
    digest = "0123456789abcdef" * 4
    manifest = {
        "repo_type": "model",
        "repo_id": "example/model",
        "commit": commit,
        "files": {"config.json": {"blob_sha256": digest, "size": 1}},
    }
    expected_manifest = manifest

    # When
    actual_manifest = _artifacts._validate_manifest(
        manifest, "model", "example/model", commit
    )

    # Then
    assert actual_manifest == expected_manifest


def test_hub_download_verifies_and_reuses_cached_blob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    payload = b"a mirrored configuration"
    commit = "a" * 40
    repository = DownloadRepository(payload, commit)
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    expected_download_count = 1

    # When
    first_path = mirror.hf_hub_download(
        "example/model",
        "config.json",
        revision=commit,
        cache_dir=tmp_path,
    )
    second_path = mirror.hf_hub_download(
        "example/model",
        "config.json",
        revision=commit,
        cache_dir=tmp_path,
    )

    # Then
    assert first_path == second_path
    assert Path(first_path).read_bytes() == payload
    assert repository.download_count == expected_download_count


def test_hub_download_rejects_corrupt_blob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    payload = b"the expected bytes"
    commit = "a" * 40
    repository = DownloadRepository(payload, commit)
    repository.download_payload = b"different bytes"
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    expected_error = "Integrity check failed"

    # When
    with pytest.raises(OSError) as raised:
        mirror.hf_hub_download(
            "example/model",
            "config.json",
            revision=commit,
            cache_dir=tmp_path,
        )

    # Then
    assert expected_error in str(raised.value)


@pytest.mark.parametrize(
    ("allow_hub_fallback", "repo_id", "expected_source"),
    [
        (True, "example/model", "mirror"),
        (True, "other/model", "hub"),
        (True, "legacy-id", "hub"),
        (False, "other/model", "mirror"),
    ],
)
def test_hub_download_falls_back_only_for_repositories_absent_from_mirror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    allow_hub_fallback: bool,
    repo_id: str,
    expected_source: str,
) -> None:
    # Given
    payload = b"a mirrored configuration"
    repository = DownloadRepository(payload, "a" * 40)
    hub_calls: list[tuple[str, str]] = []

    def hub_download(repo_id: str, filename: str, **kwargs: object) -> str:
        hub_calls.append((repo_id, filename))
        return "hub-path"

    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", allow_hub_fallback)
    monkeypatch.setitem(mirror._UPSTREAM, "hf_hub_download", hub_download)

    # When
    path = mirror.hf_hub_download(
        repo_id, "config.json", revision="main", cache_dir=tmp_path
    )

    # Then
    source = "hub" if hub_calls else "mirror"
    assert source == expected_source
    if source == "mirror":
        assert Path(path).read_bytes() == payload


def test_cached_ref_encodes_traversal_components(
    tmp_path: Path,
) -> None:
    # Given
    commit = "a" * 40
    revision = "../../../target"
    expected_ref = tmp_path / "refs" / "%2E%2E" / "%2E%2E" / "%2E%2E" / "target"
    escaped_target = tmp_path.parent / "target"

    # When
    _cache._write_cached_ref(_cache._CacheLayout(tmp_path), revision, commit)

    # Then
    assert expected_ref.read_text() == commit
    assert not escaped_target.exists()


def test_cached_ref_preserves_legitimate_slash_revisions(tmp_path: Path) -> None:
    # Given
    commit = "a" * 40
    revision = "refs/pr/123"
    expected_ref = tmp_path / "refs" / "refs" / "pr" / "123"

    # When
    layout = _cache._CacheLayout(tmp_path)
    _cache._write_cached_ref(layout, revision, commit)
    actual_commit = _cache._cached_commit(layout, "example/model", revision)

    # Then
    assert expected_ref.read_text() == commit
    assert actual_commit == commit


def _manifest(s3: FakeS3, repo_dir: str) -> dict:
    key = f"hf_hub/repos/{repo_dir}/snapshots/{COMMIT}/manifest.json"
    return json.loads(s3.objects[key])


def test_publisher_requires_an_explicit_revision() -> None:
    # Given
    path = "hf://datasets/example/data"
    expected_error = "Specify a revision"

    # When
    with pytest.raises(ValueError) as raised:
        publish.parse_hf_path(path, repo_type=None, revision=None)

    # Then
    assert expected_error in str(raised.value)


def test_publisher_writes_immutable_manifest_and_ref(fake_hub) -> None:
    # Given
    payload = b"mirrored data"
    digest = hashlib.sha256(payload).hexdigest()
    fake_hub.files["data.parquet"] = payload
    expected_ref_key = "hf_hub/repos/datasets/example/data/refs/main.json"
    expected_blob_key = f"hf_hub/blobs/sha256/{digest[:2]}/{digest}"

    # When
    receipt = fake_hub.run("dataset", "hf://datasets/example/data@main")

    # Then
    manifest = _manifest(fake_hub.s3, "datasets/example/data")
    ref = json.loads(fake_hub.s3.objects[expected_ref_key])
    assert fake_hub.s3.objects[expected_blob_key] == payload
    assert manifest["files"] == {
        "data.parquet": {"blob_sha256": digest, "size": len(payload)}
    }
    assert ref["commit"] == COMMIT
    assert receipt["uploaded_blob_count"] == 1


def test_tokenizer_publishes_only_tokenizer_config_and_code_files(fake_hub) -> None:
    # Given
    fake_hub.files.update(
        {
            "config.json": b"{}",
            "tokenizer.json": b"{}",
            "tokenizer_config.json": b"{}",
            "tokenization_custom.py": b"pass",
            "model-00001-of-00002.safetensors": b"weights",
            "model.safetensors.index.json": b"{}",
            "README.md": b"card",
        }
    )
    expected_files = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenization_custom.py",
    }

    # When
    fake_hub.run("tokenizer", f"example/model@{COMMIT}")

    # Then
    assert set(_manifest(fake_hub.s3, "models/example/model")["files"]) == (
        expected_files
    )


def test_later_publish_extends_the_commit_manifest(fake_hub) -> None:
    # Given
    fake_hub.files.update(
        {"config.json": b"{}", "model.safetensors": b"weights", "README.md": b"card"}
    )
    fake_hub.run("tokenizer", f"example/model@{COMMIT}")

    # When
    receipt = fake_hub.run(
        "model", f"example/model@{COMMIT}", "--ignore-patterns", "README.md"
    )

    # Then
    manifest = _manifest(fake_hub.s3, "models/example/model")
    assert set(manifest["files"]) == {"config.json", "model.safetensors"}
    assert receipt["file_count"] == 2
    assert receipt["manifest_file_count"] == 2
    assert receipt["uploaded_blob_count"] == 1


def test_publish_refuses_to_change_a_published_file(fake_hub) -> None:
    # Given
    fake_hub.files["config.json"] = b"{}"
    fake_hub.run("tokenizer", f"example/model@{COMMIT}")
    fake_hub.files["config.json"] = b'{"changed": true}'
    expected_error = "Refusing to change 'config.json'"

    # When
    with pytest.raises(RuntimeError) as raised:
        fake_hub.run("model", f"example/model@{COMMIT}")

    # Then
    assert expected_error in str(raised.value)


def test_publish_rejects_a_dataset_path_for_a_tokenizer(fake_hub) -> None:
    # Given
    path = f"hf://datasets/example/data@{COMMIT}"
    expected_error = "is not a model repository"

    # When
    with pytest.raises(ValueError) as raised:
        fake_hub.run("tokenizer", path)

    # Then
    assert expected_error in str(raised.value)


def test_publish_rejects_an_existing_blob_with_different_bytes(fake_hub) -> None:
    # Given
    payload = b"mirrored data"
    digest = hashlib.sha256(payload).hexdigest()
    blob_key = f"hf_hub/blobs/sha256/{digest[:2]}/{digest}"
    fake_hub.files["data.parquet"] = payload
    fake_hub.s3.objects[blob_key] = b"X" * len(payload)
    fake_hub.s3.metadata[blob_key] = {"sha256": digest}
    expected_error = f"s3://example-bucket/{blob_key}"
    expected_objects = dict(fake_hub.s3.objects)

    # When
    with pytest.raises(RuntimeError) as raised:
        fake_hub.run("dataset", "hf://datasets/example/data@main")

    # Then
    assert expected_error in str(raised.value)
    assert fake_hub.s3.objects == expected_objects


@pytest.mark.parametrize(
    "mismatch", [{"sha256": "0" * 64}, {"size": 0}], ids=["sha256", "size"]
)
def test_publish_rejects_a_download_that_differs_from_upstream_lfs(
    fake_hub, mismatch: dict[str, object]
) -> None:
    # Given
    payload = b"weights"
    matching = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    fake_hub.files["model.safetensors"] = payload
    fake_hub.lfs["model.safetensors"] = SimpleNamespace(**{**matching, **mismatch})
    expected_error = "Downloaded 'model.safetensors' does not match"

    # When
    with pytest.raises(RuntimeError) as raised:
        fake_hub.run("model", f"example/model@{COMMIT}")

    # Then
    assert expected_error in str(raised.value)
    assert fake_hub.s3.objects == {}


def test_publish_accepts_a_download_that_matches_upstream_lfs(fake_hub) -> None:
    # Given
    payload = b"weights"
    digest = hashlib.sha256(payload).hexdigest()
    fake_hub.files["model.safetensors"] = payload
    fake_hub.lfs["model.safetensors"] = SimpleNamespace(
        sha256=digest, size=len(payload)
    )
    expected_blob_key = f"hf_hub/blobs/sha256/{digest[:2]}/{digest}"

    # When
    receipt = fake_hub.run("model", f"example/model@{COMMIT}")

    # Then
    assert fake_hub.s3.objects[expected_blob_key] == payload
    assert receipt["uploaded_blob_count"] == 1


def test_download_rejects_manifest_traversal_before_touching_filesystem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    commit = "a" * 40
    digest = "../../../" + "v" * 55
    manifest = {
        "schema_version": 1,
        "repo_type": "model",
        "repo_id": "example/model",
        "commit": commit,
        "files": {"config.json": {"blob_sha256": digest, "size": 2}},
    }
    manifest_bytes = json.dumps(manifest).encode()
    cache = tmp_path / "cache"
    outside_lock = tmp_path / ("v" * 55 + ".lock")
    expected_contents = b"This file must not be truncated by a malformed manifest."
    expected_error = "Invalid manifest entry: config.json"
    outside_lock.write_bytes(expected_contents)

    class ManifestTransport:
        def get_object(self, **kwargs: Any) -> dict[str, io.BytesIO]:
            return {"Body": io.BytesIO(manifest_bytes)}

    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: ManifestTransport())
    repository = mirror._S3Repository("audit-fixture-bucket", "audit")
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", False)

    # When
    with pytest.raises(ValueError) as raised:
        mirror.hf_hub_download(
            "example/model", "config.json", revision=commit, cache_dir=cache
        )

    # Then
    assert str(raised.value) == expected_error
    assert outside_lock.read_bytes() == expected_contents
    assert not cache.exists()


def test_snapshot_keeps_the_commit_resolved_before_a_branch_moves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    repository = SnapshotRepository()
    expected_commit = repository.commit
    expected_files = dict(repository.files)
    expected_downloads = [expected_commit] * len(expected_files)
    repository.next_commit = "b" * 40
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", False)
    expected_ref = tmp_path / "models--example--model" / "refs" / "main"

    # When
    snapshot = Path(mirror.snapshot_download("example/model", cache_dir=tmp_path))

    # Then
    assert snapshot.name == expected_commit
    assert snapshot.is_dir()
    assert {
        name: (snapshot / name).read_bytes() for name in expected_files
    } == expected_files
    assert repository.downloads == expected_downloads
    assert expected_ref.read_text() == expected_commit


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize(
    "operation", [mirror.hf_hub_download, mirror.snapshot_download]
)
def test_local_only_reads_never_contact_s3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation, fallback: bool
) -> None:
    # Given
    repository = SnapshotRepository()
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", False)
    mirror.hf_hub_download("example/model", "config.json", cache_dir=tmp_path)
    monkeypatch.setattr(mirror, "_BACKEND", None)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", fallback)
    args = (
        ("example/model", "config.json")
        if operation is mirror.hf_hub_download
        else ("example/model",)
    )
    expected_payload = repository.files["config.json"]
    expected_file = (
        tmp_path
        / "models--example--model"
        / "snapshots"
        / repository.commit
        / "config.json"
    )

    # When
    result = Path(operation(*args, cache_dir=tmp_path, local_files_only=True))

    # Then
    cached_file = result / "config.json" if result.is_dir() else result
    assert cached_file == expected_file
    assert cached_file.read_bytes() == expected_payload


def test_snapshot_honors_force_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    repository = SnapshotRepository()
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", False)
    mirror.snapshot_download("example/model", cache_dir=tmp_path)
    expected_download_count = 2 * len(repository.files)

    # When
    mirror.snapshot_download("example/model", cache_dir=tmp_path, force_download=True)

    # Then
    assert len(repository.downloads) == expected_download_count


def test_snapshot_accepts_directory_allow_patterns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    repository = SnapshotRepository()
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", False)
    expected_files = ["data/part.txt"]

    # When
    snapshot = Path(
        mirror.snapshot_download(
            "example/model", cache_dir=tmp_path, allow_patterns="data/"
        )
    )

    # Then
    assert snapshot.is_dir()
    assert (
        sorted(
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file()
        )
        == expected_files
    )
