from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hf_hub_s3_mirror import mirror


class FakeRepository:
    def __init__(self, payload: bytes, commit: str) -> None:
        self.expected_payload = payload
        self.download_payload = payload
        self.commit = commit
        self.download_count = 0
        self.mirrored_repo_ids = {"example/model"}

    def has_repository(self, repo_type: str, repo_id: str) -> bool:
        return repo_id in self.mirrored_repo_ids

    def resolve(self, repo_type: str, repo_id: str, revision: str) -> str:
        return self.commit

    def entry(
        self, repo_type: str, repo_id: str, commit: str, filename: str
    ) -> dict[str, str | int]:
        return {
            "blob_sha256": hashlib.sha256(self.expected_payload).hexdigest(),
            "size": len(self.expected_payload),
        }

    def download(
        self,
        repo_type: str,
        repo_id: str,
        commit: str,
        filename: str,
        destination: str,
    ) -> None:
        self.download_count += 1
        Path(destination).write_bytes(self.download_payload)


class FilesystemRepository:
    def __init__(self, commit: str, files: dict[str, dict[str, str | int]]) -> None:
        self.commit = commit
        self.files = files

    def resolve(self, repo_type: str, repo_id: str, revision: str) -> str:
        return self.commit

    def manifest(self, repo_type: str, repo_id: str, commit: str) -> dict[str, object]:
        return {"files": self.files}


def test_hub_download_verifies_and_reuses_cached_blob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    payload = b"a mirrored configuration"
    commit = "a" * 40
    repository = FakeRepository(payload, commit)
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
    repository = FakeRepository(payload, commit)
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
    repository = FakeRepository(payload, "a" * 40)
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


def test_api_routes_repositories_absent_from_mirror_to_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    repository = FakeRepository(b"", "a" * 40)

    class HubApi:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.token = kwargs.get("token")

        def dataset_info(self, repo_id: str, **kwargs: object) -> str:
            return f"hub:{repo_id}:{self.token}"

        def whoami(self) -> str:
            return "hub-user"

    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", True)
    monkeypatch.setitem(mirror._UPSTREAM, "HfApi", HubApi)
    api = mirror.S3HfApi(token="secret")

    # When
    info = api.dataset_info("other/data", revision="main")
    user = api.whoami()

    # Then
    assert info == "hub:other/data:secret"
    assert user == "hub-user"


def test_install_with_fallback_keeps_the_hub_endpoint_in_fresh_process() -> None:
    # Given
    script = """
import os
from unittest.mock import patch
import hf_hub_s3_mirror
from hf_hub_s3_mirror import mirror
with patch('boto3.client', return_value=object()):
    hf_hub_s3_mirror.install(bucket='example-bucket')
print(mirror._HUB_FALLBACK, os.environ.get('HF_ENDPOINT'))
"""
    expected_output = "True None"
    environment = {
        key: value for key, value in os.environ.items() if key != "HF_ENDPOINT"
    }
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        HF_S3_MIRROR_ALLOW_HUB_FALLBACK="1",
    )

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output


def test_snapshot_download_uses_cached_snapshot_when_local_files_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    payload = b"an already cached file"
    commit = "a" * 40
    repository = FakeRepository(payload, commit)
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    mirror.hf_hub_download(
        "example/model", "config.json", revision="main", cache_dir=tmp_path
    )
    monkeypatch.setattr(mirror, "_BACKEND", None)
    expected_snapshot = (
        tmp_path / "models--example--model" / "snapshots" / commit
    ).absolute()

    # When
    actual_snapshot = mirror.snapshot_download(
        "example/model",
        revision="main",
        cache_dir=tmp_path,
        local_files_only=True,
    )

    # Then
    assert Path(actual_snapshot) == expected_snapshot
    assert (Path(actual_snapshot) / "config.json").read_bytes() == payload


def test_cached_ref_encodes_traversal_components(
    tmp_path: Path,
) -> None:
    # Given
    commit = "a" * 40
    revision = "../../../target"
    expected_ref = tmp_path / "refs" / "%2E%2E" / "%2E%2E" / "%2E%2E" / "target"
    escaped_target = tmp_path.parent / "target"

    # When
    mirror._write_cached_ref(mirror._CacheLayout(tmp_path), revision, commit)

    # Then
    assert expected_ref.read_text() == commit
    assert not escaped_target.exists()


def test_cached_ref_preserves_legitimate_slash_revisions(tmp_path: Path) -> None:
    # Given
    commit = "a" * 40
    revision = "refs/pr/123"
    expected_ref = tmp_path / "refs" / "refs" / "pr" / "123"

    # When
    layout = mirror._CacheLayout(tmp_path)
    mirror._write_cached_ref(layout, revision, commit)
    actual_commit = mirror._cached_commit(layout, "example/model", revision)

    # Then
    assert expected_ref.read_text() == commit
    assert actual_commit == commit


def test_filesystem_glob_does_not_match_nested_files_for_single_star(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    commit = "a" * 40
    files = {
        "top.txt": {"blob_sha256": "a" * 64, "size": 1},
        "dir/nested.txt": {"blob_sha256": "b" * 64, "size": 1},
    }
    monkeypatch.setattr(mirror, "_BACKEND", FilesystemRepository(commit, files))
    filesystem = mirror._filesystem_class()()
    pattern = "example/model@main/*"
    expected_matches = [f"example/model@{commit}/top.txt"]

    # When
    actual_matches = filesystem.glob(pattern)

    # Then
    assert actual_matches == expected_matches


def test_filesystem_glob_matches_an_encoded_slash_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    commit = "a" * 40
    files = {"top.txt": {"blob_sha256": "a" * 64, "size": 1}}
    monkeypatch.setattr(mirror, "_BACKEND", FilesystemRepository(commit, files))
    filesystem = mirror._filesystem_class()()
    pattern = "example/model@feature%2Ffoo/*"
    expected_matches = [f"example/model@{commit}/top.txt"]

    # When
    actual_matches = filesystem.glob(pattern)

    # Then
    assert actual_matches == expected_matches


def test_filesystem_ls_returns_a_file_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    commit = "a" * 40
    files = {"top.txt": {"blob_sha256": "a" * 64, "size": 1}}
    monkeypatch.setattr(mirror, "_BACKEND", FilesystemRepository(commit, files))
    filesystem = mirror._filesystem_class()()
    expected_listing = [
        {"name": f"example/model@{commit}/top.txt", "type": "file", "size": 1}
    ]

    # When
    actual_listing = filesystem.ls("example/model@main/top.txt")

    # Then
    assert actual_listing == expected_listing


def test_filesystem_ls_raises_for_a_missing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    commit = "a" * 40
    files = {"top.txt": {"blob_sha256": "a" * 64, "size": 1}}
    monkeypatch.setattr(mirror, "_BACKEND", FilesystemRepository(commit, files))
    filesystem = mirror._filesystem_class()()
    missing_path = "example/model@main/missing.txt"

    # When
    with pytest.raises(FileNotFoundError) as raised:
        filesystem.ls(missing_path)

    # Then
    assert raised.value.args == (missing_path,)


def test_import_is_inert_until_explicit_install() -> None:
    # Given
    script = (
        "import sys; import hf_hub_s3_mirror; print('huggingface_hub' in sys.modules)"
    )
    expected_output = "False"
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output


def test_install_resolves_a_callable_bucket_on_first_access_in_fresh_process() -> None:
    # Given
    script = """
from unittest.mock import patch
import hf_hub_s3_mirror
from hf_hub_s3_mirror import mirror
calls = []
def bucket():
    calls.append(1)
    return 'example-bucket'
with patch('boto3.client', return_value=object()):
    hf_hub_s3_mirror.install(bucket=bucket, allow_hub_fallback=True)
before = len(calls)
print(before, mirror._BACKEND.bucket, mirror._BACKEND.bucket, len(calls))
"""
    expected_output = "0 example-bucket example-bucket 1"
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output


def test_install_patches_hub_in_fresh_process() -> None:
    # Given
    script = """
from unittest.mock import patch
import hf_hub_s3_mirror
with patch('boto3.client', return_value=object()):
    hf_hub_s3_mirror.install(bucket='example-bucket')
from huggingface_hub import HfApi
from hf_hub_s3_mirror.mirror import S3HfApi
print(HfApi is S3HfApi)
"""
    expected_output = "True"
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output


def test_install_rebinds_module_level_api_helpers_in_fresh_process() -> None:
    # Given
    script = """
from unittest.mock import patch
import hf_hub_s3_mirror
with patch('boto3.client', return_value=object()):
    hf_hub_s3_mirror.install(bucket='example-bucket')
import huggingface_hub
import huggingface_hub.hf_api
from hf_hub_s3_mirror.mirror import S3HfApi
helpers = ('model_info', 'list_repo_files')
print(all(
    isinstance(getattr(module, name).__self__, S3HfApi)
    for module in (huggingface_hub, huggingface_hub.hf_api)
    for name in helpers
))
"""
    expected_output = "True"
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output
