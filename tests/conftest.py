"""Shared fixtures for the public contract tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _support import (
    COMMIT,
    NETWORK_ENTRY_POINTS,
    BlobRepository,
    FakeS3,
    FallbackRepository,
    ForbiddenHubApi,
    HubApi,
    Observed,
    RoutingRepository,
    _forbidden,
    _recording_hub_api,
)

from hf_hub_s3_mirror import mirror, publish


@pytest.fixture
def observed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Observed:
    # Given
    observed = Observed(RoutingRepository(), [], [])
    monkeypatch.setattr(mirror, "_BACKEND", observed.repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", True)
    monkeypatch.setitem(
        mirror._UPSTREAM, "HfApi", _recording_hub_api(observed.hub_calls)
    )
    for name in ("hf_hub_download", "snapshot_download"):
        monkeypatch.setitem(
            mirror._UPSTREAM, name, _forbidden(name, observed.hub_calls)
        )
    for target in NETWORK_ENTRY_POINTS:
        monkeypatch.setattr(target, _forbidden(target, observed.network_calls))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub-cache"))
    return observed


@pytest.fixture
def filesystem(observed: Observed) -> Any:
    # Given
    instance = mirror._filesystem_class()(token="dummy-token")
    instance._api = ForbiddenHubApi(observed.hub_calls)
    return instance


@pytest.fixture
def configured_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FallbackRepository:
    # Given
    repository = FallbackRepository()
    HubApi.calls = []
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", True)
    monkeypatch.setitem(mirror._UPSTREAM, "HfApi", HubApi)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub-cache"))
    return repository


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Serve `files` as one upstream commit and publish into a fake bucket."""
    s3 = FakeS3()
    files: dict[str, bytes] = {}
    lfs: dict[str, SimpleNamespace] = {}

    class FakeHfApi:
        def __init__(self, token: str | None) -> None:
            pass

        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            siblings = [
                SimpleNamespace(rfilename=name, lfs=lfs.get(name)) for name in files
            ]
            return SimpleNamespace(sha=COMMIT, siblings=siblings)

        def list_repo_files(self, *args: object, **kwargs: object) -> list[str]:
            return list(files)

    def download(repo_id: str, filename: str, **kwargs: object) -> str:
        path = tmp_path / "hub" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[filename])
        return str(path)

    monkeypatch.setattr(publish, "HfApi", FakeHfApi)
    monkeypatch.setattr(publish, "hf_hub_download", download)
    monkeypatch.setattr(publish.boto3, "client", lambda *args, **kwargs: s3)

    def run(*argv: str) -> dict:
        return publish.publish(
            publish.parse_args([*argv, "--bucket", "example-bucket"])
        )

    return SimpleNamespace(s3=s3, files=files, lfs=lfs, run=run)


@pytest.fixture
def fake_s3() -> BlobRepository:
    return BlobRepository(expected_payload=b"GOOD", object_payload=b"GOOD")
