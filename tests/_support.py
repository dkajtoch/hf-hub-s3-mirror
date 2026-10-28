"""Shared in-memory backends and call recording for contract tests."""

from __future__ import annotations

import hashlib
import inspect
import io
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from huggingface_hub.hf_api import HfApi as UpstreamHfApi

from hf_hub_s3_mirror import mirror

COMMIT = "a" * 40


PAYLOAD = b"only mirror bytes"


MODEL_ID = "example/model"


DATASET_ID = "example/dataset"


FILE = f"{MODEL_ID}/config.json"


NEW_FILE = f"{MODEL_ID}/new.json"


NEW_DIRECTORY = f"{MODEL_ID}/new-directory"


IMMUTABLE_FILE = f"{MODEL_ID}@{COMMIT}/config.json"


FILE_INFO = {"name": IMMUTABLE_FILE, "size": len(PAYLOAD), "type": "file"}


NETWORK_ENTRY_POINTS = (
    "httpx.Client.send",
    "socket.socket.connect",
    "huggingface_hub.hf_file_system.http_get",
    "huggingface_hub.hf_file_system.http_backoff",
    "huggingface_hub.hf_file_system.http_stream_backoff",
)


UPSTREAM_API_METHODS = sorted(
    name
    for name, _ in inspect.getmembers(UpstreamHfApi, inspect.isfunction)
    if not name.startswith("_")
)


class RoutingRepository:
    def __init__(self) -> None:
        self.mirrored = {("model", MODEL_ID), ("dataset", DATASET_ID)}
        self.files = {
            "config.json": {
                "blob_sha256": hashlib.sha256(PAYLOAD).hexdigest(),
                "size": len(PAYLOAD),
            }
        }
        self.blob_downloads: list[str] = []

    def has_repository(self, repo_type: str, repo_id: str) -> bool:
        return (repo_type, repo_id) in self.mirrored

    def resolve(self, repo_type: str, repo_id: str, revision: str | None) -> str:
        if not self.has_repository(repo_type, repo_id):
            raise mirror._hub_error("repo", "not mirrored")
        if revision not in {None, "main", COMMIT}:
            raise mirror._hub_error("revision", "missing revision")
        return COMMIT

    def manifest(
        self, repo_type: str, repo_id: str, commit: str
    ) -> dict[str, dict[str, dict[str, str | int]]]:
        return {"files": self.files}

    def entry(
        self, repo_type: str, repo_id: str, commit: str, path: str
    ) -> dict[str, str | int]:
        try:
            return self.files[path]
        except KeyError as error:
            raise mirror._hub_error("entry", "missing file", error) from error

    def download(
        self,
        repo_type: str,
        repo_id: str,
        commit: str,
        path: str,
        destination: str,
    ) -> None:
        self.entry(repo_type, repo_id, commit, path)
        self.blob_downloads.append(path)
        Path(destination).write_bytes(PAYLOAD)


class ForbiddenHubApi:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __getattr__(self, name: str) -> Any:
        self.calls.append(f"HfFileSystem._api.{name}")
        raise AssertionError(f"HfFileSystem._api.{name} used for a mirrored repo")


def _forbidden(name: str, calls: list[str]) -> Callable[..., Any]:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        raise AssertionError(f"{name} must not run for a mirrored repository")

    return forbidden


def _recording_hub_api(calls: list[str]) -> type:
    def recorded(name: str) -> Callable[..., Any]:
        @wraps(getattr(UpstreamHfApi, name))
        def method(self: object, *args: Any, **kwargs: Any) -> str:
            calls.append(f"HfApi.{name}")
            if name not in mirror._SAFE_HUB_API_METHODS:
                raise AssertionError(f"upstream HfApi.{name} must not run")
            return f"hub {name}"

        return method

    def construct(self: object, *args: Any, **kwargs: Any) -> None:
        calls.append("HfApi()")

    methods = {name: recorded(name) for name in UPSTREAM_API_METHODS}
    return type("RecordingHubApi", (), {**methods, "__init__": construct})


@dataclass
class Observed:
    repository: RoutingRepository
    hub_calls: list[str]
    network_calls: list[str]


class FallbackRepository:
    def __init__(self) -> None:
        self.commit = "a" * 40
        self.payload = b"only mirror bytes"
        self.files = {
            "config.json": {
                "blob_sha256": hashlib.sha256(self.payload).hexdigest(),
                "size": len(self.payload),
            }
        }

    def has_repository(self, repo_type: str, repo_id: str) -> bool:
        return repo_id == "example/model"

    def resolve(self, repo_type: str, repo_id: str, revision: str | None) -> str:
        if not self.has_repository(repo_type, repo_id):
            raise mirror._hub_error("repo", "not mirrored")
        if revision not in {None, "main", self.commit}:
            raise mirror._hub_error("revision", "missing revision")
        return self.commit

    def manifest(
        self, repo_type: str, repo_id: str, commit: str
    ) -> dict[str, dict[str, dict[str, str | int]]]:
        return {"files": self.files}

    def entry(
        self, repo_type: str, repo_id: str, commit: str, path: str
    ) -> dict[str, str | int]:
        try:
            return self.files[path]
        except KeyError as error:
            raise mirror._hub_error("entry", "missing file", error) from error

    def read(
        self,
        repo_type: str,
        repo_id: str,
        commit: str,
        path: str,
        start: int | None = None,
        end: int | None = None,
    ) -> io.BytesIO:
        self.entry(repo_type, repo_id, commit, path)
        return io.BytesIO(self.payload[slice(start, end)])

    def download(
        self,
        repo_type: str,
        repo_id: str,
        commit: str,
        path: str,
        destination: str,
    ) -> None:
        self.entry(repo_type, repo_id, commit, path)
        Path(destination).write_bytes(self.payload)


class HubApi:
    calls: list[tuple[str, str]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.token = kwargs.get("token")

    def whoami(self) -> str:
        return "hub-user"

    def file_exists(self, repo_id: str, filename: str, **kwargs: object) -> bool:
        self.calls.append((repo_id, filename))
        return repo_id == "other/model"

    def auth_check(self, repo_id: str, **kwargs: object) -> None:
        self.calls.append((repo_id, "auth_check"))

    def upload_file(
        self, path_or_fileobj: object, path_in_repo: str, repo_id: str, **kwargs: object
    ) -> str:
        self.calls.append((repo_id, path_in_repo))
        return "uploaded"


class DownloadRepository:
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


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {
            "ContentLength": len(self.objects[Key]),
            "Metadata": self.metadata.get(Key, {}),
        }

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, **kwargs: object
    ) -> None:
        self.objects[Key] = Body

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict,
    ) -> None:
        self.objects[key] = Path(filename).read_bytes()
        self.metadata[key] = ExtraArgs["Metadata"]


class SnapshotRepository:
    def __init__(self) -> None:
        self.commit = "a" * 40
        self.next_commit = self.commit
        self.downloads: list[str] = []
        self.files = {"config.json": b"{}", "data/part.txt": b"payload"}

    def has_repository(self, repo_type: str, repo_id: str) -> bool:
        return True

    def resolve(self, repo_type: str, repo_id: str, revision: str) -> str:
        if len(revision) == 40:
            return revision
        commit = self.commit
        self.commit = self.next_commit
        return commit

    def manifest(self, repo_type: str, repo_id: str, commit: str) -> dict:
        return {
            "files": {
                path: {
                    "blob_sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
                for path, data in self.files.items()
            }
        }

    def entry(self, repo_type: str, repo_id: str, commit: str, path: str) -> dict:
        return self.manifest(repo_type, repo_id, commit)["files"][path]

    def download(
        self, repo_type: str, repo_id: str, commit: str, path: str, destination: str
    ) -> None:
        self.downloads.append(commit)
        Path(destination).write_bytes(self.files[path])


class BlobRepository:
    def __init__(self, expected_payload: bytes, object_payload: bytes) -> None:
        self.commit = "a" * 40
        self.expected_payload = expected_payload
        self.object_payload = object_payload
        self.download_count = 0
        self.read_count = 0

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
        Path(destination).write_bytes(self.object_payload)

    def read(self, *args: object, **kwargs: object) -> None:
        self.read_count += 1
        raise AssertionError("Mirrored reads must not expose unverified range bytes")
