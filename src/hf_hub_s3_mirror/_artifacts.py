"""Shared immutable artifact format and validation for publisher and reader."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal, TypedDict, cast

RepoType = Literal["model", "dataset"]


class FileEntry(TypedDict):
    blob_sha256: str
    size: int


class Manifest(TypedDict):
    schema_version: int
    repo_type: RepoType
    repo_id: str
    commit: str
    files: dict[str, FileEntry]


_REPO_DIRECTORIES: dict[RepoType, str] = {"model": "models", "dataset": "datasets"}
_DEFAULT_MIRROR_PREFIX = "hf_hub"
_DEFAULT_REVISION = "main"
_COMMIT_LENGTH = 40
_SHA256_LENGTH = 64
_HASH_CHUNK_SIZE = 8 * 1024**2


def _sha256_stream(source: BinaryIO) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    while chunk := source.read(_HASH_CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _sha256_file(path: str | Path) -> tuple[str, int]:
    with open(path, "rb") as source:
        return _sha256_stream(source)


def _is_commit(value: str) -> bool:
    return len(value) == _COMMIT_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_repo_type(repo_type: str | None) -> RepoType:
    value = repo_type or "model"
    if value not in _REPO_DIRECTORIES:
        raise ValueError(f"Unsupported repo_type: {repo_type!r}")
    return cast(RepoType, value)


def _validate_repo_id(repo_id: str) -> str:
    parts = repo_id.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("repo_id must have the form namespace/name")
    return repo_id


def _validate_repo_path(path: str) -> str:
    if not path or path.startswith("/") or "\\" in path:
        raise ValueError("Repository paths must be relative POSIX paths")
    if any(part in {"", ".", ".."} for part in PurePosixPath(path).parts):
        raise ValueError("Invalid repository path")
    return path


def _validate_manifest(
    value: dict[str, Any], repo_type: RepoType, repo_id: str, commit: str
) -> Manifest:
    if (value.get("repo_type"), value.get("repo_id"), value.get("commit")) != (
        repo_type,
        repo_id,
        commit,
    ):
        raise ValueError("Manifest identity does not match its S3 location")
    files = value.get("files")
    if not isinstance(files, dict):
        raise ValueError("Invalid manifest files")  # noqa: TRY004 - malformed external data.
    for path, entry in files.items():
        if not isinstance(path, str) or not isinstance(entry, Mapping):
            raise ValueError("Invalid manifest entry")  # noqa: TRY004 - malformed external data.
        _validate_repo_path(path)
        digest, size = entry.get("blob_sha256"), entry.get("size")
        if (
            not isinstance(digest, str)
            or len(digest) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"Invalid manifest entry: {path}")
    return cast(Manifest, value)
