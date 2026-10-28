"""Read-only Hugging Face Hub compatibility layer backed by an S3 mirror.

Call :func:`install` before importing datasets or transformers. The adapter is
tested against huggingface_hub 0.36.2, datasets 5.0.1, and transformers 4.57.6.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, BinaryIO, Literal, TypedDict, cast
from urllib.parse import quote, unquote

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


@dataclass(frozen=True)
class _CacheLayout:
    root: Path

    def blob(self, entry: FileEntry) -> Path:
        return self.root / "blobs" / entry["blob_sha256"]

    def snapshot(self, commit: str) -> Path:
        return self.root / "snapshots" / commit

    def file(self, commit: str, filename: str) -> Path:
        return self.snapshot(commit) / filename

    def lock(self, entry: FileEntry) -> Path:
        return self.root / ".locks" / f"{entry['blob_sha256']}.lock"

    def ref(self, revision: str) -> Path:
        return (
            self.root
            / "refs"
            / Path(*(_cache_ref_component(part) for part in revision.split("/")))
        )


_REPO_DIRECTORIES: dict[RepoType, str] = {"model": "models", "dataset": "datasets"}
_DEFAULT_MIRROR_PREFIX = "hf_hub"
_FAIL_CLOSED_HF_ENDPOINT = "http://127.0.0.1:9"
_DEFAULT_REVISION = "main"
_COMMIT_LENGTH = 40
_SHA256_LENGTH = 64
_MEBIBYTE = 1024**2
_HASH_CHUNK_SIZE = 8 * _MEBIBYTE
_RANGE_BLOCK_SIZE = 5 * _MEBIBYTE
_NOT_FOUND_CODES = {"NoSuchKey", "404", "NotFound"}
_ACCESS_DENIED_CODES = {"AccessDenied", "403", "InvalidAccessKeyId", "ExpiredToken"}
_API_HELPERS = (
    "repo_info",
    "model_info",
    "dataset_info",
    "list_repo_tree",
    "list_repo_files",
)
_TRUE_VALUES = {"1", "true", "yes", "on"}
_INSTALLED = False
_BACKEND: _S3Repository | None = None
_HUB_FALLBACK = False
_UPSTREAM: dict[str, Any] = {}


def _sha256_file(path: str | Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as source:
        while chunk := source.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _is_commit(value: str) -> bool:
    return len(value) == _COMMIT_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _read_positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


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


def _cache_ref_component(component: str) -> str:
    if component == ".":
        return "%2E"
    if component == "..":
        return "%2E%2E"
    return quote(component, safe="") or "%2F"


def _hub_error(
    kind: Literal["repo", "revision", "entry", "local"], message: str, cause=None
):
    from huggingface_hub import errors
    from requests import Response

    response = Response()
    response.status_code = 404
    response.url = "s3://private-hugging-face-mirror"
    cls = {
        "repo": errors.RepositoryNotFoundError,
        "revision": errors.RevisionNotFoundError,
        "entry": errors.EntryNotFoundError,
        "local": errors.LocalEntryNotFoundError,
    }[kind]
    error = cls(message) if kind == "local" else cls(message, response=response)
    error.__cause__ = cause
    return error


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
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"Invalid manifest entry: {path}")
    return cast(Manifest, value)


class _S3Repository:
    def __init__(self, bucket: str, prefix: str) -> None:
        import boto3
        from boto3.s3.transfer import TransferConfig

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client("s3", endpoint_url=os.environ.get("HF_S3_ENDPOINT_URL"))
        self.transfer_config = TransferConfig(
            multipart_threshold=_read_positive_int_env(
                "HF_S3_MULTIPART_THRESHOLD_MB", 64
            )
            * _MEBIBYTE,
            multipart_chunksize=_read_positive_int_env("HF_S3_MULTIPART_CHUNK_MB", 32)
            * _MEBIBYTE,
            max_concurrency=_read_positive_int_env("HF_S3_TRANSFER_CONCURRENCY", 8),
            num_download_attempts=5,
            use_threads=True,
            preferred_transfer_client=os.environ.get("HF_S3_TRANSFER_CLIENT", "auto"),
        )

    def key(self, *parts: str) -> str:
        return "/".join(part for part in (self.prefix, *parts) if part)

    def repo_key(self, repo_type: RepoType, repo_id: str, *parts: str) -> str:
        return self.key("repos", _REPO_DIRECTORIES[repo_type], repo_id, *parts)

    @lru_cache(maxsize=1024)  # noqa: B019 - the repository backend is a process singleton.
    def has_repository(self, repo_type: RepoType, repo_id: str) -> bool:
        prefix = self.repo_key(repo_type, repo_id) + "/"
        return bool(
            self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1).get(
                "Contents"
            )
        )

    def json(
        self, key: str, missing_kind: Literal["repo", "revision", "entry"]
    ) -> dict[str, Any]:
        from botocore.exceptions import ClientError

        try:
            return json.loads(
                self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            )
        except ClientError as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code in _NOT_FOUND_CODES:
                raise _hub_error(missing_kind, f"Mirror object not found: {key}", error)
            if code in _ACCESS_DENIED_CODES:
                raise PermissionError(
                    f"S3 access denied: bucket={self.bucket!r}, key={key!r}"
                ) from error
            raise

    def resolve(
        self, repo_type: RepoType, repo_id: str, revision: str | None = None
    ) -> str:
        from huggingface_hub.errors import RevisionNotFoundError

        _validate_repo_id(repo_id)
        revision = revision or _DEFAULT_REVISION
        if _is_commit(revision):
            self.manifest(repo_type, repo_id, revision)
            return revision
        key = self.repo_key(
            repo_type, repo_id, "refs", quote(revision, safe="") + ".json"
        )
        try:
            ref = self.json(key, "revision")
        except RevisionNotFoundError as error:
            prefix = self.repo_key(repo_type, repo_id) + "/"
            exists = self.s3.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix, MaxKeys=1
            ).get("Contents")
            if not exists:
                raise _hub_error(
                    "repo", f"Repository {repo_id!r} is not in the mirror", error
                )
            raise
        if (
            ref.get("repo_type") != repo_type
            or ref.get("repo_id") != repo_id
            or ref.get("revision") != revision
        ):
            raise ValueError("Ref identity does not match its S3 location")
        commit = ref.get("commit")
        if not isinstance(commit, str) or not _is_commit(commit):
            raise ValueError("Ref contains an invalid commit")
        return commit

    @lru_cache(maxsize=128)  # noqa: B019 - the repository backend is a process singleton.
    def manifest(self, repo_type: RepoType, repo_id: str, commit: str) -> Manifest:
        key = self.repo_key(repo_type, repo_id, "snapshots", commit, "manifest.json")
        value = self.json(key, "revision")
        return _validate_manifest(value, repo_type, repo_id, commit)

    def entry(
        self, repo_type: RepoType, repo_id: str, commit: str, path: str
    ) -> FileEntry:
        _validate_repo_path(path)
        try:
            return self.manifest(repo_type, repo_id, commit)["files"][path]
        except KeyError as error:
            raise _hub_error(
                "entry", f"File {path!r} is not in {repo_id}@{commit}", error
            )

    def read(
        self,
        repo_type: RepoType,
        repo_id: str,
        commit: str,
        path: str,
        start: int | None = None,
        end: int | None = None,
    ) -> BinaryIO:
        entry = self.entry(repo_type, repo_id, commit, path)
        key = self._blob_key(entry)
        request = {"Bucket": self.bucket, "Key": key}
        if start is not None:
            request["Range"] = f"bytes={start}-{'' if end is None else end - 1}"
        return self.s3.get_object(**request)["Body"]

    def download(
        self,
        repo_type: RepoType,
        repo_id: str,
        commit: str,
        path: str,
        destination: str,
    ) -> None:
        entry = self.entry(repo_type, repo_id, commit, path)
        key = self._blob_key(entry)
        self.s3.download_file(
            self.bucket,
            key,
            destination,
            Config=self.transfer_config,
        )

    def _blob_key(self, entry: FileEntry) -> str:
        digest = entry["blob_sha256"]
        return self.key("blobs", "sha256", digest[:2], digest)


def _repo() -> _S3Repository:
    if _BACKEND is None:
        raise RuntimeError("S3 mirror is not installed")
    return _BACKEND


def _use_hub(repo_type: str | None, repo_id: str) -> bool:
    """Route a repository absent from the mirror to the Hub when fallback is on.

    A mirrored repository is never mixed with Hub reads: its manifest stays
    authoritative, so missing revisions or files still fail closed.
    """
    if not _HUB_FALLBACK:
        return False
    try:
        validated_repo_type = _validate_repo_type(repo_type)
        _validate_repo_id(repo_id)
    except ValueError:
        return True
    return not _repo().has_repository(validated_repo_type, repo_id)


def _cache_root(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    if hub_cache := os.environ.get("HF_HUB_CACHE"):
        return Path(hub_cache).expanduser()
    return Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"


def _repo_cache_root(
    cache_dir: str | Path | None, repo_type: RepoType, repo_id: str
) -> Path:
    return (
        _cache_root(cache_dir)
        / f"{_REPO_DIRECTORIES[repo_type]}--{repo_id.replace('/', '--')}"
    )


def _cache_layout(
    cache_dir: str | Path | None, repo_type: RepoType, repo_id: str
) -> _CacheLayout:
    return _CacheLayout(_repo_cache_root(cache_dir, repo_type, repo_id))


def _valid_cached_blob(path: Path, entry: FileEntry) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry["size"]
        and _sha256_file(path)[0] == entry["blob_sha256"]
    )


def _download_verified_blob(
    repo_type: RepoType,
    repo_id: str,
    commit: str,
    filename: str,
    entry: FileEntry,
    blob: Path,
) -> None:
    blob.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=blob.parent, prefix=".download-")
    os.close(descriptor)
    try:
        _repo().download(repo_type, repo_id, commit, filename, temporary)
        digest, size = _sha256_file(temporary)
        if size != entry["size"] or digest != entry["blob_sha256"]:
            raise OSError(f"Integrity check failed for {repo_id}@{commit}/{filename}")
        os.replace(temporary, blob)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.symlink_to(os.path.relpath(source, destination.parent))
    except OSError:
        shutil.copy2(source, destination)


def _write_cached_ref(layout: _CacheLayout, revision: str, commit: str) -> None:
    if revision == commit:
        return
    ref = layout.ref(revision)
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(commit)


def _cache_file(
    repo_type: RepoType,
    repo_id: str,
    revision: str,
    commit: str,
    filename: str,
    entry: FileEntry,
    cache_dir: str | Path | None,
    force: bool = False,
) -> Path:
    from filelock import FileLock

    layout = _cache_layout(cache_dir, repo_type, repo_id)
    blob = layout.blob(entry)
    target = layout.file(commit, filename)
    lock_path = layout.lock(entry)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        if force or not _valid_cached_blob(blob, entry):
            _download_verified_blob(repo_type, repo_id, commit, filename, entry, blob)
        _link_or_copy(blob, target)
        _write_cached_ref(layout, revision, commit)
    return target.absolute()


def _cached_commit(layout: _CacheLayout, repo_id: str, revision: str) -> str:
    if _is_commit(revision):
        return revision
    try:
        return layout.ref(revision).read_text().strip()
    except FileNotFoundError as error:
        raise _hub_error("local", f"No cached revision for {repo_id}@{revision}", error)


def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    **kwargs: Any,
) -> str:
    if _use_hub(repo_type, repo_id):
        return _UPSTREAM["hf_hub_download"](
            repo_id,
            filename,
            subfolder=subfolder,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            **kwargs,
        )
    del kwargs
    validated_repo_type = _validate_repo_type(repo_type)
    _validate_repo_id(repo_id)
    revision = revision or _DEFAULT_REVISION
    filename = f"{subfolder.rstrip('/')}/{filename}" if subfolder else filename
    _validate_repo_path(filename)
    layout = _cache_layout(cache_dir, validated_repo_type, repo_id)
    if local_files_only:
        commit = _cached_commit(layout, repo_id, revision)
        target = layout.file(commit, filename)
        if not target.exists():
            raise _hub_error(
                "local", f"No cached file for {repo_id}@{revision}/{filename}"
            )
    else:
        repository = _repo()
        commit = repository.resolve(validated_repo_type, repo_id, revision)
        entry = repository.entry(validated_repo_type, repo_id, commit, filename)
        target = _cache_file(
            validated_repo_type,
            repo_id,
            revision,
            commit,
            filename,
            entry,
            cache_dir,
            force_download,
        )
    if local_dir is not None:
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, destination)
        target = destination
    return str(target.absolute())


def _as_patterns(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else value


def _selected_files(
    files: Sequence[str] | Mapping[str, FileEntry],
    allow_patterns: list[str],
    ignore_patterns: list[str],
) -> list[str]:
    return [
        name
        for name in files
        if (
            not allow_patterns
            or any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns)
        )
        and not any(fnmatch.fnmatch(name, pattern) for pattern in ignore_patterns)
    ]


def _copy_files(
    source_root: Path, filenames: Sequence[str], destination_root: Path
) -> None:
    for filename in filenames:
        destination = destination_root / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / filename, destination)


def _snapshot_result(snapshot: Path, local_dir: str | Path | None) -> str:
    return str((Path(local_dir) if local_dir is not None else snapshot).absolute())


def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    max_workers: int = 8,
    local_files_only: bool = False,
    **kwargs: Any,
) -> str:
    if _use_hub(repo_type, repo_id):
        return _UPSTREAM["snapshot_download"](
            repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
            local_files_only=local_files_only,
            **kwargs,
        )
    del kwargs
    validated_repo_type = _validate_repo_type(repo_type)
    revision = revision or _DEFAULT_REVISION
    allow = _as_patterns(allow_patterns)
    ignore = _as_patterns(ignore_patterns)
    layout = _cache_layout(cache_dir, validated_repo_type, repo_id)
    files: Sequence[str] | Mapping[str, FileEntry]
    if local_files_only:
        commit = _cached_commit(layout, repo_id, revision)
        snapshot = layout.snapshot(commit)
        if not snapshot.is_dir():
            raise _hub_error("local", f"No cached snapshot for {repo_id}@{revision}")
        files = [
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file()
        ]
    else:
        repository = _repo()
        commit = repository.resolve(validated_repo_type, repo_id, revision)
        files = repository.manifest(validated_repo_type, repo_id, commit)["files"]
    selected = _selected_files(files, allow, ignore)

    if local_files_only:
        if local_dir is not None:
            _copy_files(snapshot, selected, Path(local_dir))
        return _snapshot_result(snapshot, local_dir)

    def download_file(name: str) -> str:
        return hf_hub_download(
            repo_id,
            name,
            repo_type=validated_repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(download_file, selected))
    return _snapshot_result(layout.snapshot(commit), local_dir)


class S3HfApi:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._hub_args = (args, kwargs)
        self._hub_api: Any = None

    def _hub(self) -> Any:
        if self._hub_api is None:
            args, kwargs = self._hub_args
            self._hub_api = _UPSTREAM["HfApi"](*args, **kwargs)
        return self._hub_api

    def repo_info(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        repo_type: str | None = None,
        **kwargs: Any,
    ):
        if _use_hub(repo_type, repo_id):
            return self._hub().repo_info(
                repo_id, revision=revision, repo_type=repo_type, **kwargs
            )
        del kwargs
        if repo_type == "dataset":
            return self.dataset_info(repo_id, revision=revision)
        return self.model_info(repo_id, revision=revision)

    def _info(self, repo_id: str, repo_type: RepoType, revision: str | None):
        from huggingface_hub import DatasetInfo, ModelInfo

        repository = _repo()
        commit = repository.resolve(repo_type, repo_id, revision)
        manifest = repository.manifest(repo_type, repo_id, commit)
        siblings = [
            {"rfilename": path, "size": entry["size"], "blobId": entry["blob_sha256"]}
            for path, entry in manifest["files"].items()
        ]
        info_class = DatasetInfo if repo_type == "dataset" else ModelInfo
        return info_class(id=repo_id, sha=commit, private=True, siblings=siblings)

    def model_info(self, repo_id: str, *, revision: str | None = None, **kwargs: Any):
        if _use_hub("model", repo_id):
            return self._hub().model_info(repo_id, revision=revision, **kwargs)
        del kwargs
        return self._info(repo_id, "model", revision)

    def dataset_info(self, repo_id: str, *, revision: str | None = None, **kwargs: Any):
        if _use_hub("dataset", repo_id):
            return self._hub().dataset_info(repo_id, revision=revision, **kwargs)
        del kwargs
        return self._info(repo_id, "dataset", revision)

    def list_repo_tree(
        self,
        repo_id: str,
        path_in_repo: str | None = None,
        *,
        recursive: bool = False,
        revision: str | None = None,
        repo_type: str | None = None,
        **kwargs: Any,
    ):
        if _use_hub(repo_type, repo_id):
            yield from self._hub().list_repo_tree(
                repo_id,
                path_in_repo,
                recursive=recursive,
                revision=revision,
                repo_type=repo_type,
                **kwargs,
            )
            return
        del kwargs
        from huggingface_hub.hf_api import RepoFile, RepoFolder

        validated_repo_type = _validate_repo_type(repo_type)
        repository = _repo()
        commit = repository.resolve(validated_repo_type, repo_id, revision)
        prefix = (path_in_repo.rstrip("/") + "/") if path_in_repo else ""
        folders = set()
        for path, entry in sorted(
            repository.manifest(validated_repo_type, repo_id, commit)["files"].items()
        ):
            if not path.startswith(prefix):
                continue
            remainder = path[len(prefix) :]
            if not recursive and "/" in remainder:
                folder = prefix + remainder.split("/", 1)[0]
                if folder not in folders:
                    folders.add(folder)
                    yield RepoFolder(path=folder, oid=commit)
            else:
                yield RepoFile(path=path, size=entry["size"], oid=entry["blob_sha256"])

    def list_repo_files(self, repo_id: str, **kwargs: Any) -> list[str]:
        if _use_hub(kwargs.get("repo_type"), repo_id):
            return self._hub().list_repo_files(repo_id, **kwargs)
        return [
            entry.path
            for entry in self.list_repo_tree(repo_id, recursive=True, **kwargs)
        ]

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: Any) -> str:
        return hf_hub_download(repo_id, filename, **kwargs)

    def __getattr__(self, name: str):
        if _HUB_FALLBACK and not name.startswith("_"):
            return getattr(self._hub(), name)
        raise NotImplementedError(
            f"Hub operation {name!r} is unavailable: the S3 mirror is read-only"
        )


def _parse_hf_path(path: str | Path, revision: str | None = None):
    from huggingface_hub.hf_file_system import HfFileSystemResolvedPath

    path = str(path).removeprefix("hf://").strip("/")
    repo_type = "dataset" if path.startswith("datasets/") else "model"
    if repo_type == "dataset":
        path = path[len("datasets/") :]
    pieces = path.split("/")
    if len(pieces) < 2:
        raise NotImplementedError("Repository listing is not supported")
    name = pieces[1]
    if "@" in name:
        name, path_revision = name.split("@", 1)
        path_revision = unquote(path_revision)
        if revision is not None and revision != path_revision:
            raise ValueError("Conflicting revisions")
        revision = path_revision
    repo_id = _validate_repo_id(f"{pieces[0]}/{name}")
    path_in_repo = "/".join(pieces[2:])
    if path_in_repo:
        _validate_repo_path(path_in_repo)
    return HfFileSystemResolvedPath(
        repo_type, repo_id, revision or _DEFAULT_REVISION, path_in_repo
    )


def _hf_base(repo_type: RepoType, repo_id: str, revision: str) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"{prefix}{repo_id}@{revision}"


def _filesystem_class() -> type:
    from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
    from huggingface_hub import HfFileSystem as OriginalHfFileSystem

    class S3HfFileSystem(OriginalHfFileSystem):
        def __init__(
            self, *args: Any, block_size: int | None = None, **kwargs: Any
        ) -> None:
            if _HUB_FALLBACK:
                OriginalHfFileSystem.__init__(self, *args, **kwargs)
            else:
                AbstractFileSystem.__init__(self)
            self.block_size = block_size or _RANGE_BLOCK_SIZE

        def _on_hub(self, path: str) -> bool:
            if not _HUB_FALLBACK:
                return False
            try:
                resolved = _parse_hf_path(path)
            except (ValueError, NotImplementedError):
                return True
            return _use_hub(resolved.repo_type, resolved.repo_id)

        def _resolved(self, path: str, revision: str | None = None):
            resolved = _parse_hf_path(path, revision)
            return resolved, _repo().resolve(
                resolved.repo_type, resolved.repo_id, resolved.revision
            )

        def _repository_view(self, path: str, revision: str | None = None):
            resolved, commit = self._resolved(path, revision)
            files = _repo().manifest(resolved.repo_type, resolved.repo_id, commit)[
                "files"
            ]
            return (
                resolved,
                commit,
                _hf_base(resolved.repo_type, resolved.repo_id, commit),
                files,
            )

        def resolve_path(self, path: str, revision: str | None = None):
            if self._on_hub(path):
                return super().resolve_path(path, revision)
            resolved, _ = self._resolved(path, revision)
            return resolved

        def info(
            self,
            path: str,
            refresh: bool = False,
            revision: str | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if self._on_hub(path):
                return super().info(path, refresh=refresh, revision=revision, **kwargs)
            del refresh, kwargs
            resolved, _, base, files = self._repository_view(path, revision)
            if not resolved.path_in_repo:
                return {"name": base, "size": 0, "type": "directory"}
            if resolved.path_in_repo in files:
                entry = files[resolved.path_in_repo]
                return {
                    "name": f"{base}/{resolved.path_in_repo}",
                    "size": entry["size"],
                    "type": "file",
                }
            prefix = resolved.path_in_repo.rstrip("/") + "/"
            if any(name.startswith(prefix) for name in files):
                return {
                    "name": f"{base}/{resolved.path_in_repo}",
                    "size": 0,
                    "type": "directory",
                }
            raise FileNotFoundError(path)

        def ls(
            self,
            path: str,
            detail: bool = True,
            refresh: bool = False,
            revision: str | None = None,
            **kwargs: Any,
        ):
            if self._on_hub(path):
                return super().ls(
                    path, detail=detail, refresh=refresh, revision=revision, **kwargs
                )
            del refresh, kwargs
            resolved, _, base, files = self._repository_view(path, revision)
            if resolved.path_in_repo in files:
                entry = files[resolved.path_in_repo]
                result = {
                    "name": f"{base}/{resolved.path_in_repo}",
                    "type": "file",
                    "size": entry["size"],
                }
                return [result] if detail else [result["name"]]
            prefix = (
                (resolved.path_in_repo.rstrip("/") + "/")
                if resolved.path_in_repo
                else ""
            )
            if resolved.path_in_repo and not any(
                name.startswith(prefix) for name in files
            ):
                raise FileNotFoundError(path)
            results: list[dict[str, Any]] = []
            seen: set[str] = set()
            for name, entry in sorted(files.items()):
                if not name.startswith(prefix):
                    continue
                remainder = name[len(prefix) :]
                if "/" in remainder:
                    item, kind, size = (
                        prefix + remainder.split("/", 1)[0],
                        "directory",
                        0,
                    )
                else:
                    item, kind, size = name, "file", entry["size"]
                if item not in seen:
                    seen.add(item)
                    results.append(
                        {"name": f"{base}/{item}", "type": kind, "size": size}
                    )
            return results if detail else [item["name"] for item in results]

        def glob(self, path: str, detail: bool = False, **kwargs: Any):
            if self._on_hub(path):
                return super().glob(path, detail=detail, **kwargs)
            del kwargs
            from fsspec.utils import glob_translate

            stripped = str(path).removeprefix("hf://")
            wildcard = min(
                (stripped.find(char) for char in "*?[" if char in stripped), default=-1
            )
            if wildcard < 0:
                try:
                    info = self.info(stripped)
                except FileNotFoundError:
                    return {} if detail else []
                return {info["name"]: info} if detail else [info["name"]]
            root = stripped[:wildcard].rsplit("/", 1)[0]
            resolved, commit, _, files = self._repository_view(root)
            alias = _hf_base(
                resolved.repo_type,
                resolved.repo_id,
                quote(resolved.revision, safe=""),
            )
            immutable = _hf_base(resolved.repo_type, resolved.repo_id, commit)
            pattern = stripped.replace(alias, immutable, 1)
            matcher = re.compile(glob_translate(pattern))
            matches = [
                f"{immutable}/{name}"
                for name in files
                if matcher.match(f"{immutable}/{name}")
            ]
            return {name: self.info(name) for name in matches} if detail else matches

        def exists(self, path: str, **kwargs: Any) -> bool:
            if self._on_hub(path):
                return super().exists(path, **kwargs)
            try:
                self.info(path, **kwargs)
                return True
            except (FileNotFoundError, OSError):
                return False

        def _open(
            self,
            path: str,
            mode: str = "rb",
            revision: str | None = None,
            block_size: int | None = None,
            **kwargs: Any,
        ) -> Any:
            if self._on_hub(path):
                return super()._open(
                    path, mode, revision=revision, block_size=block_size, **kwargs
                )
            del kwargs
            if mode not in {"rb", "r"}:
                raise NotImplementedError("The S3 mirror is read-only")
            resolved, commit = self._resolved(path, revision)
            entry = _repo().entry(
                resolved.repo_type, resolved.repo_id, commit, resolved.path_in_repo
            )

            class RangeFile(AbstractBufferedFile):
                def _fetch_range(inner, start: int, end: int) -> bytes:
                    body = _repo().read(
                        resolved.repo_type,
                        resolved.repo_id,
                        commit,
                        resolved.path_in_repo,
                        start,
                        end,
                    )
                    try:
                        return body.read()
                    finally:
                        body.close()

            return RangeFile(
                self,
                str(path),
                mode="rb",
                size=entry["size"],
                block_size=block_size or self.block_size,
            )

    return S3HfFileSystem


class _DatasetsConfigLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self.wrapped = wrapped

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return (
            self.wrapped.create_module(spec)
            if hasattr(self.wrapped, "create_module")
            else None
        )

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        module.__dict__["USE_PARQUET_EXPORT"] = False


class _DatasetsConfigFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != "datasets.config":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _DatasetsConfigLoader(spec.loader)
        return spec


def _install_api_helpers(hub_module: ModuleType, api_module: ModuleType) -> None:
    api = S3HfApi()
    setattr(api_module, "api", api)
    for name in _API_HELPERS:
        helper = getattr(api, name)
        setattr(api_module, name, helper)
        setattr(hub_module, name, helper)


def install(
    *,
    bucket: str | None = None,
    prefix: str | None = None,
    allow_hub_fallback: bool | None = None,
) -> None:
    """Install the mirror for this process before importing Hub consumers.

    The S3 bucket must be supplied explicitly or by ``HF_S3_MIRROR_BUCKET``.
    Installation is process-wide and fails closed to the public Hub unless
    ``allow_hub_fallback`` (or ``HF_S3_MIRROR_ALLOW_HUB_FALLBACK``) is set, in
    which case repositories absent from the mirror are read from the Hub.
    """
    global _BACKEND, _INSTALLED, _HUB_FALLBACK
    if allow_hub_fallback is None:
        allow_hub_fallback = (
            os.environ.get("HF_S3_MIRROR_ALLOW_HUB_FALLBACK", "").lower()
            in _TRUE_VALUES
        )
    if _INSTALLED and bucket is None and prefix is None:
        if allow_hub_fallback != _HUB_FALLBACK:
            raise RuntimeError("S3 mirror is already installed with different settings")
        return
    resolved_bucket = bucket or os.environ.get("HF_S3_MIRROR_BUCKET")
    resolved_prefix = prefix or os.environ.get(
        "HF_S3_MIRROR_PREFIX", _DEFAULT_MIRROR_PREFIX
    )
    if not resolved_bucket or "://" in resolved_bucket or "/" in resolved_bucket:
        raise ValueError("A valid S3 bucket is required for the mirror")
    if not resolved_prefix.strip("/"):
        raise ValueError("The mirror prefix must not be empty")
    if _INSTALLED:
        assert _BACKEND is not None
        if (_BACKEND.bucket, _BACKEND.prefix, _HUB_FALLBACK) != (
            resolved_bucket,
            resolved_prefix.strip("/"),
            allow_hub_fallback,
        ):
            raise RuntimeError("S3 mirror is already installed with different settings")
        return
    already_loaded = [
        name
        for name in ("huggingface_hub", "datasets", "transformers")
        if name in sys.modules
    ]
    if already_loaded:
        raise RuntimeError(
            "hf_hub_s3_mirror.install must run before Hub consumers; already loaded: "
            + ", ".join(already_loaded)
        )
    _BACKEND = _S3Repository(resolved_bucket, resolved_prefix)
    if not allow_hub_fallback:
        # Fail closed if a pinned downstream call path escapes our monkeypatches:
        # it must fail against localhost instead of silently reaching huggingface.co.
        os.environ["HF_ENDPOINT"] = _FAIL_CLOSED_HF_ENDPOINT
        # S3 is the only supported transfer backend while the mirror is installed.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    import huggingface_hub
    import huggingface_hub._snapshot_download
    import huggingface_hub.file_download
    import huggingface_hub.hf_api
    import huggingface_hub.hf_file_system

    _UPSTREAM.update(
        HfApi=huggingface_hub.hf_api.HfApi,
        hf_hub_download=huggingface_hub.file_download.hf_hub_download,
        snapshot_download=huggingface_hub._snapshot_download.snapshot_download,
    )
    _HUB_FALLBACK = allow_hub_fallback

    filesystem = _filesystem_class()
    patches = {
        "HfApi": S3HfApi,
        "HfFileSystem": filesystem,
        "hf_hub_download": hf_hub_download,
        "snapshot_download": snapshot_download,
    }
    for name, value in patches.items():
        setattr(huggingface_hub, name, value)
    setattr(huggingface_hub.hf_api, "HfApi", S3HfApi)
    _install_api_helpers(huggingface_hub, huggingface_hub.hf_api)
    setattr(huggingface_hub.hf_file_system, "HfFileSystem", filesystem)
    setattr(huggingface_hub.file_download, "hf_hub_download", hf_hub_download)
    setattr(huggingface_hub._snapshot_download, "snapshot_download", snapshot_download)
    sys.meta_path.insert(0, _DatasetsConfigFinder())
    _INSTALLED = True


if os.environ.get("HF_S3_MIRROR_ENABLED", "").lower() in _TRUE_VALUES:
    install()
