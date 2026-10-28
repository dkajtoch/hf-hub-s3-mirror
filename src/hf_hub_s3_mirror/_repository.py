"""S3 access and immutable revision resolution; no Hub network fallback."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from functools import cached_property, lru_cache
from typing import Any, Literal
from urllib.parse import quote

from hf_hub_s3_mirror._artifacts import (
    FileEntry,
    Manifest,
    RepoType,
    _DEFAULT_REVISION,
    _REPO_DIRECTORIES,
    _is_commit,
    _validate_manifest,
    _validate_repo_id,
    _validate_repo_path,
)
from hf_hub_s3_mirror._errors import _hub_error

_MEBIBYTE = 1024**2
_NOT_FOUND_CODES = {"NoSuchKey", "404", "NotFound"}
_ACCESS_DENIED_CODES = {"AccessDenied", "403", "InvalidAccessKeyId", "ExpiredToken"}


def _read_positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_bucket(bucket: str | None) -> str:
    if not bucket or "://" in bucket or "/" in bucket:
        raise ValueError("A valid S3 bucket is required for the mirror")
    return bucket


class _S3Repository:
    def __init__(self, bucket: str | Callable[[], str], prefix: str) -> None:
        import boto3
        from boto3.s3.transfer import TransferConfig

        self.bucket_source = bucket
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

    @cached_property
    def bucket(self) -> str:
        source = self.bucket_source
        return _validate_bucket(source() if callable(source) else source)

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
