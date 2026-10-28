#!/usr/bin/env python3
"""Publish a Hugging Face repository revision to an S3 mirror."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import boto3
from botocore.exceptions import ClientError
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import BlobLfsInfo
from huggingface_hub.utils import filter_repo_objects

from hf_hub_s3_mirror._artifacts import (
    FileEntry,
    RepoType,
    _DEFAULT_MIRROR_PREFIX,
    _REPO_DIRECTORIES,
    _is_commit,
    _sha256_file,
    _sha256_stream,
    _validate_repo_path,
    _validate_repo_type,
)


DEFAULT_PREFIX = _DEFAULT_MIRROR_PREFIX
_KIND_REPO_TYPES = {"model": "model", "tokenizer": "model", "dataset": "dataset"}
# Everything AutoTokenizer and AutoConfig read, including remote-code modules.
TOKENIZER_PATTERNS = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "additional_chat_templates/*.jinja",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "*.model",
    "*.tiktoken",
    "*.py",
)


def _s3_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part)


def parse_hf_path(
    value: str, repo_type: str | None, revision: str | None
) -> tuple[str, str, str]:
    path = value.removeprefix("hf://").strip("/")
    inferred_type = "model"
    if path.startswith("datasets/"):
        inferred_type = "dataset"
        path = path.removeprefix("datasets/")
    elif path.startswith("models/"):
        path = path.removeprefix("models/")

    if "@" in path:
        path, path_revision = path.split("@", 1)
        path_revision = unquote(path_revision)
        if revision is not None and revision != path_revision:
            raise ValueError(
                f"Conflicting revisions: {revision!r} and {path_revision!r}"
            )
        revision = path_revision

    parts = path.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError(
            "HF path must identify one namespaced repository, for example org/name"
        )
    if revision is None:
        raise ValueError("Specify a revision in the HF path or with --revision")
    return path, repo_type or inferred_type, revision


def validate_repo_file(path: str) -> None:
    try:
        _validate_repo_path(path)
    except ValueError as error:
        raise ValueError(f"Invalid repository file path: {path!r}") from error


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def object_exists(s3, bucket: str, key: str) -> dict | None:
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise


def upload_blob(
    s3, bucket: str, key: str, source: Path, digest: str, size: int
) -> bool:
    existing = object_exists(s3, bucket, key)
    if existing is not None:
        if existing.get("ContentLength") != size:
            raise RuntimeError(
                f"Existing immutable blob has an unexpected size: s3://{bucket}/{key}"
            )
        existing_digest = existing.get("Metadata", {}).get("sha256")
        if existing_digest not in {None, digest}:
            raise RuntimeError(
                f"Existing immutable blob has an unexpected hash: s3://{bucket}/{key}"
            )
        # Size and metadata are writer-controlled; only the bytes prove the key.
        body = s3.get_object(Bucket=bucket, Key=key)["Body"]
        if _sha256_stream(body) != (digest, size):
            raise RuntimeError(
                f"Existing immutable blob has unexpected content: s3://{bucket}/{key}"
            )
        return False

    s3.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={"Metadata": {"sha256": digest}},
    )
    uploaded = s3.head_object(Bucket=bucket, Key=key)
    if uploaded.get("ContentLength") != size:
        raise RuntimeError(
            f"Uploaded blob failed size verification: s3://{bucket}/{key}"
        )
    return True


def merge_manifest_files(
    current: dict[str, dict], new: dict[str, dict], location: str
) -> dict[str, dict]:
    """Union two file sets of one commit; a shared path must keep its blob."""
    for filename, entry in new.items():
        if filename in current and current[filename] != entry:
            raise RuntimeError(
                f"Refusing to change {filename!r} in immutable manifest: {location}"
            )
    return {**current, **new}


def put_manifest(s3, bucket: str, key: str, manifest: dict) -> dict:
    """Write a commit manifest, extending any files published for it earlier."""
    if object_exists(s3, bucket, key) is not None:
        current = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        identity = ("schema_version", "repo_type", "repo_id", "commit")
        if any(current.get(field) != manifest[field] for field in identity):
            raise RuntimeError(
                f"Existing manifest has a different identity: s3://{bucket}/{key}"
            )
        files = merge_manifest_files(
            current["files"], manifest["files"], f"s3://{bucket}/{key}"
        )
        if files == current["files"]:
            return current
        manifest = {**manifest, "files": files}
    body = json_bytes(manifest)
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    current = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if current != body:
        raise RuntimeError(
            f"Uploaded manifest failed verification: s3://{bucket}/{key}"
        )
    return manifest


def select_files(
    repo_files: list[str],
    kind: str,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> list[str]:
    allow = list(allow_patterns or [])
    if kind == "tokenizer":
        allow = [*TOKENIZER_PATTERNS, *allow]
    return list(
        filter_repo_objects(
            repo_files,
            allow_patterns=allow or None,
            ignore_patterns=ignore_patterns or None,
        )
    )


@dataclass(frozen=True)
class _SourceRevision:
    repo_id: str
    repo_type: RepoType
    revision: str
    commit: str
    api: HfApi
    upstream_lfs: dict[str, BlobLfsInfo | None]


def _resolve_source(args: argparse.Namespace) -> _SourceRevision:
    expected_type = _KIND_REPO_TYPES[args.kind]
    path = args.hf_path.removeprefix("hf://")
    if (path.startswith("datasets/") and expected_type != "dataset") or (
        path.startswith("models/") and expected_type != "model"
    ):
        raise ValueError(f"{args.hf_path!r} is not a {expected_type} repository")
    repo_id, repo_type, revision = parse_hf_path(
        args.hf_path, expected_type, args.revision
    )
    api = HfApi(token=args.token)
    info = api.repo_info(
        repo_id, repo_type=repo_type, revision=revision, files_metadata=True
    )
    commit = info.sha
    if not isinstance(commit, str) or not _is_commit(commit):
        raise RuntimeError(
            f"Upstream revision did not resolve to a 40-character commit: {commit!r}"
        )
    upstream_lfs = {
        sibling.rfilename: sibling.lfs
        for sibling in getattr(info, "siblings", None) or []
    }

    return _SourceRevision(
        repo_id, _validate_repo_type(repo_type), revision, commit, api, upstream_lfs
    )


def _publish_files(
    args: argparse.Namespace, source: _SourceRevision, s3: Any
) -> tuple[dict[str, FileEntry], int]:
    blob_root = _s3_key(args.prefix.strip("/"), "blobs", "sha256")
    files: dict[str, FileEntry] = {}
    uploaded_blobs = 0
    repo_files = select_files(
        source.api.list_repo_files(
            source.repo_id, repo_type=source.repo_type, revision=source.commit
        ),
        args.kind,
        args.allow_patterns,
        args.ignore_patterns,
    )
    if not repo_files:
        raise ValueError(
            f"No files in {source.repo_id}@{source.commit} match the requested patterns"
        )
    for index, filename in enumerate(repo_files, start=1):
        validate_repo_file(filename)
        local_path = Path(
            hf_hub_download(
                source.repo_id,
                filename,
                repo_type=source.repo_type,
                revision=source.commit,
                token=args.token,
                cache_dir=args.cache_dir,
            )
        )
        sha256, size = _sha256_file(local_path)
        lfs = source.upstream_lfs.get(filename)
        if lfs is not None and (sha256, size) != (lfs.sha256, lfs.size):
            raise RuntimeError(
                f"Downloaded {filename!r} does not match its upstream LFS metadata"
            )
        blob_key = f"{blob_root}/{sha256[:2]}/{sha256}"
        if not args.dry_run:
            assert s3 is not None
            uploaded_blobs += upload_blob(
                s3, args.bucket, blob_key, local_path, sha256, size
            )
        files[filename] = {"blob_sha256": sha256, "size": size}
        print(f"[{index}/{len(repo_files)}] {filename} -> {sha256}", file=sys.stderr)

    return files, uploaded_blobs


def publish(args: argparse.Namespace) -> dict:
    source = _resolve_source(args)
    repo_id, repo_type = source.repo_id, source.repo_type
    revision, commit = source.revision, source.commit
    publish_ref = None if _is_commit(revision) else revision
    s3 = None if args.dry_run else boto3.client("s3", region_name=args.region)
    prefix = args.prefix.strip("/")
    repository_root = _s3_key(prefix, "repos", _REPO_DIRECTORIES[repo_type], repo_id)
    files, uploaded_blobs = _publish_files(args, source, s3)

    manifest: dict = {
        "schema_version": 1,
        "repo_type": repo_type,
        "repo_id": repo_id,
        "commit": commit,
        "files": files,
    }
    manifest_key = f"{repository_root}/snapshots/{commit}/manifest.json"
    ref_key = (
        f"{repository_root}/refs/{quote(publish_ref, safe='')}.json"
        if publish_ref
        else None
    )

    manifest_file_count = len(files)
    if not args.dry_run:
        assert s3 is not None
        manifest = put_manifest(s3, args.bucket, manifest_key, manifest)
        manifest_file_count = len(manifest["files"])
        if publish_ref:
            # Mutable ref is deliberately written last.
            ref = {
                "schema_version": 1,
                "repo_type": repo_type,
                "repo_id": repo_id,
                "revision": publish_ref,
                "commit": commit,
                "updated_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
            s3.put_object(
                Bucket=args.bucket,
                Key=ref_key,
                Body=json_bytes(ref),
                ContentType="application/json",
            )

    return {
        "repo_type": repo_type,
        "repo_id": repo_id,
        "source_revision": revision,
        "commit": commit,
        "published_ref": publish_ref,
        "kind": args.kind,
        "file_count": len(files),
        "manifest_file_count": manifest_file_count,
        "uploaded_blob_count": uploaded_blobs,
        "manifest": f"s3://{args.bucket}/{manifest_key}",
        "ref": f"s3://{args.bucket}/{ref_key}" if ref_key else None,
        "dry_run": args.dry_run,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "hf_path",
        help="org/name or hf://org/name; datasets also accept hf://datasets/org/name",
    )
    common.add_argument(
        "--revision",
        help="upstream branch, tag, or commit; required unless encoded in the HF path",
    )
    common.add_argument(
        "--allow-patterns",
        nargs="+",
        metavar="GLOB",
        help="publish only matching files (tokenizer adds these to its defaults)",
    )
    common.add_argument(
        "--ignore-patterns", nargs="+", metavar="GLOB", help="skip matching files"
    )
    common.add_argument("--bucket", required=True)
    common.add_argument("--prefix", default=DEFAULT_PREFIX)
    common.add_argument("--region", help="S3 region; defaults to the AWS configuration")
    common.add_argument("--cache-dir", type=Path)
    common.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    common.add_argument(
        "--dry-run", action="store_true", help="download and hash without writing S3"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    kinds = parser.add_subparsers(dest="kind", required=True)
    kinds.add_parser("model", parents=[common], help="publish a model repository")
    kinds.add_parser(
        "tokenizer",
        parents=[common],
        help="publish only a model's tokenizer, config, and remote-code files",
    )
    kinds.add_parser("dataset", parents=[common], help="publish a dataset repository")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(publish(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
