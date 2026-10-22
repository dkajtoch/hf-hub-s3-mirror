#!/usr/bin/env python3
"""Publish a Hugging Face repository revision to an S3 mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote

import boto3
from botocore.exceptions import ClientError
from huggingface_hub import HfApi, hf_hub_download


DEFAULT_PREFIX = "hf_hub"
_COMMIT_LENGTH = 40
_HASH_CHUNK_SIZE = 8 * 1024 * 1024
_REPOSITORY_DIRECTORIES = {"model": "models", "dataset": "datasets"}


def _is_commit(value: str) -> bool:
    return len(value) == _COMMIT_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


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
    if not path or path.startswith("/") or "\\" in path:
        raise ValueError(f"Invalid repository file path: {path!r}")
    if any(part in {"", ".", ".."} for part in PurePosixPath(path).parts):
        raise ValueError(f"Invalid repository file path: {path!r}")


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


def put_immutable_manifest(s3, bucket: str, key: str, body: bytes) -> None:
    existing = object_exists(s3, bucket, key)
    if existing is not None:
        current = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        if current != body:
            raise RuntimeError(
                f"Refusing to overwrite immutable manifest: s3://{bucket}/{key}"
            )
        return
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    current = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if current != body:
        raise RuntimeError(
            f"Uploaded manifest failed verification: s3://{bucket}/{key}"
        )


def publish(args: argparse.Namespace) -> dict:
    if os.environ.get("HF_S3_MIRROR_ENABLED", "").lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "Disable HF_S3_MIRROR_ENABLED when publishing from the upstream Hugging Face Hub"
        )

    repo_id, repo_type, revision = parse_hf_path(
        args.hf_path, args.repo_type, args.revision
    )
    if repo_type not in {"model", "dataset"}:
        raise ValueError(f"Unsupported repository type: {repo_type!r}")
    publish_ref = None if _is_commit(revision) else revision
    api = HfApi(token=args.token)
    info = api.repo_info(
        repo_id, repo_type=repo_type, revision=revision, files_metadata=True
    )
    commit = info.sha
    if not isinstance(commit, str) or not _is_commit(commit):
        raise RuntimeError(
            f"Upstream revision did not resolve to a 40-character commit: {commit!r}"
        )

    s3 = None if args.dry_run else boto3.client("s3", region_name=args.region)
    prefix = args.prefix.strip("/")
    repository_root = _s3_key(
        prefix, "repos", _REPOSITORY_DIRECTORIES[repo_type], repo_id
    )
    blob_root = _s3_key(prefix, "blobs", "sha256")
    files: dict[str, dict] = {}
    uploaded_blobs = 0

    repo_files = api.list_repo_files(repo_id, repo_type=repo_type, revision=commit)
    for index, filename in enumerate(repo_files, start=1):
        validate_repo_file(filename)
        local_path = Path(
            hf_hub_download(
                repo_id,
                filename,
                repo_type=repo_type,
                revision=commit,
                token=args.token,
                cache_dir=args.cache_dir,
            )
        )
        sha256, size = _sha256_file(local_path)
        blob_key = f"{blob_root}/{sha256[:2]}/{sha256}"
        if not args.dry_run:
            assert s3 is not None
            uploaded_blobs += upload_blob(
                s3, args.bucket, blob_key, local_path, sha256, size
            )
        files[filename] = {"blob_sha256": sha256, "size": size}
        print(f"[{index}/{len(repo_files)}] {filename} -> {sha256}", file=sys.stderr)

    manifest = {
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

    if not args.dry_run:
        assert s3 is not None
        put_immutable_manifest(s3, args.bucket, manifest_key, json_bytes(manifest))
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
        "file_count": len(files),
        "uploaded_blob_count": uploaded_blobs,
        "manifest": f"s3://{args.bucket}/{manifest_key}",
        "ref": f"s3://{args.bucket}/{ref_key}" if ref_key else None,
        "dry_run": args.dry_run,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hf_path", help="org/name, hf://org/name, or hf://datasets/org/name"
    )
    parser.add_argument(
        "--repo-type",
        choices=["model", "dataset"],
        help="inferred from hf://datasets/ when omitted",
    )
    parser.add_argument(
        "--revision",
        help="upstream branch, tag, or commit; required unless encoded in the HF path",
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--region", help="S3 region; defaults to the AWS configuration")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--dry-run", action="store_true", help="download and hash without writing S3"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(publish(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
