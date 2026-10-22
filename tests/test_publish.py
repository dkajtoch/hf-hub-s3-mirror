from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from hf_hub_s3_mirror import publish


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


def test_publisher_requires_an_explicit_revision() -> None:
    # Given
    path = "hf://datasets/example/data"
    expected_error = "Specify a revision"

    # When
    with pytest.raises(ValueError) as raised:
        publish.parse_hf_path(path, repo_type=None, revision=None)

    # Then
    assert expected_error in str(raised.value)


def test_publisher_writes_immutable_manifest_and_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    commit = "a" * 40
    payload = b"mirrored data"
    digest = hashlib.sha256(payload).hexdigest()
    source_file = tmp_path / "data.parquet"
    source_file.write_bytes(payload)
    s3 = FakeS3()
    args = argparse.Namespace(
        hf_path="hf://datasets/example/data@main",
        repo_type=None,
        revision=None,
        bucket="example-bucket",
        prefix="hf_hub",
        region="us-east-1",
        cache_dir=None,
        token=None,
        dry_run=False,
    )
    expected_manifest_key = (
        f"hf_hub/repos/datasets/example/data/snapshots/{commit}/manifest.json"
    )
    expected_ref_key = "hf_hub/repos/datasets/example/data/refs/main.json"
    expected_blob_key = f"hf_hub/blobs/sha256/{digest[:2]}/{digest}"

    class FakeHfApi:
        def __init__(self, token: str | None) -> None:
            pass

        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(sha=commit)

        def list_repo_files(self, *args: object, **kwargs: object) -> list[str]:
            return ["data.parquet"]

    monkeypatch.setattr(publish, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        publish, "hf_hub_download", lambda *args, **kwargs: str(source_file)
    )
    monkeypatch.setattr(publish.boto3, "client", lambda *args, **kwargs: s3)
    monkeypatch.delenv("HF_S3_MIRROR_ENABLED", raising=False)

    # When
    receipt = publish.publish(args)

    # Then
    manifest = json.loads(s3.objects[expected_manifest_key])
    ref = json.loads(s3.objects[expected_ref_key])
    assert s3.objects[expected_blob_key] == payload
    assert manifest["files"] == {
        "data.parquet": {"blob_sha256": digest, "size": len(payload)}
    }
    assert ref["commit"] == commit
    assert receipt["uploaded_blob_count"] == 1
