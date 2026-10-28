from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from hf_hub_s3_mirror import publish

COMMIT = "a" * 40


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


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Serve `files` as one upstream commit and publish into a fake bucket."""
    s3 = FakeS3()
    files: dict[str, bytes] = {}

    class FakeHfApi:
        def __init__(self, token: str | None) -> None:
            pass

        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(sha=COMMIT)

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

    return SimpleNamespace(s3=s3, files=files, run=run)


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
