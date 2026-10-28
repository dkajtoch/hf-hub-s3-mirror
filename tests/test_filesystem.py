"""Filesystem listings, globbing, verified reads, and exhaustive read-only routing."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _support import (
    COMMIT,
    DATASET_ID,
    FILE,
    FILE_INFO,
    IMMUTABLE_FILE,
    MODEL_ID,
    NEW_DIRECTORY,
    NEW_FILE,
    PAYLOAD,
    BlobRepository,
    FallbackRepository,
    HubApi,
    Observed,
    SnapshotRepository,
)
from fsspec.utils import tokenize
from huggingface_hub.hf_file_system import HfFileSystemResolvedRepositoryPath

from hf_hub_s3_mirror import mirror

NON_PATH_FILESYSTEM_OPERATIONS = frozenset(
    {
        "clear_instance_cache",
        "current",
        "from_dict",
        "from_json",
        "to_dict",
        "to_json",
        "transaction_type",
        "unstrip_protocol",
    }
)


@dataclass(frozen=True)
class Operation:
    method: str
    call: Callable[[Any, Path], Any]
    expected: Any = None
    variant: str | None = None


def _params(operations: list[Operation]) -> list[Any]:
    return [
        pytest.param(
            operation,
            id=operation.method
            if operation.variant is None
            else f"{operation.method}[{operation.variant}]",
        )
        for operation in operations
    ]


def _read_opened(filesystem: Any, tmp_path: Path) -> bytes:
    with filesystem.open(FILE, "rb") as handle:
        return handle.read()


def _write_opened(filesystem: Any, tmp_path: Path) -> None:
    with filesystem.open(NEW_FILE, "wb") as handle:
        handle.write(b"{}")


def _write_mapped(filesystem: Any, tmp_path: Path) -> None:
    filesystem.get_mapper(MODEL_ID)["new.json"] = b"{}"


def _downloaded_with(method: str) -> Callable[[Any, Path], bytes]:
    def download(filesystem: Any, tmp_path: Path) -> bytes:
        destination = tmp_path / "downloaded.json"
        getattr(filesystem, method)(FILE, str(destination))
        return destination.read_bytes()

    return download


def _uploaded_with(method: str) -> Callable[[Any, Path], None]:
    def upload(filesystem: Any, tmp_path: Path) -> None:
        source = tmp_path / "local.json"
        source.write_bytes(b"{}")
        getattr(filesystem, method)(str(source), NEW_FILE)

    return upload


READ_OPERATIONS = [
    Operation("cat", lambda fs, _: fs.cat(FILE), PAYLOAD),
    Operation("cat_file", lambda fs, _: fs.cat_file(FILE), PAYLOAD),
    Operation(
        "cat_ranges", lambda fs, _: fs.cat_ranges([FILE], [0], [4]), [PAYLOAD[:4]]
    ),
    Operation(
        "checksum", lambda fs, _: fs.checksum(FILE), int(tokenize(FILE_INFO), 16)
    ),
    Operation("disk_usage", lambda fs, _: fs.disk_usage(MODEL_ID), len(PAYLOAD)),
    Operation("download", _downloaded_with("download"), PAYLOAD),
    Operation("du", lambda fs, _: fs.du(MODEL_ID), len(PAYLOAD)),
    Operation("exists", lambda fs, _: fs.exists(FILE), True),
    Operation(
        "expand_path",
        lambda fs, _: fs.expand_path(MODEL_ID, recursive=True),
        [MODEL_ID, IMMUTABLE_FILE],
    ),
    Operation("find", lambda fs, _: fs.find(MODEL_ID), [IMMUTABLE_FILE]),
    Operation("get", _downloaded_with("get"), PAYLOAD),
    Operation("get_file", _downloaded_with("get_file"), PAYLOAD),
    Operation(
        "get_mapper", lambda fs, _: fs.get_mapper(MODEL_ID)["config.json"], PAYLOAD
    ),
    Operation(
        "glob", lambda fs, _: fs.glob(f"{MODEL_ID}@main/*.json"), [IMMUTABLE_FILE]
    ),
    Operation("head", lambda fs, _: fs.head(FILE, 4), PAYLOAD[:4]),
    Operation("info", lambda fs, _: fs.info(FILE), FILE_INFO),
    Operation("invalidate_cache", lambda fs, _: fs.invalidate_cache(FILE), None),
    Operation("isdir", lambda fs, _: fs.isdir(MODEL_ID), True),
    Operation("isfile", lambda fs, _: fs.isfile(FILE), True),
    Operation("lexists", lambda fs, _: fs.lexists(FILE), True),
    Operation("listdir", lambda fs, _: fs.listdir(MODEL_ID), [FILE_INFO]),
    Operation("ls", lambda fs, _: fs.ls(MODEL_ID), [FILE_INFO]),
    Operation("open", _read_opened, PAYLOAD, variant="rb"),
    Operation("read_block", lambda fs, _: fs.read_block(FILE, 0, 4), PAYLOAD[:4]),
    Operation("read_bytes", lambda fs, _: fs.read_bytes(FILE), PAYLOAD),
    Operation("read_text", lambda fs, _: fs.read_text(FILE), PAYLOAD.decode()),
    Operation(
        "resolve_path",
        lambda fs, _: fs.resolve_path(FILE),
        HfFileSystemResolvedRepositoryPath("model", MODEL_ID, "main", "config.json"),
    ),
    Operation("size", lambda fs, _: fs.size(FILE), len(PAYLOAD)),
    Operation("sizes", lambda fs, _: fs.sizes([FILE]), [len(PAYLOAD)]),
    Operation("stat", lambda fs, _: fs.stat(FILE), FILE_INFO),
    Operation("tail", lambda fs, _: fs.tail(FILE, 5), PAYLOAD[-5:]),
    Operation("tree", lambda fs, _: fs.tree(MODEL_ID), f"{MODEL_ID}\n└── config.json"),
    Operation(
        "ukey",
        lambda fs, _: fs.ukey(FILE),
        hashlib.sha256(str(FILE_INFO).encode()).hexdigest(),
    ),
    Operation(
        "walk",
        lambda fs, _: list(fs.walk(MODEL_ID)),
        [(MODEL_ID, [], ["config.json"])],
    ),
]


REJECTED_OPERATIONS = [
    Operation("copy", lambda fs, _: fs.copy(FILE, NEW_FILE)),
    Operation("cp", lambda fs, _: fs.cp(FILE, NEW_FILE)),
    Operation("cp_file", lambda fs, _: fs.cp_file(FILE, NEW_FILE)),
    Operation("created", lambda fs, _: fs.created(FILE)),
    Operation("delete", lambda fs, _: fs.delete(FILE)),
    Operation("end_transaction", lambda fs, _: fs.end_transaction()),
    Operation("get_mapper", _write_mapped, variant="setitem"),
    Operation("makedir", lambda fs, _: fs.makedir(NEW_DIRECTORY)),
    Operation("makedirs", lambda fs, _: fs.makedirs(NEW_DIRECTORY)),
    Operation("mkdir", lambda fs, _: fs.mkdir(NEW_DIRECTORY)),
    Operation("mkdirs", lambda fs, _: fs.mkdirs(NEW_DIRECTORY)),
    Operation("modified", lambda fs, _: fs.modified(FILE)),
    Operation("move", lambda fs, _: fs.move(FILE, NEW_FILE)),
    Operation("mv", lambda fs, _: fs.mv(FILE, NEW_FILE)),
    Operation("open", _write_opened, variant="wb"),
    Operation("pipe", lambda fs, _: fs.pipe(NEW_FILE, b"{}")),
    Operation("pipe_file", lambda fs, _: fs.pipe_file(NEW_FILE, b"{}")),
    Operation("put", _uploaded_with("put")),
    Operation("put_file", _uploaded_with("put_file")),
    Operation("rename", lambda fs, _: fs.rename(FILE, NEW_FILE)),
    Operation("rm", lambda fs, _: fs.rm(FILE)),
    Operation("rm_file", lambda fs, _: fs.rm_file(FILE)),
    Operation("rmdir", lambda fs, _: fs.rmdir(MODEL_ID)),
    Operation("sign", lambda fs, _: fs.sign(FILE)),
    Operation("start_transaction", lambda fs, _: fs.start_transaction()),
    Operation("touch", lambda fs, _: fs.touch(NEW_FILE)),
    Operation("upload", _uploaded_with("upload")),
    Operation("url", lambda fs, _: fs.url(FILE)),
    Operation("write_bytes", lambda fs, _: fs.write_bytes(NEW_FILE, b"{}")),
    Operation("write_text", lambda fs, _: fs.write_text(NEW_FILE, "{}")),
]


def test_filesystem_sweep_covers_every_public_operation() -> None:
    # Given
    filesystem_class = mirror._filesystem_class()
    public_operations = {
        name
        for name in dir(filesystem_class)
        if not name.startswith("_") and callable(getattr(filesystem_class, name))
    }
    expected_unswept = NON_PATH_FILESYSTEM_OPERATIONS
    expected_unknown: set[str] = set()

    # When
    swept = {operation.method for operation in READ_OPERATIONS + REJECTED_OPERATIONS}

    # Then
    assert public_operations - swept == expected_unswept
    assert swept - public_operations == expected_unknown


@pytest.mark.parametrize("operation", _params(READ_OPERATIONS))
def test_filesystem_read_on_mirrored_repo_returns_mirror_data_without_hub(
    filesystem: Any, observed: Observed, tmp_path: Path, operation: Operation
) -> None:
    # Given
    expected_hub_calls: list[str] = []
    expected_network_calls: list[str] = []

    # When
    result = operation.call(filesystem, tmp_path)

    # Then
    assert result == operation.expected
    assert observed.hub_calls == expected_hub_calls
    assert observed.network_calls == expected_network_calls


@pytest.mark.parametrize("operation", _params(REJECTED_OPERATIONS))
def test_filesystem_mutation_on_mirrored_repo_is_rejected_without_hub_or_s3(
    filesystem: Any, observed: Observed, tmp_path: Path, operation: Operation
) -> None:
    # Given
    expected_hub_calls: list[str] = []
    expected_network_calls: list[str] = []
    expected_blob_downloads: list[str] = []

    # When
    with pytest.raises(NotImplementedError):
        operation.call(filesystem, tmp_path)

    # Then
    assert observed.hub_calls == expected_hub_calls
    assert observed.network_calls == expected_network_calls
    assert observed.repository.blob_downloads == expected_blob_downloads


@dataclass(frozen=True)
class Spelling:
    root: str
    repo_type: str
    repo_id: str

    @property
    def immutable_file(self) -> str:
        prefix = "datasets/" if self.repo_type == "dataset" else ""
        return f"{prefix}{self.repo_id}@{COMMIT}/config.json"


@dataclass(frozen=True)
class SpelledRead:
    method: str
    call: Callable[[Any, str, Path], Any]
    expected: Callable[[Spelling], Any]


def _file_info(name: str) -> dict[str, Any]:
    return {"name": name, "size": len(PAYLOAD), "type": "file"}


def _identity(resolved: Any) -> tuple[str, str, str]:
    return resolved.repo_type, resolved.repo_id, resolved.path_in_repo


def _spelled_download(filesystem: Any, root: str, tmp_path: Path) -> bytes:
    destination = tmp_path / "downloaded.json"
    filesystem.get_file(f"{root}/config.json", str(destination))
    return destination.read_bytes()


MIRRORED_SPELLINGS = [
    pytest.param(Spelling("hf://example/model", "model", MODEL_ID), id="hf-url"),
    pytest.param(Spelling("hf::example/model", "model", MODEL_ID), id="hf-chained"),
    pytest.param(
        Spelling("models/example/model", "model", MODEL_ID), id="models-prefix"
    ),
    pytest.param(
        Spelling("hf://models/example/model", "model", MODEL_ID),
        id="hf-url-models-prefix",
    ),
    pytest.param(Spelling("/example/model", "model", MODEL_ID), id="leading-slash"),
    pytest.param(
        Spelling(f"example/model@{COMMIT}", "model", MODEL_ID), id="pinned-commit"
    ),
    pytest.param(
        Spelling("datasets/example/dataset", "dataset", DATASET_ID),
        id="datasets-prefix",
    ),
    pytest.param(
        Spelling("hf::datasets/example/dataset", "dataset", DATASET_ID),
        id="hf-chained-datasets-prefix",
    ),
]


SPELLED_READS = [
    SpelledRead(
        "resolve_path",
        lambda fs, root, _: _identity(fs.resolve_path(f"{root}/config.json")),
        lambda spelling: (spelling.repo_type, spelling.repo_id, "config.json"),
    ),
    SpelledRead(
        "info",
        lambda fs, root, _: fs.info(f"{root}/config.json"),
        lambda spelling: _file_info(spelling.immutable_file),
    ),
    SpelledRead(
        "ls",
        lambda fs, root, _: fs.ls(root),
        lambda spelling: [_file_info(spelling.immutable_file)],
    ),
    SpelledRead(
        "glob",
        lambda fs, root, _: fs.glob(f"{root}/config.json"),
        lambda spelling: [spelling.immutable_file],
    ),
    SpelledRead(
        "find",
        lambda fs, root, _: fs.find(root),
        lambda spelling: [spelling.immutable_file],
    ),
    SpelledRead(
        "exists",
        lambda fs, root, _: fs.exists(f"{root}/config.json"),
        lambda spelling: True,
    ),
    SpelledRead(
        "cat_file",
        lambda fs, root, _: fs.cat_file(f"{root}/config.json"),
        lambda spelling: PAYLOAD,
    ),
    SpelledRead("get_file", _spelled_download, lambda spelling: PAYLOAD),
]


SPELLED_REJECTIONS = [
    pytest.param(lambda fs, root: fs.url(f"{root}/config.json"), id="url"),
    pytest.param(
        lambda fs, root: fs.cp_file(f"{root}/config.json", f"{root}/new.json"),
        id="cp_file",
    ),
    pytest.param(lambda fs, root: fs.rm(f"{root}/config.json"), id="rm"),
    pytest.param(lambda fs, root: fs.rm_file(f"{root}/config.json"), id="rm_file"),
    pytest.param(lambda fs, root: fs.mkdir(f"{root}/new-directory"), id="mkdir"),
    pytest.param(
        lambda fs, root: fs.pipe_file(f"{root}/new.json", b"{}"), id="pipe_file"
    ),
]


@pytest.mark.parametrize(
    "operation", [pytest.param(read, id=read.method) for read in SPELLED_READS]
)
@pytest.mark.parametrize("spelling", MIRRORED_SPELLINGS)
def test_filesystem_read_routes_every_mirrored_spelling_to_mirror(
    filesystem: Any,
    observed: Observed,
    tmp_path: Path,
    spelling: Spelling,
    operation: SpelledRead,
) -> None:
    # Given
    expected_result = operation.expected(spelling)
    expected_hub_calls: list[str] = []

    # When
    result = operation.call(filesystem, spelling.root, tmp_path)

    # Then
    assert result == expected_result
    assert observed.hub_calls == expected_hub_calls
    assert observed.network_calls == []


@pytest.mark.parametrize("reject", SPELLED_REJECTIONS)
@pytest.mark.parametrize("spelling", MIRRORED_SPELLINGS)
def test_filesystem_mutation_rejects_every_mirrored_spelling_without_hub(
    filesystem: Any,
    observed: Observed,
    spelling: Spelling,
    reject: Callable[[Any, str], Any],
) -> None:
    # Given
    expected_hub_calls: list[str] = []

    # When
    with pytest.raises(NotImplementedError):
        reject(filesystem, spelling.root)

    # Then
    assert observed.hub_calls == expected_hub_calls
    assert observed.network_calls == []


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(f"{MODEL_ID}/nested/../config.json", id="parent-segment"),
        pytest.param(f"{MODEL_ID}//config.json", id="empty-segment"),
        pytest.param(f"{MODEL_ID}/nested\\config.json", id="backslash"),
    ],
)
@pytest.mark.parametrize(
    ("method", "expected_error"),
    [
        ("resolve_path", ValueError),
        ("info", ValueError),
        ("cat_file", ValueError),
        ("url", NotImplementedError),
        ("rm", NotImplementedError),
    ],
)
def test_filesystem_non_normalized_mirrored_path_fails_closed_without_hub(
    filesystem: Any,
    observed: Observed,
    method: str,
    expected_error: type[Exception],
    path: str,
) -> None:
    # Given
    expected_hub_calls: list[str] = []

    # When
    with pytest.raises(expected_error):
        getattr(filesystem, method)(path)

    # Then
    assert observed.hub_calls == expected_hub_calls
    assert observed.network_calls == []


def test_filesystem_copy_rejects_mirrored_paths_before_public_hub_calls(
    configured_fallback: FallbackRepository,
) -> None:
    # Given
    filesystem = mirror._filesystem_class()(token="dummy-token")
    expected_error = "read-only"

    # When
    with pytest.raises(NotImplementedError, match=expected_error):
        filesystem.cp_file("example/model/config.json", "other/model/copied.json")

    # Then
    assert HubApi.calls == []


class FilesystemRepository:
    def __init__(self, commit: str, files: dict[str, dict[str, str | int]]) -> None:
        self.commit = commit
        self.files = files

    def resolve(self, repo_type: str, repo_id: str, revision: str) -> str:
        return self.commit

    def manifest(self, repo_type: str, repo_id: str, commit: str) -> dict[str, object]:
        return {"files": self.files}


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
    expected_matches = [
        f"example/model@{commit}/dir",
        f"example/model@{commit}/top.txt",
    ]

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


@pytest.mark.parametrize(
    "pattern",
    [
        "example/model/*.json",
        "hf://models/example/model@main/*.json",
        "example/model@feature%2ffoo/*.json",
        "example/model@feature%2Ffoo/*.json",
    ],
)
def test_filesystem_glob_normalizes_repository_and_revision_spelling(
    monkeypatch: pytest.MonkeyPatch, pattern: str
) -> None:
    # Given
    repository = SnapshotRepository()
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", False)
    filesystem = mirror._filesystem_class()()
    expected_matches = [f"example/model@{repository.commit}/config.json"]

    # When
    matches = filesystem.glob(pattern)

    # Then
    assert matches == expected_matches


def test_filesystem_glob_honors_recursive_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    repository = SnapshotRepository()
    repository.files["data/nested/other.txt"] = b"nested"
    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", False)
    filesystem = mirror._filesystem_class()()
    expected_matches = [f"example/model@{repository.commit}/data/part.txt"]

    # When
    matches = filesystem.glob("example/model@main/data/**/*.txt", maxdepth=1)

    # Then
    assert matches == expected_matches


def _filesystem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: BlobRepository
):
    monkeypatch.setattr(mirror, "_BACKEND", backend)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    return mirror._filesystem_class()()


def test_filesystem_cat_file_rejects_same_size_blob_with_wrong_manifest_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_s3: BlobRepository
) -> None:
    # Given
    fake_s3.object_payload = b"EVIL"
    filesystem = _filesystem(monkeypatch, tmp_path, fake_s3)
    path = "example/model@main/config.json"
    expected_error = "Integrity check failed"
    expected_download_count = 1
    expected_range_reads = 0

    # When
    with pytest.raises(OSError, match=expected_error):
        filesystem.cat_file(path)

    # Then
    assert fake_s3.download_count == expected_download_count
    assert fake_s3.read_count == expected_range_reads


def test_filesystem_open_supports_seek_after_verifying_full_blob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_s3: BlobRepository
) -> None:
    # Given
    filesystem = _filesystem(monkeypatch, tmp_path, fake_s3)
    path = "example/model@main/config.json"
    offset = 1
    read_size = 2
    expected_bytes = b"OO"
    expected_size = 4
    expected_download_count = 1
    expected_range_reads = 0

    # When
    with filesystem.open(path, "rb") as source:
        source.seek(offset)
        actual_bytes = source.read(read_size)
        actual_size = source.size
    is_closed = source.closed

    # Then
    assert actual_bytes == expected_bytes
    assert actual_size == expected_size
    assert is_closed
    assert fake_s3.download_count == expected_download_count
    assert fake_s3.read_count == expected_range_reads


def test_filesystem_cat_file_supports_negative_ranges_after_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_s3: BlobRepository
) -> None:
    # Given
    filesystem = _filesystem(monkeypatch, tmp_path, fake_s3)
    path = "example/model@main/config.json"
    start = -3
    expected_bytes = b"OOD"
    expected_download_count = 1
    expected_range_reads = 0

    # When
    actual_bytes = filesystem.cat_file(path, start=start)

    # Then
    assert actual_bytes == expected_bytes
    assert fake_s3.download_count == expected_download_count
    assert fake_s3.read_count == expected_range_reads


def test_filesystem_revalidates_a_preexisting_corrupted_blob_before_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_s3: BlobRepository
) -> None:
    # Given
    filesystem = _filesystem(monkeypatch, tmp_path, fake_s3)
    path = "example/model@main/config.json"
    entry = fake_s3.entry("model", "example/model", fake_s3.commit, "config.json")
    blob = mirror._cache_layout(tmp_path, "model", "example/model").blob(entry)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"EVIL")
    expected_bytes = b"GOOD"
    expected_download_count = 1

    # When
    actual_bytes = filesystem.cat_file(path)

    # Then
    assert actual_bytes == expected_bytes
    assert fake_s3.download_count == expected_download_count


def test_filesystem_ignores_a_preexisting_unverified_snapshot_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_s3: BlobRepository
) -> None:
    # Given
    path = "example/model@main/config.json"
    entry = fake_s3.entry("model", "example/model", fake_s3.commit, "config.json")
    snapshot_file = mirror._cache_layout(tmp_path, "model", "example/model").file(
        fake_s3.commit, "config.json"
    )
    snapshot_file.parent.mkdir(parents=True)
    snapshot_file.write_bytes(b"EVIL")
    filesystem = _filesystem(monkeypatch, tmp_path, fake_s3)
    expected_bytes = b"GOOD"
    expected_blob = mirror._cache_layout(tmp_path, "model", "example/model").blob(entry)

    # When
    actual_bytes = filesystem.cat_file(path)

    # Then
    assert actual_bytes == expected_bytes
    assert expected_blob.read_bytes() == expected_bytes
