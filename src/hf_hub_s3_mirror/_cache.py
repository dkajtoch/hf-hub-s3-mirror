"""Verified local blobs, snapshot links, and offline revision refs."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from filelock import FileLock

from hf_hub_s3_mirror._artifacts import (
    FileEntry,
    RepoType,
    _REPO_DIRECTORIES,
    _is_commit,
    _sha256_file,
)
from hf_hub_s3_mirror._errors import _hub_error


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


def _cache_ref_component(component: str) -> str:
    if component == ".":
        return "%2E"
    if component == "..":
        return "%2E%2E"
    return quote(component, safe="") or "%2F"


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
    entry: FileEntry,
    blob: Path,
    download: Callable[[str], None],
    location: str,
) -> None:
    blob.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=blob.parent, prefix=".download-")
    os.close(descriptor)
    try:
        download(temporary)
        digest, size = _sha256_file(temporary)
        if size != entry["size"] or digest != entry["blob_sha256"]:
            raise OSError(f"Integrity check failed for {location}")
        os.replace(temporary, blob)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _link_or_copy(source: Path, destination: Path, entry: FileEntry) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    link_target = os.path.relpath(source, destination.parent)
    if destination.is_symlink() and os.readlink(destination) == link_target:
        return
    temporary = destination.parent / f".tmp-{os.urandom(8).hex()}"
    try:
        try:
            temporary.symlink_to(link_target)
        except OSError:
            if not destination.is_symlink() and _valid_cached_blob(destination, entry):
                return
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_cached_ref(layout: _CacheLayout, revision: str, commit: str) -> None:
    if revision == commit:
        return
    ref = layout.ref(revision)
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(commit)


@contextmanager
def _verified_blob(
    layout: _CacheLayout,
    entry: FileEntry,
    download: Callable[[str], None],
    location: str,
    force: bool,
) -> Iterator[Path]:
    lock_path = layout.lock(entry)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        blob = layout.blob(entry)
        if force or not _valid_cached_blob(blob, entry):
            _download_verified_blob(entry, blob, download, location)
        yield blob


def _cached_verified_blob(
    repo_type: RepoType,
    repo_id: str,
    commit: str,
    filename: str,
    entry: FileEntry,
    cache_dir: str | Path | None,
    force: bool = False,
    *,
    download: Callable[[str], None],
) -> Path:
    layout = _cache_layout(cache_dir, repo_type, repo_id)
    with _verified_blob(
        layout, entry, download, f"{repo_id}@{commit}/{filename}", force
    ) as blob:
        return blob.absolute()


def _cache_file(
    repo_type: RepoType,
    repo_id: str,
    revision: str,
    commit: str,
    filename: str,
    entry: FileEntry,
    cache_dir: str | Path | None,
    force: bool = False,
    *,
    download: Callable[[str], None],
) -> Path:
    """Cache a manifest-verified blob and expose it in the Hugging Face layout."""
    layout = _cache_layout(cache_dir, repo_type, repo_id)
    target = layout.file(commit, filename)
    with _verified_blob(
        layout, entry, download, f"{repo_id}@{commit}/{filename}", force
    ) as blob:
        _link_or_copy(blob, target, entry)
        _write_cached_ref(layout, revision, commit)
    return target.absolute()


def _cached_commit(layout: _CacheLayout, repo_id: str, revision: str) -> str:
    if _is_commit(revision):
        return revision
    try:
        commit = layout.ref(revision).read_text().strip()
    except FileNotFoundError as error:
        raise _hub_error("local", f"No cached revision for {repo_id}@{revision}", error)
    if not _is_commit(commit):
        raise _hub_error("local", f"Invalid cached revision for {repo_id}@{revision}")
    return commit
