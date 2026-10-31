"""Read-only Hugging Face Hub compatibility layer backed by an S3 mirror.

Call :func:`install` before importing datasets or transformers. The adapter is
tested against huggingface_hub 1.33.0, datasets 5.0.1, and transformers 5.12.1.
"""

from __future__ import annotations

import inspect
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

from hf_hub_s3_mirror._artifacts import (
    FileEntry,
    RepoType,
    _DEFAULT_MIRROR_PREFIX,
    _DEFAULT_REVISION,
    _REPO_DIRECTORIES,
    _validate_repo_id,
    _validate_repo_path,
    _validate_repo_type,
)
from hf_hub_s3_mirror._cache import (
    _cache_file,
    _cache_layout,
    _cached_commit,
    _write_cached_ref,
)
from hf_hub_s3_mirror._datasets import _DatasetsPatchFinder
from hf_hub_s3_mirror._errors import _hub_error
from hf_hub_s3_mirror._filesystem import create_filesystem
from hf_hub_s3_mirror._repository import _S3Repository, _validate_bucket

_FAIL_CLOSED_HF_ENDPOINT = "https://hf-mirror.invalid"
_API_HELPERS = (
    "repo_info",
    "model_info",
    "dataset_info",
    "list_repo_tree",
    "list_repo_files",
    "repo_exists",
    "revision_exists",
    "file_exists",
)
_SAFE_HUB_API_METHODS = frozenset(
    {
        "whoami",
        "get_token_permission",
        "get_full_repo_name",
    }
)
_TRUE_VALUES = {"1", "true", "yes", "on"}
_INSTALLED = False
_BACKEND: _S3Repository | None = None
_HUB_FALLBACK = False
_UPSTREAM: dict[str, Any] = {}


def _repo() -> _S3Repository:
    if _BACKEND is None:
        raise RuntimeError("S3 mirror is not installed")
    return _BACKEND


def _use_hub(
    repo_type: str | None, repo_id: str, *, local_files_only: bool = False
) -> bool:
    """Route a repository absent from the mirror to the Hub when fallback is on.

    Online, a mirrored repository is never mixed with Hub reads: its manifest stays
    authoritative, so missing revisions or files still fail closed.
    Offline fallback uses the Hub's local resolver without querying S3.
    """
    if not _HUB_FALLBACK:
        return False
    if local_files_only:
        return True
    try:
        validated_repo_type = _validate_repo_type(repo_type)
        _validate_repo_id(repo_id)
    except ValueError:
        return True
    return not _repo().has_repository(validated_repo_type, repo_id)


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
    if _use_hub(repo_type, repo_id, local_files_only=local_files_only):
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
    if local_files_only and force_download:
        raise ValueError(
            "Cannot pass 'force_download=True' and 'local_files_only=True' at the same time."
        )
    if local_files_only and local_dir is not None:
        local_dir = Path(local_dir).expanduser().resolve()
        local_file = local_dir / filename
        if local_file.is_file():
            return str(local_file)
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
            download=partial(
                repository.download, validated_repo_type, repo_id, commit, filename
            ),
        )
    if local_dir is not None:
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, destination)
        target = destination
    return str(target.absolute())


def _selected_files(
    files: Sequence[str] | Mapping[str, FileEntry],
    allow_patterns: list[str] | str | None,
    ignore_patterns: list[str] | str | None,
) -> list[str]:
    from huggingface_hub.utils import filter_repo_objects

    return list(
        filter_repo_objects(
            files,
            allow_patterns=allow_patterns or None,
            ignore_patterns=ignore_patterns or None,
        )
    )


def _copy_files(
    source_root: Path, filenames: Sequence[str], destination_root: Path
) -> None:
    for filename in filenames:
        destination = destination_root / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / filename, destination)


def _snapshot_result(snapshot: Path, local_dir: str | Path | None) -> str:
    destination = Path(local_dir) if local_dir is not None else snapshot
    destination.mkdir(parents=True, exist_ok=True)
    return str(destination.absolute())


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
    force_download: bool = False,
    **kwargs: Any,
) -> str:
    if _use_hub(repo_type, repo_id, local_files_only=local_files_only):
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
            force_download=force_download,
            **kwargs,
        )
    del kwargs
    validated_repo_type = _validate_repo_type(repo_type)
    _validate_repo_id(repo_id)
    revision = revision or _DEFAULT_REVISION
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
    selected = _selected_files(files, allow_patterns, ignore_patterns)

    if local_files_only:
        if local_dir is not None:
            _copy_files(snapshot, selected, Path(local_dir))
        return _snapshot_result(snapshot, local_dir)

    def download_file(name: str) -> str:
        return hf_hub_download(
            repo_id,
            name,
            repo_type=validated_repo_type,
            revision=commit,
            cache_dir=cache_dir,
            local_dir=local_dir,
            force_download=force_download,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(download_file, selected))
    _write_cached_ref(layout, revision, commit)
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

    def repo_exists(
        self, repo_id: str, *, repo_type: str | None = None, **kwargs: Any
    ) -> bool:
        if _use_hub(repo_type, repo_id):
            return self._hub().repo_exists(repo_id, repo_type=repo_type, **kwargs)
        del kwargs
        return _repo().has_repository(_validate_repo_type(repo_type), repo_id)

    def revision_exists(
        self,
        repo_id: str,
        revision: str,
        *,
        repo_type: str | None = None,
        **kwargs: Any,
    ) -> bool:
        from huggingface_hub.errors import (
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        if _use_hub(repo_type, repo_id):
            return self._hub().revision_exists(
                repo_id, revision, repo_type=repo_type, **kwargs
            )
        del kwargs
        try:
            _repo().resolve(_validate_repo_type(repo_type), repo_id, revision)
        except (RepositoryNotFoundError, RevisionNotFoundError):
            return False
        return True

    def file_exists(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ) -> bool:
        from huggingface_hub.errors import (
            EntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        if _use_hub(repo_type, repo_id):
            return self._hub().file_exists(
                repo_id,
                filename,
                repo_type=repo_type,
                revision=revision,
                **kwargs,
            )
        del kwargs
        validated_repo_type = _validate_repo_type(repo_type)
        try:
            repository = _repo()
            commit = repository.resolve(validated_repo_type, repo_id, revision)
            repository.entry(validated_repo_type, repo_id, commit, filename)
        except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
            return False
        return True

    @staticmethod
    def _downloads_from_hub(repo_id: str, kwargs: Mapping[str, Any]) -> bool:
        return _use_hub(
            kwargs.get("repo_type"),
            repo_id,
            local_files_only=kwargs.get("local_files_only", False),
        )

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: Any) -> str:
        if self._downloads_from_hub(repo_id, kwargs):
            return self._hub().hf_hub_download(repo_id, filename, **kwargs)
        return hf_hub_download(repo_id, filename, **kwargs)

    def snapshot_download(self, repo_id: str, **kwargs: Any) -> str:
        if self._downloads_from_hub(repo_id, kwargs):
            return self._hub().snapshot_download(repo_id, **kwargs)
        return snapshot_download(repo_id, **kwargs)

    def __getattr__(self, name: str):
        if _HUB_FALLBACK and name in _SAFE_HUB_API_METHODS:
            return getattr(self._hub(), name)
        upstream_method = getattr(_UPSTREAM.get("HfApi"), name, None)
        if callable(upstream_method) and not name.startswith("_"):
            signature = inspect.signature(upstream_method)

            def guarded_hub_method(*args: Any, **kwargs: Any):
                bound = signature.bind_partial(self, *args, **kwargs)
                repo_id = bound.arguments.get("repo_id")
                # Without a repo_type parameter the target type is implicit (e.g.
                # get_dataset_leaderboard), so the id must be unmirrored for every type.
                repo_types = (
                    [bound.arguments.get("repo_type")]
                    if "repo_type" in signature.parameters
                    else list(_REPO_DIRECTORIES)
                )
                if (
                    _HUB_FALLBACK
                    and isinstance(repo_id, str)
                    and all(_use_hub(repo_type, repo_id) for repo_type in repo_types)
                ):
                    return getattr(self._hub(), name)(*args, **kwargs)
                raise NotImplementedError(
                    f"Hub operation {name!r} is unavailable for mirrored repositories: "
                    "the S3 mirror is read-only"
                )

            return guarded_hub_method
        raise NotImplementedError(
            f"Hub operation {name!r} is unavailable: the S3 mirror is read-only"
        )


def _filesystem_class() -> type:
    return create_filesystem(_repo, _use_hub, _HUB_FALLBACK)


def _install_api_helpers(hub_module: ModuleType, api_module: ModuleType) -> None:
    original_api_class = _UPSTREAM["HfApi"]
    module_helpers = {
        name
        for name, value in vars(api_module).items()
        if inspect.ismethod(value) and isinstance(value.__self__, original_api_class)
    }
    api = S3HfApi()
    setattr(api_module, "api", api)
    for name in module_helpers | set(_API_HELPERS):
        helper = getattr(api, name)
        setattr(api_module, name, helper)
        if hasattr(hub_module, name) or name in _API_HELPERS:
            setattr(hub_module, name, helper)


def install(
    *,
    bucket: str | Callable[[], str] | None = None,
    prefix: str | None = None,
    allow_hub_fallback: bool | None = None,
) -> None:
    """Install the mirror for this process before importing Hub consumers.

    The S3 bucket must be supplied explicitly or by ``HF_S3_MIRROR_BUCKET``.
    A callable bucket is resolved on first S3 access instead of at install.
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
    resolved_bucket: str | Callable[[], str] | None = bucket or os.environ.get(
        "HF_S3_MIRROR_BUCKET"
    )
    resolved_prefix = prefix or os.environ.get(
        "HF_S3_MIRROR_PREFIX", _DEFAULT_MIRROR_PREFIX
    )
    if not callable(resolved_bucket):
        resolved_bucket = _validate_bucket(resolved_bucket)
    if not resolved_prefix.strip("/"):
        raise ValueError("The mirror prefix must not be empty")
    if _INSTALLED:
        assert _BACKEND is not None
        if (_BACKEND.bucket_source, _BACKEND.prefix, _HUB_FALLBACK) != (
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
    # Hub agent detection otherwise fetches a registry even for cache-only reads.
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if not allow_hub_fallback:
        # A call path that escapes our monkeypatches (or a child process) must fail
        # on an unresolvable HTTPS host and never carry the implicit Hub token.
        os.environ["HF_ENDPOINT"] = _FAIL_CLOSED_HF_ENDPOINT
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
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
    hffs = filesystem()
    patches = {
        "HfApi": S3HfApi,
        "HfFileSystem": filesystem,
        "hffs": hffs,
        "hf_hub_download": hf_hub_download,
        "snapshot_download": snapshot_download,
    }
    for name, value in patches.items():
        setattr(huggingface_hub, name, value)
    setattr(huggingface_hub.hf_api, "HfApi", S3HfApi)
    _install_api_helpers(huggingface_hub, huggingface_hub.hf_api)
    setattr(huggingface_hub.hf_file_system, "HfFileSystem", filesystem)
    setattr(huggingface_hub.hf_file_system, "hffs", hffs)
    setattr(huggingface_hub.file_download, "hf_hub_download", hf_hub_download)
    setattr(huggingface_hub._snapshot_download, "snapshot_download", snapshot_download)
    sys.meta_path.insert(0, _DatasetsPatchFinder(allow_hub_fallback))
    _INSTALLED = True
