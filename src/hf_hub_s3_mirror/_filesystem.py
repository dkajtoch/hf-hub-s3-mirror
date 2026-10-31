"""Read-only fsspec adapter, constructed after process-wide Hub configuration."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast
from urllib.parse import unquote

from fsspec.callbacks import DEFAULT_CALLBACK

from hf_hub_s3_mirror._artifacts import (
    RepoType,
    _DEFAULT_REVISION,
    _validate_repo_id,
    _validate_repo_path,
    _validate_repo_type,
)
from hf_hub_s3_mirror._cache import _cached_verified_blob
from hf_hub_s3_mirror._repository import _S3Repository

if TYPE_CHECKING:
    from huggingface_hub.hf_file_system import HfFileSystemResolvedRepositoryPath

_RANGE_BLOCK_SIZE = 5 * 1024**2


def _parse_hf_path(
    path: str | Path, revision: str | None = None
) -> HfFileSystemResolvedRepositoryPath:
    import huggingface_hub.hf_file_system as hf_file_system

    path = str(hf_file_system.HfFileSystem._strip_protocol(path)).strip("/")
    type_prefix, _, remainder = path.partition("/")
    repo_type = cast(RepoType, "dataset" if type_prefix == "datasets" else "model")
    if type_prefix in {"datasets", "models"}:
        path = remainder
    pieces = path.split("/")
    if len(pieces) < 2:
        raise NotImplementedError("Repository listing is not supported")
    name = pieces[1]
    raw_revision = None
    if "@" in name:
        name, raw_revision = name.split("@", 1)
        path_revision = unquote(raw_revision)
        if revision is not None and revision != path_revision:
            raise ValueError("Conflicting revisions")
        revision = path_revision
    repo_id = _validate_repo_id(f"{pieces[0]}/{name}")
    path_in_repo = "/".join(pieces[2:])
    return hf_file_system.HfFileSystemResolvedRepositoryPath(
        repo_type,
        repo_id,
        revision or _DEFAULT_REVISION,
        path_in_repo,
        _raw_revision=raw_revision,
    )


def _hf_base(repo_type: RepoType, repo_id: str, revision: str) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"{prefix}{repo_id}@{revision}"


def create_filesystem(
    repository: Callable[[], _S3Repository],
    use_hub: Callable[[str | None, str], bool],
    allow_hub_fallback: bool,
) -> type:
    from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
    from huggingface_hub import HfFileSystem as OriginalHfFileSystem

    class S3HfFileSystem(OriginalHfFileSystem):
        def __init__(
            self, *args: Any, block_size: int | None = None, **kwargs: Any
        ) -> None:
            if allow_hub_fallback:
                OriginalHfFileSystem.__init__(self, *args, **kwargs)
            else:
                AbstractFileSystem.__init__(self)
                self._repo_and_revision_exists_cache: dict[
                    tuple[str, str, str | None], tuple[bool, Exception | None]
                ] = {}
                self._bucket_exists_cache: dict[str, tuple[bool, Exception | None]] = {}
            self.block_size = block_size or _RANGE_BLOCK_SIZE

        def _on_hub(self, path: str) -> bool:
            if not allow_hub_fallback:
                return False
            try:
                resolved = _parse_hf_path(path)
            except (ValueError, NotImplementedError):
                return True
            return use_hub(resolved.repo_type, resolved.repo_id)

        def _resolved(self, path: str, revision: str | None = None):
            resolved = _parse_hf_path(path, revision)
            # Validate after routing: a malformed mirrored path must not reach the Hub.
            if resolved.path_in_repo:
                _validate_repo_path(resolved.path_in_repo)
            return resolved, repository().resolve(
                _validate_repo_type(resolved.repo_type),
                resolved.repo_id,
                resolved.revision,
            )

        def _repository_view(self, path: str, revision: str | None = None):
            resolved, commit = self._resolved(path, revision)
            files = repository().manifest(resolved.repo_type, resolved.repo_id, commit)[
                "files"
            ]
            return (
                resolved,
                commit,
                _hf_base(resolved.repo_type, resolved.repo_id, commit),
                files,
            )

        def _pinned_path(self, path: str | Path, revision: str | None = None) -> str:
            raw_path = str(path)
            resolved, commit = self._resolved(raw_path, revision)
            base = _hf_base(resolved.repo_type, resolved.repo_id, commit)
            pinned_path = (
                f"{base}/{resolved.path_in_repo}" if resolved.path_in_repo else base
            )
            return f"{pinned_path}/" if raw_path.endswith("/") else pinned_path

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

        def glob(self, path: str, maxdepth: int | None = None, **kwargs: Any):
            if self._on_hub(path):
                return super().glob(path, maxdepth=maxdepth, **kwargs)
            resolved = _parse_hf_path(path, kwargs.pop("revision", None))
            repo_type = _validate_repo_type(resolved.repo_type)
            commit = repository().resolve(
                repo_type, resolved.repo_id, resolved.revision
            )
            immutable = _hf_base(repo_type, resolved.repo_id, commit)
            pattern = f"{immutable}/{resolved.path_in_repo}"
            if path.endswith("/") and not pattern.endswith("/"):
                pattern += "/"
            return AbstractFileSystem.glob(self, pattern, maxdepth=maxdepth, **kwargs)

        def find(
            self,
            path: str,
            maxdepth: int | None = None,
            withdirs: bool = False,
            detail: bool = False,
            refresh: bool = False,
            revision: str | None = None,
            **kwargs: Any,
        ):
            if self._on_hub(path):
                return super().find(
                    path,
                    maxdepth=maxdepth,
                    withdirs=withdirs,
                    detail=detail,
                    refresh=refresh,
                    revision=revision,
                    **kwargs,
                )
            del refresh, kwargs
            if maxdepth is not None and maxdepth < 1:
                raise ValueError("maxdepth must be at least 1")
            resolved, _, base, files = self._repository_view(path, revision)
            root = f"{base}/{resolved.path_in_repo}" if resolved.path_in_repo else base
            matches: dict[str, dict[str, Any]] = {}
            if resolved.path_in_repo in files:
                matches[root] = {
                    "name": root,
                    "size": files[resolved.path_in_repo]["size"],
                    "type": "file",
                }
            else:
                prefix = f"{resolved.path_in_repo}/" if resolved.path_in_repo else ""
                root_exists = not resolved.path_in_repo
                for filename, entry in files.items():
                    if not filename.startswith(prefix):
                        continue
                    root_exists = True
                    parts = filename[len(prefix) :].split("/")
                    if maxdepth is None or len(parts) <= maxdepth:
                        name = f"{base}/{filename}"
                        matches[name] = {
                            "name": name,
                            "size": entry["size"],
                            "type": "file",
                        }
                    if withdirs:
                        directory = root
                        for component in parts[:-1][:maxdepth]:
                            directory = f"{directory}/{component}"
                            matches.setdefault(
                                directory,
                                {"name": directory, "size": 0, "type": "directory"},
                            )
                if withdirs and root_exists:
                    matches[root] = {"name": root, "size": 0, "type": "directory"}
            names = sorted(matches)
            return {name: matches[name] for name in names} if detail else names

        def expand_path(
            self,
            path: Any,
            recursive: bool = False,
            maxdepth: int | None = None,
            assume_literal: bool = False,
            **kwargs: Any,
        ) -> list[str]:
            if (
                not recursive
                or not isinstance(path, (str, Path))
                or self._on_hub(str(path))
            ):
                return super().expand_path(
                    path,
                    recursive=recursive,
                    maxdepth=maxdepth,
                    assume_literal=assume_literal,
                    **kwargs,
                )
            revision = kwargs.pop("revision", None)
            return super().expand_path(
                self._pinned_path(path, revision),
                recursive=recursive,
                maxdepth=maxdepth,
                assume_literal=assume_literal,
                **kwargs,
            )

        def get(
            self,
            rpath: Any,
            lpath: Any,
            recursive: bool = False,
            callback: Any = DEFAULT_CALLBACK,
            maxdepth: int | None = None,
            **kwargs: Any,
        ) -> None:
            if isinstance(rpath, list) and all(
                isinstance(path, (str, Path)) and not self._on_hub(str(path))
                for path in rpath
            ):
                revision = kwargs.pop("revision", None)
                pinned_path: Any = [self._pinned_path(path, revision) for path in rpath]
            elif isinstance(rpath, (str, Path)) and not self._on_hub(str(rpath)):
                revision = kwargs.pop("revision", None)
                pinned_path = self._pinned_path(rpath, revision)
            else:
                pinned_path = rpath
            return super().get(
                pinned_path,
                lpath,
                recursive=recursive,
                callback=callback,
                maxdepth=maxdepth,
                **kwargs,
            )

        def url(self, path: str) -> str:
            if self._on_hub(path):
                return super().url(path)
            raise NotImplementedError("Mirrored files do not have public Hub URLs")

        def exists(self, path: str, **kwargs: Any) -> bool:
            if self._on_hub(path):
                return super().exists(path, **kwargs)
            try:
                self.info(path, **kwargs)
                return True
            except (FileNotFoundError, OSError):
                return False

        def get_file(
            self,
            rpath: str,
            lpath: str | Path | BinaryIO | None = None,
            callback: Any = None,
            outfile: BinaryIO | None = None,
            **kwargs: Any,
        ) -> None:
            """Copy mirrored files through ``open`` instead of Hub ``http_get``."""
            callback = callback or DEFAULT_CALLBACK
            if self._on_hub(rpath):
                return super().get_file(
                    rpath, lpath, callback=callback, outfile=outfile, **kwargs
                )
            from fsspec.utils import isfilelike

            revision = kwargs.pop("revision", None)
            if kwargs:
                raise TypeError(f"Unsupported get_file arguments: {sorted(kwargs)}")
            if outfile is None and lpath is None:
                raise ValueError("Either `lpath` or `outfile` must be provided.")
            if outfile is None and isfilelike(lpath):
                outfile = cast(BinaryIO, lpath)
            if outfile is None:
                if not isinstance(lpath, (str, Path)):
                    raise TypeError("lpath must be a local path or file-like object")
                if self.isdir(rpath):
                    Path(lpath).mkdir(parents=True, exist_ok=True)
                    return None
                destination = Path(lpath)
                destination.parent.mkdir(parents=True, exist_ok=True)
                outfile = open(destination, "wb")
                close_file = True
            else:
                close_file = False
            try:
                expected_size = self.info(rpath, revision=revision)["size"]
                callback.set_size(expected_size)
                with self.open(rpath, "rb", revision=revision) as source:
                    while chunk := source.read(self.block_size):
                        outfile.write(chunk)
                        callback.relative_update(len(chunk))
            finally:
                if close_file:
                    outfile.close()

        def _reject_mirrored_write(self, *paths: str) -> None:
            if any(not self._on_hub(path) for path in paths):
                raise NotImplementedError("The S3 mirror is read-only")

        def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
            self._reject_mirrored_write(path)
            return super().mkdir(path, create_parents=create_parents, **kwargs)

        def makedirs(self, path: str, exist_ok: bool = False) -> None:
            self._reject_mirrored_write(path)
            return super().makedirs(path, exist_ok=exist_ok)

        def mkdirs(self, path: str, exist_ok: bool = False) -> None:
            self._reject_mirrored_write(path)
            return super().mkdirs(path, exist_ok=exist_ok)

        def rmdir(self, path: str) -> None:
            self._reject_mirrored_write(path)
            return super().rmdir(path)

        def cp_file(
            self,
            path1: str,
            path2: str,
            revision: str | None = None,
            **kwargs: Any,
        ) -> None:
            self._reject_mirrored_write(path1, path2)
            return super().cp_file(path1, path2, revision=revision, **kwargs)

        def _rm(self, path: str, revision: str | None = None, **kwargs: Any) -> None:
            self._reject_mirrored_write(path)
            return super()._rm(path, revision=revision, **kwargs)

        def rm(
            self,
            path: str,
            recursive: bool = False,
            maxdepth: int | None = None,
            revision: str | None = None,
            **kwargs: Any,
        ) -> None:
            self._reject_mirrored_write(path)
            return super().rm(
                path,
                recursive=recursive,
                maxdepth=maxdepth,
                revision=revision,
                **kwargs,
            )

        def _open(
            self,
            path: str,
            mode: str = "rb",
            block_size: int | None = None,
            revision: str | None = None,
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
            entry = repository().entry(
                resolved.repo_type, resolved.repo_id, commit, resolved.path_in_repo
            )
            blob = _cached_verified_blob(
                resolved.repo_type,
                resolved.repo_id,
                commit,
                resolved.path_in_repo,
                entry,
                None,
                download=partial(
                    repository().download,
                    resolved.repo_type,
                    resolved.repo_id,
                    commit,
                    resolved.path_in_repo,
                ),
            )

            class VerifiedRangeFile(AbstractBufferedFile):
                def _fetch_range(inner, start: int, end: int) -> bytes:
                    with blob.open("rb") as source:
                        source.seek(start)
                        return source.read(end - start)

            return VerifiedRangeFile(
                self,
                str(path),
                mode="rb",
                size=entry["size"],
                block_size=block_size or self.block_size,
            )

    return S3HfFileSystem
