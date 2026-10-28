"""Lazy datasets integration without importing consumers before install."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
from collections.abc import Callable, Sequence
from functools import partial
from types import ModuleType

_PUBLIC_HF_ENDPOINT = "https://huggingface.co"


def _patch_datasets_config(config: ModuleType, *, allow_hub_fallback: bool) -> None:
    config.__dict__["USE_PARQUET_EXPORT"] = False
    if not allow_hub_fallback:
        # Matching Hub URLs become hf:// paths for the mirror instead of plain HTTP.
        config.__dict__["HF_ENDPOINT"] = _PUBLIC_HF_ENDPOINT
        config.__dict__["HUB_DATASETS_URL"] = (
            _PUBLIC_HF_ENDPOINT + "/datasets/{repo_id}/resolve/{revision}/{path}"
        )
        config.__dict__["HF_UPDATE_DOWNLOAD_COUNTS"] = False


def _patch_datasets_hub(hub: ModuleType, *, allow_hub_fallback: bool) -> None:
    if not allow_hub_fallback:
        # Build repo file URLs under the endpoint datasets maps to hf:// paths.
        hub.__dict__["hf_dataset_url"] = partial(
            hub.__dict__["hf_dataset_url"], endpoint=_PUBLIC_HF_ENDPOINT
        )


class _PatchingLoader(importlib.abc.Loader):
    def __init__(
        self, wrapped: importlib.abc.Loader, patch: Callable[[ModuleType], None]
    ) -> None:
        self.wrapped = wrapped
        self.patch = patch

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return (
            self.wrapped.create_module(spec)
            if hasattr(self.wrapped, "create_module")
            else None
        )

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        self.patch(module)


class _DatasetsPatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self, allow_hub_fallback: bool) -> None:
        self.patches: dict[str, Callable[[ModuleType], None]] = {
            "datasets.config": partial(
                _patch_datasets_config, allow_hub_fallback=allow_hub_fallback
            ),
            "datasets.utils.hub": partial(
                _patch_datasets_hub, allow_hub_fallback=allow_hub_fallback
            ),
        }

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        patch = self.patches.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _PatchingLoader(spec.loader, patch)
        return spec
