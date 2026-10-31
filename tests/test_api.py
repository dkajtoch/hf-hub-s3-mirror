"""Hugging Face API routing, helper binding, errors, and exhaustive fallback guards."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from _support import (
    COMMIT,
    DATASET_ID,
    MODEL_ID,
    PAYLOAD,
    UPSTREAM_API_METHODS,
    DownloadRepository,
    FallbackRepository,
    HubApi,
    Observed,
)
from huggingface_hub import errors
from huggingface_hub.hf_api import HfApi as UpstreamHfApi

from hf_hub_s3_mirror import mirror


@pytest.mark.parametrize(
    ("kind", "expected_class"),
    [
        ("repo", errors.RepositoryNotFoundError),
        ("revision", errors.RevisionNotFoundError),
        ("entry", errors.EntryNotFoundError),
        ("local", errors.LocalEntryNotFoundError),
    ],
)
def test_mirror_missing_errors_preserve_hub_exception_contract(
    kind: str, expected_class: type[Exception]
) -> None:
    # Given
    expected_message = "Fixture is absent from the mirror"
    expected_cause = KeyError("fixture")

    # When
    actual_error = mirror._hub_error(kind, expected_message, expected_cause)

    # Then
    assert isinstance(actual_error, expected_class)
    assert expected_message in str(actual_error)
    assert actual_error.__cause__ is expected_cause


@pytest.mark.parametrize("kind", ["repo", "revision", "entry"])
def test_mirror_remote_missing_errors_have_a_synthetic_not_found_response(
    kind: str,
) -> None:
    # Given
    expected_status = 404
    expected_url = "s3://private-hugging-face-mirror"

    # When
    actual_error = mirror._hub_error(kind, "Fixture is absent from the mirror")

    # Then
    assert actual_error.response.status_code == expected_status
    assert str(actual_error.response.url) == expected_url


@dataclass(frozen=True)
class Target:
    repo_type: str
    repo_id: str


MODEL = Target("model", MODEL_ID)


DATASET = Target("dataset", DATASET_ID)


def _summary(info: Any) -> tuple[str, str, list[str]]:
    return info.id, info.sha, [sibling.rfilename for sibling in info.siblings]


def _signature(name: str) -> inspect.Signature:
    return inspect.signature(getattr(UpstreamHfApi, name))


def _minimal_arguments(name: str, target: Target | None = None) -> dict[str, Any]:
    parameters = _signature(name).parameters
    arguments: dict[str, Any] = {
        parameter.name: f"<{parameter.name}>"
        for parameter in parameters.values()
        if parameter.name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    if target is not None:
        arguments["repo_id"] = target.repo_id
        if "repo_type" in parameters:
            arguments["repo_type"] = target.repo_type
    return arguments


MIRROR_API_CALLS: dict[str, Callable[[Any, Target], Any]] = {
    "repo_info": lambda api, target: _summary(
        api.repo_info(target.repo_id, repo_type=target.repo_type)
    ),
    "model_info": lambda api, target: _summary(api.model_info(target.repo_id)),
    "dataset_info": lambda api, target: _summary(api.dataset_info(target.repo_id)),
    "list_repo_tree": lambda api, target: [
        (entry.path, entry.size)
        for entry in api.list_repo_tree(target.repo_id, repo_type=target.repo_type)
    ],
    "list_repo_files": lambda api, target: api.list_repo_files(
        target.repo_id, repo_type=target.repo_type
    ),
    "repo_exists": lambda api, target: api.repo_exists(
        target.repo_id, repo_type=target.repo_type
    ),
    "revision_exists": lambda api, target: api.revision_exists(
        target.repo_id, COMMIT, repo_type=target.repo_type
    ),
    "file_exists": lambda api, target: api.file_exists(
        target.repo_id, "config.json", repo_type=target.repo_type
    ),
    "hf_hub_download": lambda api, target: Path(
        api.hf_hub_download(target.repo_id, "config.json", repo_type=target.repo_type)
    ).read_bytes(),
    "snapshot_download": lambda api, target: (
        Path(api.snapshot_download(target.repo_id, repo_type=target.repo_type))
        / "config.json"
    ).read_bytes(),
}


MIRROR_API_CASES = [
    ("repo_info", MODEL, (MODEL_ID, COMMIT, ["config.json"])),
    ("repo_info", DATASET, (DATASET_ID, COMMIT, ["config.json"])),
    ("model_info", MODEL, (MODEL_ID, COMMIT, ["config.json"])),
    ("dataset_info", DATASET, (DATASET_ID, COMMIT, ["config.json"])),
    ("list_repo_tree", MODEL, [("config.json", len(PAYLOAD))]),
    ("list_repo_tree", DATASET, [("config.json", len(PAYLOAD))]),
    ("list_repo_files", MODEL, ["config.json"]),
    ("list_repo_files", DATASET, ["config.json"]),
    ("repo_exists", MODEL, True),
    ("repo_exists", DATASET, True),
    ("revision_exists", MODEL, True),
    ("revision_exists", DATASET, True),
    ("file_exists", MODEL, True),
    ("file_exists", DATASET, True),
    ("hf_hub_download", MODEL, PAYLOAD),
    ("hf_hub_download", DATASET, PAYLOAD),
    ("snapshot_download", MODEL, PAYLOAD),
    ("snapshot_download", DATASET, PAYLOAD),
]


REPO_SCOPED_API_METHODS = [
    name for name in UPSTREAM_API_METHODS if "repo_id" in _signature(name).parameters
]


GUARDED_API_METHODS = [
    name for name in REPO_SCOPED_API_METHODS if name not in MIRROR_API_CALLS
]


SAFE_API_METHODS = [
    name for name in UPSTREAM_API_METHODS if name in mirror._SAFE_HUB_API_METHODS
]


UNSCOPED_API_METHODS = [
    name
    for name in UPSTREAM_API_METHODS
    if name not in REPO_SCOPED_API_METHODS and name not in SAFE_API_METHODS
]


def test_api_sweep_covers_every_mirror_implemented_method() -> None:
    # Given
    expected_methods = {
        name for name in vars(mirror.S3HfApi) if not name.startswith("_")
    }

    # When
    swept = {method for method, _, _ in MIRROR_API_CASES}

    # Then
    assert swept == expected_methods
    assert set(MIRROR_API_CALLS) == expected_methods


@pytest.mark.parametrize(
    ("method", "target", "expected"),
    [
        pytest.param(method, target, expected, id=f"{method}-{target.repo_type}")
        for method, target, expected in MIRROR_API_CASES
    ],
)
def test_api_mirror_method_serves_mirror_data_without_hub(
    observed: Observed, method: str, target: Target, expected: Any
) -> None:
    # Given
    api = mirror.S3HfApi(token="dummy-token")
    expected_hub_calls: list[str] = []

    # When
    result = MIRROR_API_CALLS[method](api, target)

    # Then
    assert result == expected
    assert observed.hub_calls == expected_hub_calls
    assert observed.network_calls == []


@pytest.mark.parametrize("target", [MODEL, DATASET], ids=["model", "dataset"])
@pytest.mark.parametrize("method", GUARDED_API_METHODS)
def test_api_repo_scoped_method_rejects_mirrored_repo_without_hub(
    observed: Observed, method: str, target: Target
) -> None:
    # Given
    api = mirror.S3HfApi(token="dummy-token")
    arguments = _minimal_arguments(method, target)
    expected_hub_calls: list[str] = []

    # When
    with pytest.raises(NotImplementedError, match="unavailable for mirrored"):
        getattr(api, method)(**arguments)

    # Then
    assert observed.hub_calls == expected_hub_calls
    assert observed.network_calls == []


@pytest.mark.parametrize("method", UNSCOPED_API_METHODS)
def test_api_method_without_repo_id_is_unavailable_without_hub(
    observed: Observed, method: str
) -> None:
    # Given
    api = mirror.S3HfApi(token="dummy-token")
    expected_hub_calls: list[str] = []

    # When
    with pytest.raises(NotImplementedError, match="read-only"):
        getattr(api, method)(**_minimal_arguments(method))

    # Then
    assert observed.hub_calls == expected_hub_calls
    assert observed.network_calls == []


@pytest.mark.parametrize("method", SAFE_API_METHODS)
def test_api_safe_identity_method_is_forwarded_to_hub(
    observed: Observed, method: str
) -> None:
    # Given
    api = mirror.S3HfApi(token="dummy-token")
    expected_result = f"hub {method}"
    expected_hub_calls = ["HfApi()", f"HfApi.{method}"]

    # When
    result = getattr(api, method)(**_minimal_arguments(method))

    # Then
    assert result == expected_result
    assert observed.hub_calls == expected_hub_calls


def test_rebound_module_helpers_keep_mirror_contract_without_hub(
    observed: Observed,
) -> None:
    # Given
    upstream_api = mirror._UPSTREAM["HfApi"]()
    hub_module = ModuleType("huggingface_hub")
    api_module = ModuleType("huggingface_hub.hf_api")
    for name in ("model_info", "upload_file"):
        setattr(api_module, name, getattr(upstream_api, name))
        setattr(hub_module, name, getattr(upstream_api, name))
    observed.hub_calls.clear()
    expected_summary = (MODEL_ID, COMMIT, ["config.json"])
    expected_hub_calls: list[str] = []

    # When
    mirror._install_api_helpers(hub_module, api_module)
    summary = _summary(hub_module.model_info(MODEL_ID))
    with pytest.raises(NotImplementedError, match="unavailable for mirrored"):
        api_module.upload_file(
            path_or_fileobj=b"{}", path_in_repo="new.json", repo_id=MODEL_ID
        )

    # Then
    assert summary == expected_summary
    assert observed.hub_calls == expected_hub_calls
    assert observed.network_calls == []


def test_mirrored_api_checks_never_call_hub_and_missing_file_stays_missing(
    configured_fallback: FallbackRepository,
) -> None:
    # Given
    api = mirror.S3HfApi(token="dummy-token")
    expected_present = True
    expected_missing = False
    expected_hub_calls: list[tuple[str, str]] = []

    # When
    present = api.file_exists("example/model", "config.json")
    missing = api.file_exists("example/model", "not-in-mirror.json")

    # Then
    assert present is expected_present
    assert missing is expected_missing
    assert HubApi.calls == expected_hub_calls


def test_absent_repo_api_and_safe_identity_helper_keep_hub_behavior(
    configured_fallback: FallbackRepository,
) -> None:
    # Given
    api = mirror.S3HfApi(token="dummy-token")
    expected_file_exists = True
    expected_user = "hub-user"
    expected_hub_calls = [("other/model", "config.json")]

    # When
    exists = api.file_exists("other/model", "config.json")
    user = api.whoami()

    # Then
    assert exists is expected_file_exists
    assert user == expected_user
    assert HubApi.calls == expected_hub_calls


@pytest.mark.parametrize("local_files_only", [False, True], ids=["online", "offline"])
@pytest.mark.parametrize(
    ("method", "arguments", "upstream_download", "expected_result"),
    [
        pytest.param(
            "snapshot_download",
            ("other/model",),
            "huggingface_hub._snapshot_download.snapshot_download",
            "hub snapshot",
            id="snapshot",
        ),
        pytest.param(
            "hf_hub_download",
            ("other/model", "config.json"),
            "huggingface_hub.file_download.hf_hub_download",
            "hub file",
            id="file",
        ),
    ],
)
def test_download_for_unmirrored_repo_uses_hub_instance_defaults_and_overrides(
    configured_fallback: FallbackRepository,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    method: str,
    arguments: tuple[str, ...],
    upstream_download: str,
    expected_result: str,
    local_files_only: bool,
) -> None:
    # Given
    download_calls: list[dict[str, Any]] = []
    expected_client_settings: dict[str, Any] = {
        "endpoint": "https://private-hub.example",
        "token": "default-token",
        "library_name": "mirror-client",
        "library_version": "1.0",
        "user_agent": {"component": "test"},
        "headers": {"X-Test": "default"},
    }
    expected_forwarded_settings = {
        name: value
        for name, value in expected_client_settings.items()
        if name != "token"
    }

    def record_download(**kwargs: Any) -> str:
        download_calls.append(kwargs)
        return expected_result

    monkeypatch.setitem(mirror._UPSTREAM, "HfApi", UpstreamHfApi)
    monkeypatch.setattr(upstream_download, record_download)
    api = mirror.S3HfApi(**expected_client_settings)
    expected_tokens = [
        expected_client_settings["token"],
        expected_client_settings["token"],
        False,
        "per-call-token",
    ]
    expected_results = [expected_result] * len(expected_tokens)

    # When
    results = [
        getattr(api, method)(
            *arguments,
            revision="main",
            cache_dir=tmp_path / "default-cache",
            local_files_only=local_files_only,
        ),
        getattr(api, method)(
            *arguments,
            revision="none",
            cache_dir=tmp_path / "none-cache",
            token=None,
            local_files_only=local_files_only,
        ),
        getattr(api, method)(
            *arguments,
            revision="false",
            cache_dir=tmp_path / "false-cache",
            token=False,
            local_files_only=local_files_only,
        ),
        getattr(api, method)(
            *arguments,
            revision="override",
            cache_dir=tmp_path / "override-cache",
            token="per-call-token",
            force_download=True,
            local_files_only=local_files_only,
        ),
    ]

    # Then
    assert results == expected_results
    assert [call["token"] for call in download_calls] == expected_tokens
    for call in download_calls:
        for name, expected_value in expected_forwarded_settings.items():
            assert call[name] == expected_value


def test_snapshot_download_for_mirrored_repo_stays_on_s3(
    configured_fallback: FallbackRepository,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    hub_snapshot_calls: list[dict[str, Any]] = []

    class HubApi:
        def __init__(self, *args: object, **kwargs: object) -> None:
            hub_snapshot_calls.append({"init": kwargs})

        def snapshot_download(self, **kwargs: Any) -> str:
            hub_snapshot_calls.append(kwargs)
            return "hub snapshot"

    monkeypatch.setitem(mirror._UPSTREAM, "HfApi", HubApi)
    api = mirror.S3HfApi(endpoint="https://private-hub.example", token="token")
    expected_payload = configured_fallback.payload
    expected_hub_snapshot_calls: list[dict[str, Any]] = []

    # When
    snapshot = Path(
        api.snapshot_download(
            "example/model", cache_dir=tmp_path / "mirror-cache", max_workers=1
        )
    )

    # Then
    assert (snapshot / "config.json").read_bytes() == expected_payload
    assert hub_snapshot_calls == expected_hub_snapshot_calls


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        pytest.param("hf_hub_download", ("other/model", "config.json"), id="file"),
        pytest.param("snapshot_download", ("other/model",), id="snapshot"),
    ],
)
def test_download_local_files_only_for_unmirrored_repo_uses_no_s3_or_http(
    offline_fallback: list[str],
    method: str,
    arguments: tuple[str, ...],
) -> None:
    # Given
    api = mirror.S3HfApi(endpoint="https://private-hub.example", token="token")
    expected_network_calls: list[str] = []

    # When
    with pytest.raises(errors.LocalEntryNotFoundError):
        getattr(api, method)(*arguments, local_files_only=True)

    # Then
    assert offline_fallback == expected_network_calls


def test_api_routes_repositories_absent_from_mirror_to_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    repository = DownloadRepository(b"", "a" * 40)

    class HubApi:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.token = kwargs.get("token")

        def dataset_info(self, repo_id: str, **kwargs: object) -> str:
            return f"hub:{repo_id}:{self.token}"

        def whoami(self) -> str:
            return "hub-user"

    monkeypatch.setattr(mirror, "_BACKEND", repository)
    monkeypatch.setattr(mirror, "_HUB_FALLBACK", True)
    monkeypatch.setitem(mirror._UPSTREAM, "HfApi", HubApi)
    api = mirror.S3HfApi(token="secret")

    # When
    info = api.dataset_info("other/data", revision="main")
    user = api.whoami()

    # Then
    assert info == "hub:other/data:secret"
    assert user == "hub-user"
