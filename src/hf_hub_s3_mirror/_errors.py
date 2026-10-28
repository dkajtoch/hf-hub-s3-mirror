"""Translate mirror misses into the pinned Hugging Face exception contract."""

from __future__ import annotations

from typing import Literal


def _hub_error(
    kind: Literal["repo", "revision", "entry", "local"],
    message: str,
    cause: BaseException | None = None,
) -> Exception:
    from huggingface_hub import errors
    from httpx import Request, Response

    response = Response(404, request=Request("GET", "s3://private-hugging-face-mirror"))
    remote_errors: dict[str, type[errors.HfHubHTTPError]] = {
        "repo": errors.RepositoryNotFoundError,
        "revision": errors.RevisionNotFoundError,
        "entry": errors.RemoteEntryNotFoundError,
    }
    error = (
        errors.LocalEntryNotFoundError(message)
        if kind == "local"
        else remote_errors[kind](message, response=response)
    )
    error.__cause__ = cause
    return error
