# hf-hub-s3-mirror

An opt-in, S3-backed mirror for Hugging Face Hub model and dataset repositories.
It publishes immutable, SHA-256-addressed blobs and a manifest for each pinned
Hub commit, then exposes those files through the familiar `huggingface_hub` API.
Downloads are verified and cached using the normal Hugging Face cache layout.

This is an early compatibility layer for `huggingface_hub==1.33.0`,
`datasets==5.0.1`, and `transformers==5.12.1`. It patches their Hub-facing
interfaces **for the whole Python process**. It is not a drop-in replacement
for arbitrary versions or a general S3 filesystem.

## Install

```bash
pip install hf-hub-s3-mirror
```

Before the first PyPI release, install this checkout locally with `pip install .`.
The optional `datasets` and `transformers` extras install the tested versions:

```bash
pip install 'hf-hub-s3-mirror[datasets,transformers]'
```

The core dependencies use compatible version ranges, except for Hugging Face Hub:
its version stays pinned because the adapter patches internal interfaces. The
optional integrations also stay pinned to tested versions. In particular,
`datasets==5.0.1` requires `fsspec<=2026.6.0`, so installing the `datasets` extra
selects `fsspec==2026.6.0` even though the core package allows newer versions.

## Publish a pinned Hub revision

```bash
hf-hub-s3-publish dataset 'ORG/NAME@REVISION' --bucket YOUR_BUCKET
hf-hub-s3-publish model 'ORG/NAME@REVISION' --bucket YOUR_BUCKET
hf-hub-s3-publish tokenizer 'ORG/NAME@REVISION' --bucket YOUR_BUCKET
```

`tokenizer` publishes only the files `AutoTokenizer` and `AutoConfig` read
from a model repository (tokenizer and config JSON, vocabularies, chat
templates, and remote-code `*.py` modules), not the weights. Every kind
accepts `--allow-patterns` and `--ignore-patterns` with the same glob
semantics as `snapshot_download`; for `tokenizer`, allow patterns add to its
defaults. Publishing more files of an already published commit extends its
manifest; a file that is already published must keep the same blob. Readers
fail closed on any file the manifest does not list.

`REVISION` may be a branch, tag, or full commit; the publisher resolves it to
an immutable commit and stores the commit in the manifest. You may pass
`--revision REVISION` instead of using `@REVISION`. A revision is required;
the publisher never silently defaults to `main`. For gated repositories,
provide an authorized `HF_TOKEN` in the environment. AWS credentials come
from the normal boto3 credential chain. Use `--region` to override the
configured AWS region. `--dry-run` downloads and hashes the
files without writing to S3.

## Read a mirrored revision

Call `install` before importing `huggingface_hub`, `datasets`, or `transformers`:

```python
from hf_hub_s3_mirror import install

install(bucket="YOUR_BUCKET", prefix="hf_hub")

from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "ORG/NAME", "config.json", revision="FULL_COMMIT_SHA"
)
```

Importing the package does not activate the mirror; only `install` does.
`bucket` and `prefix` default to `HF_S3_MIRROR_BUCKET` and
`HF_S3_MIRROR_PREFIX` when omitted. `bucket` may also be a function returning
the name; it is called on the first S3 access, so `install` can run before
the process knows, for example, its AWS region.

By default installation disables public-Hub fallback: an absent mirrored
repository, revision, or file fails instead of fetching it from the internet.
It also sets `HF_ENDPOINT=https://hf-mirror.invalid` and
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` for the process and its child processes, so a
Hub request that escapes the mirror fails without sending the implicit token.
`datasets` still reads full `https://huggingface.co/datasets/.../resolve/...`
URLs from the mirror and skips its download-count request.

To mirror only some repositories, set `HF_S3_MIRROR_ALLOW_HUB_FALLBACK=1` or
call `install(..., allow_hub_fallback=True)`. Repositories present in the
mirror are then read only from S3. Supported operations on other repositories
use the Hub (including `HF_ENDPOINT`, tokens, and Xet). The mirror stays
authoritative for the repositories it holds: a missing revision or file there
still fails rather than reaching the Hub, so a gated repository is never
requested upstream once it is mirrored.

Unsupported operations on mirrored repositories fail closed, including writes
and public-Hub URL generation. Account-only helpers such as `whoami` remain
available when fallback is enabled.

The runtime needs S3 `GetObject` and `ListBucket` access; publishing also
needs `PutObject`. The package never embeds AWS
credentials or dataset bytes.

## Mirror layout

The publisher stores manifests under
`<prefix>/repos/{models|datasets}/ORG/NAME/snapshots/COMMIT/manifest.json`,
mutable revision refs under `.../refs/`, and shared immutable blobs under
`<prefix>/blobs/sha256/`. The reader verifies blob sizes and SHA-256 hashes
before placing them in the local Hugging Face cache.

`HfFileSystem` also verifies a complete file before returning any of its bytes.
The first read downloads the whole file, even for a small requested range, and
requires enough local cache space for it. Subsequent opens revalidate and reuse
the cached blob; seek and partial reads operate on the verified local file.

## Development

The implementation separates artifact validation, S3 access, verified caching,
filesystem compatibility, and lazy datasets integration. The public downloads,
Hub API adapter, and installation entry point remain in `mirror.py`.

Tests are grouped by contract in four modules: `test_storage.py`, `test_api.py`,
`test_filesystem.py`, and `test_integration.py`. Shared fixtures live in
`conftest.py` and `_support.py`. The integration tests publish fixture repositories
and read their output through the real datasets and tokenizer libraries with
network access blocked; they do not replace a smoke test against your S3 bucket.

The pre-commit hooks check whitespace, YAML/TOML, merge markers, file size,
Ruff linting and formatting, and mypy. They use Python 3.12 for their isolated
environments. CI runs the full suite on Python 3.11 and 3.12 with both the minimum
direct runtime dependencies and the latest versions allowed by the package and
its test extras. Each job checks dependency consistency and logs the resolved
versions. The mypy hook keeps its own tested dependency pins.

The minimum runtime versions are pinned directly in the CI install step. Keep
those pins aligned with the lower bounds in `pyproject.toml`. For local development:

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

```bash
uv tool install pre-commit==4.7.0
pre-commit install
pre-commit run --all-files
```

The full test suite remains a CI check rather than running on every commit.

## Release status

The package has not been published to PyPI. Run the local tests and an
end-to-end read with the intended S3 permissions before releasing it.

Releases follow the same PyPI Trusted Publishing pattern as `fasttext-threaded`:
the tag-triggered workflow tests and builds an sdist plus a universal wheel,
then publishes from a separate job using GitHub OIDC. No PyPI API token is
stored in this repository.

Before the first release, configure the PyPI project's trusted publisher with
owner `dkajtoch`, repository `hf-hub-s3-mirror`, workflow `release.yaml`, and
GitHub environment `pypi`. Configure that environment to require approval.
Then tag a commit on `main` matching the version in `pyproject.toml` (for
example, `v0.1.0a1`). The workflow rejects tags whose version differs or whose
commit is not on `main`. Creating the workflow alone does not publish anything.
