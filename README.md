# hf-hub-s3-mirror

An opt-in, S3-backed mirror for Hugging Face Hub model and dataset repositories.
It publishes immutable, SHA-256-addressed blobs and a manifest for each pinned
Hub commit, then exposes those files through the familiar `huggingface_hub` API.
Downloads are verified and cached using the normal Hugging Face cache layout.

This is an early compatibility layer for `huggingface_hub==0.36.2`,
`datasets==5.0.1`, and `transformers==4.57.6`. It patches their Hub-facing
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

For an entrypoint that cannot call `install` before those imports, import the
package at process startup with `HF_S3_MIRROR_ENABLED=1` and set
`HF_S3_MIRROR_BUCKET` (and optionally `HF_S3_MIRROR_PREFIX`) first. Importing
the package alone does not activate the mirror unless that flag is set.

By default installation disables public-Hub fallback: an absent mirrored
repository, revision, or file fails instead of fetching it from the internet.

To mirror only some repositories, set `HF_S3_MIRROR_ALLOW_HUB_FALLBACK=1` or
call `install(..., allow_hub_fallback=True)`. Repositories present in the
mirror are then read only from S3, and every other repository goes to the Hub
unchanged (including `HF_ENDPOINT`, tokens, and Xet). The mirror stays
authoritative for the repositories it holds: a missing revision or file there
still fails rather than reaching the Hub, so a gated repository is never
requested upstream once it is mirrored.

The runtime needs S3 `GetObject` and `ListBucket` access; publishing also
needs `PutObject`. The package never embeds AWS
credentials or dataset bytes.

## Mirror layout

The publisher stores manifests under
`<prefix>/repos/{models|datasets}/ORG/NAME/snapshots/COMMIT/manifest.json`,
mutable revision refs under `.../refs/`, and shared immutable blobs under
`<prefix>/blobs/sha256/`. The reader verifies blob sizes and SHA-256 hashes
before placing them in the local Hugging Face cache.

## Development

The pre-commit hooks check whitespace, YAML/TOML, merge markers, file size,
Ruff linting and formatting, and mypy. They use Python 3.12 for their isolated
environments; CI also runs the tests on Python 3.11 and 3.12.

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
