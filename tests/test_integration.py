"""Process installation and real datasets/tokenizer consumers with HTTP blocked."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from textwrap import indent

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

_UNSET_HUB_SETTINGS = (
    "HF_ENDPOINT",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "HF_TOKEN",
    "HF_UPDATE_DOWNLOAD_COUNTS",
)


def _run_in_fresh_process(
    script: str, tmp_path: Path, **environment: str
) -> subprocess.CompletedProcess[str]:
    inherited = {
        key: value
        for key, value in os.environ.items()
        if key not in _UNSET_HUB_SETTINGS
    }
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={
            **inherited,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "HF_HOME": str(tmp_path / "hf-home"),
            "HF_HUB_CACHE": str(tmp_path / "cache"),
            **environment,
        },
    )


def _read_published_objects(
    objects: dict[str, bytes],
    tmp_path: Path,
    consumer: str,
    *,
    fallback: bool = False,
) -> subprocess.CompletedProcess[str]:
    store = tmp_path / "published-objects.json"
    store.write_text(
        json.dumps(
            {key: base64.b64encode(value).decode() for key, value in objects.items()}
        )
    )
    script = """
import base64
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

from botocore.exceptions import ClientError
from hf_hub_s3_mirror import install

objects = {
    key: base64.b64decode(value)
    for key, value in json.loads(Path(os.environ['PUBLISHED_OBJECTS']).read_text()).items()
}
network_calls = []

def forbid_network(*args, **kwargs):
    network_calls.append('unexpected network request')
    raise AssertionError('Only the published S3 objects may be read')

class PublishedStore:
    def list_objects_v2(self, **kwargs):
        keys = [key for key in objects if key.startswith(kwargs['Prefix'])]
        return {'Contents': [{'Key': key} for key in keys[:kwargs.get('MaxKeys', 1000)]]}

    def get_object(self, **kwargs):
        key = kwargs['Key']
        if key not in objects:
            raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
        return {'Body': io.BytesIO(objects[key])}

    def download_file(self, bucket, key, destination, **kwargs):
        Path(destination).write_bytes(objects[key])

with (
    patch('boto3.client', return_value=PublishedStore()),
    patch('requests.Session.request', side_effect=forbid_network),
    patch('httpx.Client.send', side_effect=forbid_network),
    patch('aiohttp.ClientSession._request', side_effect=forbid_network),
    patch('socket.socket.connect', side_effect=forbid_network),
):
    install(
        bucket='example-bucket',
        allow_hub_fallback=os.environ['ALLOW_FALLBACK'] == '1',
    )
"""
    return _run_in_fresh_process(
        script + indent(consumer, "    "),
        tmp_path,
        PUBLISHED_OBJECTS=str(store),
        ALLOW_FALLBACK="1" if fallback else "0",
    )


def test_fail_closed_install_uses_unresolvable_https_endpoint_and_no_implicit_token(
    tmp_path: Path,
) -> None:
    # Given
    script = """
import json
import os
from unittest.mock import patch
from urllib.parse import urlsplit
from hf_hub_s3_mirror import install

with patch('boto3.client', return_value=object()):
    install(bucket='audit-fixture-bucket', allow_hub_fallback=False)
from huggingface_hub import constants
from huggingface_hub.utils import build_hf_headers

endpoint = os.environ['HF_ENDPOINT']
print(json.dumps({
    'scheme': urlsplit(endpoint).scheme,
    'reserved_host': (urlsplit(endpoint).hostname or '').endswith('.invalid'),
    'hub_uses_endpoint': constants.ENDPOINT == endpoint,
    'implicit_token_disabled': os.environ.get('HF_HUB_DISABLE_IMPLICIT_TOKEN'),
    'implicit_authorization': 'authorization' in build_hf_headers(),
    'explicit_authorization': build_hf_headers(token='hf_explicit').get(
        'authorization'
    ),
}))
"""
    expected_output = {
        "scheme": "https",
        "reserved_host": True,
        "hub_uses_endpoint": True,
        "implicit_token_disabled": "1",
        "implicit_authorization": False,
        "explicit_authorization": "Bearer hf_explicit",
    }

    # When
    result = _run_in_fresh_process(script, tmp_path, HF_TOKEN="hf_fixture_token")

    # Then
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


def test_fallback_install_leaves_hub_endpoint_and_implicit_token_unset(
    tmp_path: Path,
) -> None:
    # Given
    script = """
import json
import os
from unittest.mock import patch
from hf_hub_s3_mirror import install

with patch('boto3.client', return_value=object()):
    install(bucket='audit-fixture-bucket', allow_hub_fallback=True)
print(json.dumps({
    name: os.environ.get(name)
    for name in ('HF_ENDPOINT', 'HF_HUB_DISABLE_IMPLICIT_TOKEN')
}))
"""
    expected_output = {"HF_ENDPOINT": None, "HF_HUB_DISABLE_IMPLICIT_TOKEN": None}

    # When
    result = _run_in_fresh_process(script, tmp_path)

    # Then
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


def test_fail_closed_datasets_config_turns_full_hub_urls_into_hf_paths(
    tmp_path: Path,
) -> None:
    # Given
    commit = "c" * 40
    script = """
import json
import os
from unittest.mock import patch
from hf_hub_s3_mirror import install

with patch('boto3.client', return_value=object()):
    install(bucket='audit-fixture-bucket', allow_hub_fallback=False)
import datasets.config
from datasets import DownloadConfig
from datasets.utils.file_utils import _prepare_single_hop_path_and_storage_options

url = os.environ['AUDIT_HUB_URL']
path, _ = _prepare_single_hop_path_and_storage_options(url, DownloadConfig())
print(json.dumps({
    'hf_endpoint': datasets.config.HF_ENDPOINT,
    'hub_datasets_url': datasets.config.HUB_DATASETS_URL,
    'prepared_path': path,
}))
"""
    hub_url = (
        f"https://huggingface.co/datasets/example/data/resolve/{commit}/data.parquet"
    )
    expected_output = {
        "hf_endpoint": "https://huggingface.co",
        "hub_datasets_url": (
            "https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"
        ),
        "prepared_path": f"hf://datasets/example/data@{commit}/data.parquet",
    }

    # When
    result = _run_in_fresh_process(script, tmp_path, AUDIT_HUB_URL=hub_url)

    # Then
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


def test_fallback_leaves_datasets_hub_settings_unchanged(tmp_path: Path) -> None:
    # Given
    script = """
import json
from unittest.mock import patch
from hf_hub_s3_mirror import install

with patch('boto3.client', return_value=object()):
    install(bucket='audit-fixture-bucket', allow_hub_fallback=True)
import datasets.config
from datasets.utils.hub import hf_dataset_url

print(json.dumps({
    'hf_endpoint': datasets.config.HF_ENDPOINT,
    'hub_datasets_url': datasets.config.HUB_DATASETS_URL,
    'update_download_counts': datasets.config.HF_UPDATE_DOWNLOAD_COUNTS,
    'dataset_file_url': hf_dataset_url('example/data', 'data.parquet'),
}))
"""
    user_endpoint = "https://hub.example"
    expected_output = {
        "hf_endpoint": user_endpoint,
        "hub_datasets_url": (
            f"{user_endpoint}/datasets/{{repo_id}}/resolve/{{revision}}/{{path}}"
        ),
        "update_download_counts": True,
        "dataset_file_url": (
            f"{user_endpoint}/datasets/example/data/resolve/main/data.parquet"
        ),
    }

    # When
    result = _run_in_fresh_process(script, tmp_path, HF_ENDPOINT=user_endpoint)

    # Then
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


def test_fail_closed_load_dataset_reads_a_full_hub_url_from_the_mirror_without_http(
    tmp_path: Path,
) -> None:
    # Given
    script = """
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from hf_hub_s3_mirror import install

commit = 'd' * 40
buffer = io.BytesIO()
pq.write_table(pa.table({'text': ['hello', 'mirror']}), buffer)
payload = buffer.getvalue()
manifest = {
    'schema_version': 1, 'repo_type': 'dataset', 'repo_id': 'example/data',
    'commit': commit,
    'files': {'data.parquet': {
        'blob_sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload),
    }},
}
http_requests = []

def forbid_http(*args, **kwargs):
    http_requests.append(' '.join(str(arg) for arg in args[:2]))
    raise AssertionError('HTTP is forbidden')

class Transport:
    def get_object(self, **kwargs):
        if not kwargs['Key'].endswith(f'/datasets/example/data/snapshots/{commit}/manifest.json'):
            raise AssertionError('Only the pinned manifest read is expected')
        return {'Body': io.BytesIO(json.dumps(manifest).encode())}
    def download_file(self, bucket, key, destination, **kwargs):
        Path(destination).write_bytes(payload)

with patch('boto3.client', return_value=Transport()), patch(
    'requests.Session.request', side_effect=forbid_http
), patch('httpx.Client.send', side_effect=forbid_http), patch(
    'aiohttp.ClientSession._request', side_effect=forbid_http
):
    install(bucket='audit-fixture-bucket', allow_hub_fallback=False)
    from datasets import load_dataset
    url = f'https://huggingface.co/datasets/example/data/resolve/{commit}/data.parquet'
    dataset = load_dataset('parquet', data_files=url, split='train')
    print(json.dumps({'rows': dataset.to_list(), 'http_requests': http_requests}))
"""
    expected_output = {
        "rows": [{"text": "hello"}, {"text": "mirror"}],
        "http_requests": [],
    }

    # When
    result = _run_in_fresh_process(script, tmp_path)

    # Then
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


def test_fail_closed_load_dataset_reads_a_mirrored_huggingface_yaml_without_http(
    tmp_path: Path,
) -> None:
    # Given
    script = """
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from hf_hub_s3_mirror import install

commit = 'e' * 40
buffer = io.BytesIO()
pq.write_table(pa.table({'text': ['hello']}), buffer)
files = {
    'data.parquet': buffer.getvalue(),
    '.huggingface.yaml': json.dumps({'configs': [{
        'config_name': 'default',
        'data_files': [{'split': 'holdout', 'path': 'data.parquet'}],
    }]}).encode(),
}
manifest = {
    'schema_version': 1, 'repo_type': 'dataset', 'repo_id': 'example/data',
    'commit': commit,
    'files': {name: {
        'blob_sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload),
    } for name, payload in files.items()},
}
http_requests = []

def forbid_http(*args, **kwargs):
    http_requests.append(' '.join(str(arg) for arg in args[:2]))
    raise AssertionError('HTTP is forbidden')

class Transport:
    def get_object(self, **kwargs):
        if not kwargs['Key'].endswith(f'/datasets/example/data/snapshots/{commit}/manifest.json'):
            raise AssertionError('Only the pinned manifest read is expected')
        return {'Body': io.BytesIO(json.dumps(manifest).encode())}
    def download_file(self, bucket, key, destination, **kwargs):
        digest = key.rsplit('/', 1)[-1]
        payload = next(value for value in files.values()
                       if hashlib.sha256(value).hexdigest() == digest)
        Path(destination).write_bytes(payload)

with patch('boto3.client', return_value=Transport()), patch(
    'requests.Session.request', side_effect=forbid_http
), patch('httpx.Client.send', side_effect=forbid_http), patch(
    'aiohttp.ClientSession._request', side_effect=forbid_http
):
    install(bucket='audit-fixture-bucket', allow_hub_fallback=False)
    from datasets import load_dataset
    dataset = load_dataset('example/data', revision=commit)
    print(json.dumps({'splits': list(dataset), 'http_requests': http_requests}))
"""
    expected_output = {"splits": ["holdout"], "http_requests": []}

    # When
    result = _run_in_fresh_process(script, tmp_path)

    # Then
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


def test_install_with_fallback_keeps_the_hub_endpoint_in_fresh_process() -> None:
    # Given
    script = """
import os
from unittest.mock import patch
import hf_hub_s3_mirror
from hf_hub_s3_mirror import mirror
with patch('boto3.client', return_value=object()):
    hf_hub_s3_mirror.install(bucket='example-bucket')
print(mirror._HUB_FALLBACK, os.environ.get('HF_ENDPOINT'))
"""
    expected_output = "True None"
    environment = {
        key: value for key, value in os.environ.items() if key != "HF_ENDPOINT"
    }
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        HF_S3_MIRROR_ALLOW_HUB_FALLBACK="1",
    )

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output


def test_import_is_inert_until_explicit_install() -> None:
    # Given
    script = (
        "import sys; import hf_hub_s3_mirror; print('huggingface_hub' in sys.modules)"
    )
    expected_output = "False"
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output


def test_install_resolves_a_callable_bucket_on_first_access_in_fresh_process() -> None:
    # Given
    script = """
from unittest.mock import patch
import hf_hub_s3_mirror
from hf_hub_s3_mirror import mirror
calls = []
def bucket():
    calls.append(1)
    return 'example-bucket'
with patch('boto3.client', return_value=object()):
    hf_hub_s3_mirror.install(bucket=bucket, allow_hub_fallback=True)
before = len(calls)
print(before, mirror._BACKEND.bucket, mirror._BACKEND.bucket, len(calls))
"""
    expected_output = "0 example-bucket example-bucket 1"
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output


def test_install_rebinds_module_level_api_helpers_in_fresh_process() -> None:
    # Given
    script = """
from unittest.mock import patch
import hf_hub_s3_mirror
with patch('boto3.client', return_value=object()):
    hf_hub_s3_mirror.install(bucket='example-bucket')
import huggingface_hub
import huggingface_hub.hf_api
from hf_hub_s3_mirror.mirror import S3HfApi
helpers = ('model_info', 'list_repo_files')
print(huggingface_hub.HfApi is S3HfApi and all(
    isinstance(getattr(module, name).__self__, S3HfApi)
    for module in (huggingface_hub, huggingface_hub.hf_api)
    for name in helpers
))
"""
    expected_output = "True"
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output


def test_installed_filesystem_reads_verified_mirror_bytes_without_http(
    tmp_path: Path,
) -> None:
    # Given
    script = """
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
from hf_hub_s3_mirror import install

commit = 'a' * 40
payload = b'GOOD'
manifest = {
    'schema_version': 1, 'repo_type': 'model', 'repo_id': 'example/model',
    'commit': commit,
    'files': {'config.json': {
        'blob_sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload)
    }},
}
bucket_calls = []
def bucket():
    bucket_calls.append(1)
    return 'audit-fixture-bucket'

class Transport:
    downloads = 0
    def list_objects_v2(self, **kwargs):
        return {'Contents': [{'Key': 'audit-fixture'}]}
    def get_object(self, **kwargs):
        if kwargs['Key'].endswith('/manifest.json'):
            return {'Body': io.BytesIO(json.dumps(manifest).encode())}
        raise AssertionError('Unverified range reads must not be used')
    def download_file(self, bucket, key, destination, **kwargs):
        self.downloads += 1
        Path(destination).write_bytes(payload)

transport = Transport()
with patch('boto3.client', return_value=transport), patch(
    'requests.Session.request', side_effect=AssertionError('HTTP is forbidden')
), patch(
    'httpx.Client.send', side_effect=AssertionError('HTTP is forbidden')
):
    install(bucket=bucket, allow_hub_fallback=True)
    from huggingface_hub import HfFileSystem
    filesystem = HfFileSystem(token=False, skip_instance_cache=True)
    path = f'example/model@{commit}/config.json'
    content = filesystem.cat_file(path)
    suffix = filesystem.cat_file(path, start=-2)
    output = io.BytesIO()
    filesystem.get_file(path, output)
    print(json.dumps({
        'content': content.decode(), 'suffix': suffix.decode(),
        'download': output.getvalue().decode(),
        'downloads': transport.downloads, 'bucket_calls': len(bucket_calls),
    }))
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "HF_HUB_CACHE": str(tmp_path / "cache"),
        "HF_HOME": str(tmp_path / "hf-home"),
        "HF_S3_MIRROR_ENABLED": "0",
    }
    expected_output = {
        "content": "GOOD",
        "suffix": "OD",
        "download": "GOOD",
        "downloads": 1,
        "bucket_calls": 1,
    }

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


def test_installed_module_level_hffs_reads_from_the_mirror_without_http(
    tmp_path: Path,
) -> None:
    # Given
    script = """
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
from hf_hub_s3_mirror import install

commit = 'a' * 40
payload = b'GOOD'
manifest = {
    'schema_version': 1, 'repo_type': 'model', 'repo_id': 'example/model',
    'commit': commit,
    'files': {'config.json': {
        'blob_sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload)
    }},
}

class Transport:
    def list_objects_v2(self, **kwargs):
        return {'Contents': [{'Key': 'audit-fixture'}]}
    def get_object(self, **kwargs):
        if kwargs['Key'].endswith('/manifest.json'):
            return {'Body': io.BytesIO(json.dumps(manifest).encode())}
        raise AssertionError('Unverified range reads must not be used')
    def download_file(self, bucket, key, destination, **kwargs):
        Path(destination).write_bytes(payload)

with patch('boto3.client', return_value=Transport()), patch(
    'requests.Session.request', side_effect=AssertionError('HTTP is forbidden')
), patch(
    'httpx.Client.send', side_effect=AssertionError('HTTP is forbidden')
):
    install(bucket='audit-fixture-bucket', allow_hub_fallback=True)
    import huggingface_hub.hf_file_system as hf_file_system
    from huggingface_hub import HfFileSystem, hffs
    content = hffs.cat_file(f'example/model@{commit}/config.json')
    print(json.dumps({
        'content': content.decode(),
        'same_instance': hf_file_system.hffs is hffs,
        'patched_class': isinstance(hffs, HfFileSystem),
    }))
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "HF_HUB_CACHE": str(tmp_path / "cache"),
        "HF_HOME": str(tmp_path / "hf-home"),
    }
    expected_output = {"content": "GOOD", "same_instance": True, "patched_class": True}

    # When
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


def test_published_model_is_readable_as_a_snapshot_and_offline_file(
    fake_hub, tmp_path: Path
) -> None:
    # Given
    fake_hub.files.update(
        {
            "config.json": b'{"model_type":"bert"}',
            "weights/model.safetensors": b"published model weights",
        }
    )
    consumer = """
from huggingface_hub import hf_hub_download, snapshot_download

snapshot = Path(snapshot_download('example/model', revision='release/one'))
config = Path(hf_hub_download(
    'example/model', 'config.json', revision='release/one', local_files_only=True,
))
print(json.dumps({
    'files': sorted(str(path.relative_to(snapshot)) for path in snapshot.rglob('*') if path.is_file()),
    'config': json.loads(config.read_text()),
    'weights': (snapshot / 'weights/model.safetensors').read_text(),
    'network_calls': network_calls,
}))
"""
    expected_output = {
        "files": ["config.json", "weights/model.safetensors"],
        "config": {"model_type": "bert"},
        "weights": "published model weights",
        "network_calls": [],
    }

    # When
    receipt = fake_hub.run("model", "example/model@release%2Fone")
    result = _read_published_objects(fake_hub.s3.objects, tmp_path, consumer)

    # Then
    assert receipt["file_count"] == 2
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


@pytest.mark.parametrize("streaming", [False, True])
def test_published_dataset_loads_through_datasets(
    fake_hub, tmp_path: Path, streaming: bool
) -> None:
    # Given
    for index, text in enumerate(["hello", "mirror"]):
        shard = io.BytesIO()
        pq.write_table(pa.table({"text": [text]}), shard)
        fake_hub.files[f"data/train-{index}.parquet"] = shard.getvalue()
    fake_hub.files[".huggingface.yaml"] = b"""
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
"""
    consumer = f"""
from datasets import load_dataset

dataset = load_dataset(
    'example/data', revision='release/one', split='train', streaming={streaming!r},
)
print(json.dumps({{'rows': list(dataset), 'network_calls': network_calls}}))
"""
    expected_output = {
        "rows": [{"text": "hello"}, {"text": "mirror"}],
        "network_calls": [],
    }

    # When
    fake_hub.run("dataset", "example/data@release%2Fone")
    result = _read_published_objects(fake_hub.s3.objects, tmp_path, consumer)

    # Then
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


@pytest.mark.parametrize("fallback", [False, True])
def test_published_tokenizer_loads_named_templates_without_weights(
    fake_hub, tmp_path: Path, fallback: bool
) -> None:
    # Given
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    template = "{{ messages[0]['content'] }}"
    fake_hub.files.update(
        {
            "config.json": b'{"model_type":"bert","vocab_size":2}',
            "tokenizer_config.json": json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "unk_token": "[UNK]",
                }
            ).encode(),
            "tokenizer.json": tokenizer.to_str().encode(),
            "additional_chat_templates/custom.jinja": template.encode(),
            "model.safetensors": b"weights are not part of a tokenizer publication",
        }
    )
    consumer = """
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    'example/tokenizer', revision='release/one', trust_remote_code=False, token=False,
)
output = Path(os.environ['HF_HOME']) / 'saved-tokenizer'
tokenizer.save_pretrained(output)
print(json.dumps({
    'token_ids': tokenizer('hello')['input_ids'],
    'template': tokenizer.get_chat_template('custom'),
    'saved': (output / 'additional_chat_templates/custom.jinja').is_file(),
    'network_calls': network_calls,
}))
"""
    expected_output = {
        "token_ids": [1],
        "template": template,
        "saved": True,
        "network_calls": [],
    }

    # When
    receipt = fake_hub.run("tokenizer", "example/tokenizer@release%2Fone")
    result = _read_published_objects(
        fake_hub.s3.objects, tmp_path, consumer, fallback=fallback
    )

    # Then
    assert receipt["file_count"] == 4
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected_output


def test_tokenizer_save_pretrained_rejects_traversal_in_named_chat_template(
    tmp_path: Path,
) -> None:
    # Given
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.chat_template = {
        "default": "{{ messages }}",
        "../../outside": "attacker-controlled template",
    }
    output_directory = tmp_path / "tokenizer"
    output_directory.mkdir()
    outside_template = tmp_path / "outside.jinja"
    expected_error = "Invalid chat template name: '../../outside'"
    expected_error_pattern = re.escape(expected_error)

    # When
    with pytest.raises(ValueError, match=expected_error_pattern):
        tokenizer.save_pretrained(output_directory)

    # Then
    assert not outside_template.exists()
