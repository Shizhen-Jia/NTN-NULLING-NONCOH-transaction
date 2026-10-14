"""Small atomic, pickle-free storage helpers for caches and experiment checkpoints."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value):
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    _atomic_write(path, lambda stream: stream.write((canonical_json(value) + "\n").encode()))


def atomic_npz(path, **arrays):
    if any(np.asarray(v).dtype.hasobject for v in arrays.values()):
        raise ValueError("Object arrays are forbidden in experiment archives")
    _atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def code_signature(names):
    return {name: file_hash(ROOT / name) for name in names}
