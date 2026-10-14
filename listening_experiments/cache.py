"""Persistent RT channel cache and read-only import of existing SectorDrop archives."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

from .config import RTConfig, positive_int
from .storage import (ROOT, SCHEMA_VERSION, atomic_json, atomic_npz, canonical_json,
                      code_signature, file_hash, fingerprint, read_npz)


@dataclass
class ChannelSnapshot:
    macro_id: int
    h_tn_dl: np.ndarray
    h_ntn_dl: np.ndarray
    h_ntn_ul: np.ndarray
    array_positions_dl: np.ndarray
    orientations: np.ndarray
    f_dl: float
    f_ul: float
    metadata: dict = field(default_factory=dict)
    source_id: str = ""

    def validate(self):
        for name in ("h_tn_dl", "h_ntn_dl", "h_ntn_ul"):
            h = np.asarray(getattr(self, name))
            if h.ndim != 4 or min(h.shape) < 1 or not np.all(np.isfinite(h)):
                raise ValueError(f"{name} must be finite, nonempty [user, RX antenna, TX, TX antenna]")
        if self.h_ntn_dl.shape != self.h_ntn_ul.shape:
            raise ValueError("UL and DL NTN channel shapes must agree")
        if self.h_ntn_ul.shape[1] != 1:
            raise ValueError("This experiment models one UL waveform per single-antenna NTN terminal")
        ntx, m = self.h_ntn_dl.shape[2:]
        if self.h_tn_dl.shape[2:] != (ntx, m):
            raise ValueError("TN and NTN BS array dimensions differ")
        if self.array_positions_dl.shape != (m, 3) or self.orientations.shape != (ntx, 3):
            raise ValueError("Exact wavelength-normalized DL array positions and orientations are required")
        if not np.all(np.isfinite(self.array_positions_dl)) or not np.all(np.isfinite(self.orientations)):
            raise ValueError("Array geometry must be finite")
        if not np.isfinite(self.f_dl + self.f_ul) or min(self.f_dl, self.f_ul) <= 0:
            raise ValueError("Invalid UL/DL frequencies")
        if not self.source_id:
            raise ValueError("Channel snapshot must have a persistent source identity")
        return self


def _load_snapshot(path, expected_key):
    a = read_npz(path)
    meta = json.loads(str(a.pop("metadata_json")))
    if meta.get("cache_key") != expected_key or meta.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"Cache metadata mismatch: {path}")
    snapshot = ChannelSnapshot(
        macro_id=int(a["macro_id"]), h_tn_dl=a["h_tn_dl"], h_ntn_dl=a["h_ntn_dl"],
        h_ntn_ul=a["h_ntn_ul"], array_positions_dl=a["array_positions_dl"],
        orientations=a["orientations"], f_dl=float(a["f_dl"]), f_ul=float(a["f_ul"]),
        metadata=meta, source_id=file_hash(path))
    return snapshot.validate()


def rt_identity(config, project_root=ROOT):
    """Asset content + relevant code/config; activity, timing, and macro count are excluded."""
    scene = (Path(project_root) / config.scene_path).resolve()
    if not scene.is_file():
        raise FileNotFoundError(scene)
    assets = [scene] + sorted(p for p in scene.parent.rglob("*")
                             if p.is_file() and p != scene and p.suffix.lower() in
                             {".xml", ".ply", ".obj", ".serialized", ".png", ".jpg", ".exr"})
    return dict(schema=SCHEMA_VERSION, config=config,
                assets={str(p.relative_to(scene.parent)): file_hash(p) for p in assets},
                code=code_signature(["SceneConfigSionnaSectorDrop.py", "tn_sector_drop.py",
                                     "multipath_support.py", "satellite_projection.py",
                                     "vsat_dish_3gpp.py", "nulling_cdf_utils.py",
                                     "sionnautils/miutils.py", "ntn_music_detection.py",
                                     "listening_experiments/config.py", "listening_experiments/storage.py",
                                     "listening_experiments/rt_backend.py"]))


def prepare_channels(config: RTConfig, num_macros, cache_root, *, policy="auto", project_root=ROOT):
    """Reuse each completed macro. 'reuse_only' never imports or calls Sionna."""
    config.validate()
    positive_int(num_macros, "num_macros")
    if policy not in {"auto", "reuse_only"}:
        raise ValueError("policy must be 'auto' or 'reuse_only'")
    identity = rt_identity(config, project_root)
    key = fingerprint(identity)
    directory = Path(cache_root) / key[:20]
    missing = [m for m in range(num_macros) if not (directory / f"macro_{m:04d}.npz").is_file()]
    if missing and policy == "reuse_only":
        raise FileNotFoundError(f"Missing RT macros {missing} in {directory}; use policy='auto' to generate them")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "rt_config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(canonical_json(identity)):
        raise ValueError("RT cache identity collision")
    atomic_json(manifest, identity)
    backend = None
    channels = []
    for macro_id in range(num_macros):
        path = directory / f"macro_{macro_id:04d}.npz"
        if not path.exists():
            if backend is None:
                from .rt_backend import RTBackend
                backend = RTBackend(config, project_root, directory)
            print(f"RT macro {macro_id + 1}/{num_macros}: tracing UL/DL", flush=True)
            arrays, metadata = backend.trace(macro_id)
            metadata.update(cache_key=key, schema=SCHEMA_VERSION)
            atomic_npz(path, **arrays, metadata_json=np.array(canonical_json(metadata)))
        else:
            print(f"RT macro {macro_id + 1}/{num_macros}: cache hit", flush=True)
        snapshot = _load_snapshot(path, key)
        if snapshot.macro_id != macro_id:
            raise ValueError(f"Wrong macro index in {path}")
        channels.append(snapshot)
    print(f"Channel cache: {directory}", flush=True)
    return channels


def load_cached_channels(cache_dir, *, num_macros, expected_config=None):
    """Explicitly reuse a frozen native archive, including its original code provenance.

    Current Python code hashes are intentionally not used to select a different
    RT run. The original manifest/NPZ identities and requested physical config
    must agree. No tracing, writes or silent fallback are performed.
    """
    positive_int(num_macros, "num_macros")
    base = Path(cache_dir)
    manifest = json.loads((base / "rt_config.json").read_text())
    if manifest.get("schema") != SCHEMA_VERSION:
        raise ValueError("Unsupported native channel archive schema")
    if expected_config is not None:
        expected_config.validate()
        if canonical_json(manifest["config"]) != canonical_json(expected_config):
            raise ValueError("Requested RT parameters differ from the frozen channel archive; select a matching cache or generate new RT")
    key = fingerprint(manifest)
    result = []
    for macro in range(num_macros):
        path = base / f"macro_{macro:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen macro {macro}: {path}; this mode never runs RT")
        snapshot = _load_snapshot(path, key)
        if snapshot.macro_id != macro:
            raise ValueError(f"Wrong macro index in {path}")
        physical = manifest["config"]
        if snapshot.f_dl != physical["f_dl"] or snapshot.f_ul != physical["f_ul"]:
            raise ValueError(f"Frequency mismatch between manifest and channels: {path}")
        if snapshot.h_ntn_dl.shape[0] != physical["ntn_users"] or snapshot.h_ntn_dl.shape[-1] != physical["tx_rows"] * physical["tx_cols"]:
            raise ValueError(f"Array/user dimensions differ from the RT manifest: {path}")
        result.append(snapshot)
    print(f"Reusing {len(result)} frozen RT macros from {base}; no ray tracing", flush=True)
    return result


def load_legacy_channels(channels_dir, *, ul_subdir, num_macros, f_dl, f_ul):
    """Read old RT archives without mutation or automatic UL-band guessing."""
    positive_int(num_macros, "num_macros")
    base = Path(channels_dir)
    result = []
    for macro in range(num_macros):
        dl_path = base / f"channels_{macro:04d}.npz"
        ul_path = base / ul_subdir / f"sensing_{macro:04d}.npz"
        dl, ul = read_npz(dl_path), read_npz(ul_path)
        if not np.isclose(float(ul["ul_frequency_hz"]), f_ul, rtol=1e-10) or not np.isclose(float(ul["dl_frequency_hz"]), f_dl, rtol=1e-10):
            raise ValueError(f"Requested UL/DL frequencies do not match {ul_path}")
        geometry = dl["array_positions_local"]
        if not np.allclose(ul["array_positions_local_ul"], geometry * f_ul / f_dl):
            raise ValueError("Legacy cache does not have a consistent fixed physical array")
        snapshot = ChannelSnapshot(
            macro, dl["h_tn"], dl["h_ntn"], ul["h_ntn_ul"], geometry,
            dl["tx_orientations_rad"], float(f_dl), float(f_ul),
            metadata=dict(origin="legacy_sector_drop", dl_archive=str(dl_path.resolve()),
                          ul_archive=str(ul_path.resolve()), normalization="raw channel coefficients; old sensing noise ignored"),
            source_id=fingerprint([file_hash(dl_path), file_hash(ul_path)]))
        result.append(snapshot.validate())
    print(f"Reusing {len(result)} legacy RT macros from {base}; no ray tracing", flush=True)
    return result
