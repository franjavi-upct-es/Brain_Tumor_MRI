# reporting.py - Lightweight utilities for reproducible, traceable reports
# -------------------------------------------------------------------------
# Centralizes helpers to fingerprint datasets/configs and capture git state
# so every metric file can include the context that produced it.

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def safe_git_commit(repo_root: Path = None) -> str:
    """
    Return the current git commit hash if available.
    Falls back to 'unknown' when the repo is not a git checkout.
    """
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def file_fingerprint(path: Path) -> Tuple[str, int]:
    """
    Build a lightweight fingerprint of a dataset folder without hashing image bytes.
    Uses relative paths + file sizes + mtimes to detect drift and returns:
        ("<count>:<12-char sha1>", count)
    """
    path = Path(path)
    hasher = hashlib.sha1()
    count = 0

    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        if file.suffix.lower() not in IMAGE_EXTS:
            continue
        count += 1
        rel = file.relative_to(path)
        stat = file.stat()
        hasher.update(str(rel).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(int(stat.st_mtime)).encode("utf-8"))

    digest = hasher.hexdigest()[:12]
    return f"{count}:{digest}", count


def config_hash(cfg_path: str) -> str:
    """Return a short hash of the config file contents for traceability."""
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        return "unknown"
    data = cfg_file.read_bytes()
    return hashlib.sha1(data).hexdigest()[:12]


def build_metadata(
    cfg_path: str, dataset_dir: str, extra: Dict = None, repo_root: Path = None
) -> Dict:
    """
    Compose a metadata dictionary that can be attached to any report JSON.
    Includes git commit, config hash, dataset fingerprint and optional extras.
    """
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    dataset_fp, dataset_count = file_fingerprint(Path(dataset_dir))
    meta = {
        "git_commit": safe_git_commit(repo_root),
        "config_path": cfg_path,
        "config_hash": config_hash(cfg_path),
        "dataset": {
            "path": str(dataset_dir),
            "fingerprint": dataset_fp,
            "num_images": dataset_count,
        },
    }
    if extra:
        meta.update(extra)
    return meta
