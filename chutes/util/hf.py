"""
HuggingFace cache verification utility.
"""

import os
import asyncio
import aiohttp
from pathlib import Path
from huggingface_hub.constants import HF_HUB_CACHE
from loguru import logger

PROXY_URL = "https://api.chutes.ai/misc/hf_repo_info"


class CacheVerificationError(Exception):
    """Raised when cache verification fails."""

    pass


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _get_symlink_hash(file_path: Path) -> str | None:
    """Extract SHA256 from symlink target (blob filename)."""
    if file_path.is_symlink():
        target = os.readlink(file_path)
        blob_name = Path(target).name
        if len(blob_name) == 64:
            return blob_name
    return None


async def verify_cache(
    repo_id: str,
    revision: str,
    cache_dir: str | None = None,
) -> dict:
    """
    Verify cached HuggingFace model files match checksums on the Hub.

    Uses fast mode: checks symlink targets + file sizes (no full hash computation).

    Args:
        repo_id: Repository ID (e.g. "deepseek-ai/DeepSeek-V3.2")
        revision: Git revision (commit hash, branch, or tag)
        cache_dir: Cache directory (default: HF_HUB_CACHE)

    Returns:
        dict with verification stats: {verified, skipped, total, skipped_api_error}

    Raises:
        CacheVerificationError: If verification fails (mismatches, missing, or extra files)
    """
    cache_dir = Path(cache_dir or HF_HUB_CACHE)

    # Fetch file metadata from proxy
    params = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
    }
    hf_token = _get_hf_token()
    if hf_token:
        params["hf_token"] = hf_token

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                PROXY_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning(
                        f"Cache verification skipped - proxy returned {resp.status}: {text}"
                    )
                    return {"verified": 0, "skipped": 0, "total": 0, "skipped_api_error": True}
                repo_info = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"Cache verification skipped - proxy request failed: {e}")
        return {"verified": 0, "skipped": 0, "total": 0, "skipped_api_error": True}

    # Build remote files dict: {path: (sha256, size)}
    remote_files = {}
    for item in repo_info["files"]:
        if item["path"].startswith("_"):
            continue
        if item.get("is_lfs"):
            remote_files[item["path"]] = (item.get("sha256"), item.get("size"))
        else:
            remote_files[item["path"]] = (item.get("blob_id"), item.get("size"))

    # Find local cache
    repo_folder_name = f"models--{repo_id.replace('/', '--')}"
    snapshot_dir = cache_dir / repo_folder_name / "snapshots" / revision

    if not snapshot_dir.exists():
        raise CacheVerificationError(f"Cache directory not found: {snapshot_dir}")

    # Get local files (ignore _ prefixed)
    local_files = {}
    for path in snapshot_dir.rglob("*"):
        if path.is_file() or path.is_symlink():
            rel_path = str(path.relative_to(snapshot_dir))
            if not any(part.startswith("_") for part in Path(rel_path).parts):
                local_files[rel_path] = path

    verified = 0
    skipped = 0
    mismatches = []
    missing = []
    errors = []

    for remote_path, (remote_hash, remote_size) in remote_files.items():
        local_path = local_files.get(remote_path)

        if not local_path or (not local_path.exists() and not local_path.is_symlink()):
            missing.append(remote_path)
            continue

        # Skip non-LFS files (sha1 blob id = 40 chars)
        if remote_hash is None or len(remote_hash) == 40:
            skipped += 1
            continue

        resolved_path = local_path.resolve()

        # Check size
        if remote_size is not None:
            try:
                actual_size = resolved_path.stat().st_size
                if actual_size != remote_size:
                    mismatches.append(
                        f"{remote_path}: size {actual_size} != expected {remote_size}"
                    )
                    continue
            except OSError as e:
                errors.append(f"{remote_path}: cannot stat: {e}")
                continue

        # Check symlink hash
        symlink_hash = _get_symlink_hash(local_path)
        if symlink_hash:
            if symlink_hash != remote_hash:
                mismatches.append(f"{remote_path}: hash {symlink_hash} != expected {remote_hash}")
                continue
        else:
            # Not a symlink - can't fast-verify, treat as error
            errors.append(f"{remote_path}: not a symlink, cannot fast-verify")
            continue

        verified += 1

    # Check for extra local files
    extra = [p for p in local_files if p not in remote_files]

    # Build error message if needed
    if mismatches or missing or extra or errors:
        msg_parts = [f"Cache verification failed for {repo_id}@{revision}"]
        if mismatches:
            msg_parts.append(f"Mismatches ({len(mismatches)}): " + "; ".join(mismatches))
        if missing:
            msg_parts.append(f"Missing ({len(missing)}): " + ", ".join(missing))
        if extra:
            msg_parts.append(f"Extra ({len(extra)}): " + ", ".join(extra))
        if errors:
            msg_parts.append(f"Errors ({len(errors)}): " + "; ".join(errors))
        raise CacheVerificationError("\n".join(msg_parts))

    return {
        "verified": verified,
        "skipped": skipped,
        "total": len(remote_files),
        "skipped_api_error": False,
    }
