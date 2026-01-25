"""
HuggingFace cache verification utility.
"""

import os
import shutil
import asyncio
import aiohttp
from pathlib import Path
from loguru import logger

PROXY_URL = "https://api.chutes.ai/misc/hf_repo_info"


def purge_model_cache(repo_id: str, cache_dir: str = "/cache") -> bool:
    """
    Recursively delete the cache directory for a specific model.
    Returns True if anything was deleted, False otherwise.
    """
    cache_dir = Path(cache_dir)
    repo_folder_name = f"models--{repo_id.replace('/', '--')}"
    model_cache_path = cache_dir / "hub" / repo_folder_name

    if model_cache_path.exists():
        logger.warning(f"Purging corrupted cache at {model_cache_path}")
        shutil.rmtree(model_cache_path, ignore_errors=True)
        return True
    return False


class CacheVerificationError(Exception):
    """Raised when cache verification fails."""

    pass


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _get_symlink_hash(file_path: Path) -> str | None:
    """
    Extract SHA256 from symlink target (blob filename).
    """
    if file_path.is_symlink():
        target = os.readlink(file_path)
        blob_name = Path(target).name
        if len(blob_name) == 64:
            return blob_name
    return None


async def verify_cache(
    repo_id: str,
    revision: str,
    cache_dir: str = "/cache",
) -> dict:
    """
    Verify cached HuggingFace model files match checksums on the Hub.
    """
    cache_dir = Path(cache_dir)
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
    snapshot_dir = cache_dir / "hub" / repo_folder_name / "snapshots" / revision

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

    logger.success(f"Successfully verified HF cache for {repo_id=} {revision=}")

    return {
        "verified": verified,
        "skipped": skipped,
        "total": len(remote_files),
        "skipped_api_error": False,
    }


if __name__ == "__main__":
    import argparse
    import sys
    from huggingface_hub.constants import HF_HUB_CACHE

    parser = argparse.ArgumentParser(description="Verify HuggingFace cache integrity")
    parser.add_argument(
        "--repo-id", required=True, help="Repository ID (e.g. deepseek-ai/DeepSeek-V3.2)"
    )
    parser.add_argument(
        "--revision", required=True, help="Git revision (commit hash, branch, or tag)"
    )
    parser.add_argument(
        "--cache-dir", default=HF_HUB_CACHE, help="Cache directory (default: HF_HUB_CACHE)"
    )
    args = parser.parse_args()

    try:
        result = asyncio.run(verify_cache(args.repo_id, args.revision, args.cache_dir))
        if result["skipped_api_error"]:
            print("⚠️  Verification skipped (API unavailable)")
        else:
            print(
                f"✅ Verified {result['verified']}/{result['total']} files (skipped {result['skipped']} non-LFS)"
            )
    except CacheVerificationError as e:
        print(f"❌ {e}")
        sys.exit(1)
