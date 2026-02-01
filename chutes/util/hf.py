"""
HuggingFace cache verification utility.
"""

import os
import shutil
import asyncio
import aiohttp
import hashlib
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

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

    def __init__(
        self,
        message: str,
        reason: str = "verification_failed",
        repo_id: str | None = None,
        revision: str | None = None,
        mismatches: list[str] | None = None,
        missing: list[str] | None = None,
        extra: list[str] | None = None,
        errors: list[str] | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.repo_id = repo_id
        self.revision = revision
        self.mismatches = mismatches or []
        self.missing = missing or []
        self.extra = extra or []
        self.errors = errors or []

    def to_dict(self) -> dict:
        return {
            "error": True,
            "reason": self.reason,
            "message": str(self),
            "repo_id": self.repo_id,
            "revision": self.revision,
            "mismatches": self.mismatches,
            "missing": self.missing,
            "extra": self.extra,
            "errors": self.errors,
        }


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _get_symlink_hash(file_path: Path) -> str | None:
    """
    Extract hash from symlink target (blob filename).
    """
    if file_path.is_symlink():
        target = os.readlink(file_path)
        blob_name = Path(target).name
        # 64 chars = SHA256 (LFS), 40 chars = SHA1 (git blob)
        if len(blob_name) in (40, 64):
            return blob_name
    return None


def git_blob_hash(filepath: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """
    Compute git blob SHA-1 for a file using streaming (memory efficient).
    Git blob format: "blob {size}\0{content}"
    """
    size = filepath.stat().st_size
    sha1 = hashlib.sha1()
    sha1.update(f"blob {size}\0".encode())
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha1.update(chunk)
    return sha1.hexdigest()


def compute_sha256(filepath: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """
    Compute SHA256 hash of a file using streaming (memory efficient).
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


async def verify_cache(
    repo_id: str,
    revision: str,
    cache_dir: str = "/cache",
    full_hash_check: bool = False,
    max_workers: int = 4,
) -> dict:
    """
    Verify cached HuggingFace model files match checksums on the Hub.

    Args:
        repo_id: HuggingFace repository ID
        revision: Git revision (commit hash, branch, or tag)
        cache_dir: Cache directory path
        full_hash_check: If True, compute full file hashes instead of just
                        checking symlink names. Slower but more thorough.
        max_workers: Number of parallel workers for hash computation
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
                if resp.status == 404:
                    text = await resp.text()
                    raise CacheVerificationError(
                        f"Repository or revision not found: {repo_id}@{revision} - {text}",
                        reason="not_found",
                        repo_id=repo_id,
                        revision=revision,
                    )
                if resp.status in (401, 403):
                    text = await resp.text()
                    raise CacheVerificationError(
                        f"Access denied to {repo_id}: {text}",
                        reason="access_denied",
                        repo_id=repo_id,
                        revision=revision,
                    )
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning(
                        f"Cache verification skipped - proxy returned {resp.status}: {text}"
                    )
                    return {"verified": 0, "skipped": 0, "total": 0, "skipped_api_error": True}
                repo_info = await resp.json()
    except CacheVerificationError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"Cache verification skipped - proxy request failed: {e}")
        return {"verified": 0, "skipped": 0, "total": 0, "skipped_api_error": True}

    # Build remote files dict: {path: (hash, size, is_lfs)}
    remote_files = {}
    for item in repo_info["files"]:
        is_lfs = item.get("is_lfs", False)
        if is_lfs:
            remote_files[item["path"]] = (item.get("sha256"), item.get("size"), True)
        else:
            remote_files[item["path"]] = (item.get("blob_id"), item.get("size"), False)

    # Directories.
    directories = repo_info.get("directories")
    if directories is not None:
        for dir_path in directories:
            if dir_path not in remote_files:
                remote_files[dir_path] = (None, None, False)
    else:
        for item in repo_info["files"]:
            parts = item["path"].split("/")
            for i in range(1, len(parts)):
                dir_path = "/".join(parts[:i])
                if dir_path not in remote_files:
                    remote_files[dir_path] = (None, None, False)

    # Find local cache
    repo_folder_name = f"models--{repo_id.replace('/', '--')}"
    snapshot_dir = cache_dir / "hub" / repo_folder_name / "snapshots" / revision

    if not snapshot_dir.exists():
        raise CacheVerificationError(
            f"Cache directory not found: {snapshot_dir}",
            reason="cache_not_found",
            repo_id=repo_id,
            revision=revision,
        )

    # Get local files and directories
    local_files = {}
    for path in snapshot_dir.rglob("*"):
        if path.is_file() or path.is_symlink() or path.is_dir():
            rel_path = str(path.relative_to(snapshot_dir))
            local_files[rel_path] = path

    verified = 0
    skipped = 0
    mismatches = []
    missing = []
    errors = []

    # Files needing hash computation: (remote_path, resolved_path, expected_hash, hash_type)
    files_to_hash = []

    for remote_path, (remote_hash, remote_size, is_lfs) in remote_files.items():
        local_path = local_files.get(remote_path)

        if not local_path or (not local_path.exists() and not local_path.is_symlink()):
            missing.append(remote_path)
            continue

        if remote_hash is None:
            skipped += 1
            continue

        resolved_path = local_path.resolve()

        # Check size first (quick sanity check)
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

        if is_lfs:
            if full_hash_check:
                # Queue for full SHA256 computation
                files_to_hash.append((remote_path, resolved_path, remote_hash, "sha256"))
            else:
                # Fast check via symlink name
                symlink_hash = _get_symlink_hash(local_path)
                if symlink_hash:
                    if symlink_hash != remote_hash:
                        mismatches.append(
                            f"{remote_path}: hash {symlink_hash} != expected {remote_hash}"
                        )
                    else:
                        verified += 1
                else:
                    errors.append(f"{remote_path}: LFS file not a symlink, cannot fast-verify")
        else:
            # Non-LFS file: verify via git blob hash
            if full_hash_check:
                # Queue for git blob hash computation
                files_to_hash.append((remote_path, resolved_path, remote_hash, "git_blob"))
            else:
                # Fast check via symlink name (if available)
                symlink_hash = _get_symlink_hash(local_path)
                if symlink_hash:
                    if symlink_hash != remote_hash:
                        mismatches.append(
                            f"{remote_path}: hash {symlink_hash} != expected {remote_hash}"
                        )
                    else:
                        verified += 1
                else:
                    # Not a symlink, must compute hash
                    files_to_hash.append((remote_path, resolved_path, remote_hash, "git_blob"))

    # Compute hashes in parallel using thread pool
    if files_to_hash:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            def compute_hash_sync(item):
                remote_path, resolved_path, expected_hash, hash_type = item
                try:
                    if hash_type == "sha256":
                        computed = compute_sha256(resolved_path)
                    else:  # git_blob
                        computed = git_blob_hash(resolved_path)
                    return (remote_path, computed, expected_hash, None)
                except Exception as e:
                    return (remote_path, None, expected_hash, str(e))

            futures = [
                loop.run_in_executor(executor, compute_hash_sync, item) for item in files_to_hash
            ]
            results = await asyncio.gather(*futures)

            for remote_path, computed, expected, error in results:
                if error:
                    errors.append(f"{remote_path}: hash computation failed: {error}")
                elif computed != expected:
                    mismatches.append(f"{remote_path}: hash {computed} != expected {expected}")
                else:
                    verified += 1

    # Check for extra local files (ignore _ prefixed paths not in remote)
    extra = [
        p
        for p in local_files
        if p not in remote_files and not any(part.startswith("_") for part in Path(p).parts)
    ]

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
        raise CacheVerificationError(
            "\n".join(msg_parts),
            reason="integrity_mismatch",
            repo_id=repo_id,
            revision=revision,
            mismatches=mismatches,
            missing=missing,
            extra=extra,
            errors=errors,
        )

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
        "--cache-dir",
        default=str(Path(HF_HUB_CACHE).parent),
        help="Cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--full-hash-check",
        action="store_true",
        help="Compute full file hashes (slower but verifies actual content integrity)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers for hash computation (default: 4)",
    )
    args = parser.parse_args()

    try:
        result = asyncio.run(
            verify_cache(
                args.repo_id,
                args.revision,
                args.cache_dir,
                full_hash_check=args.full_hash_check,
                max_workers=args.max_workers,
            )
        )
        if result["skipped_api_error"]:
            print("⚠️  Verification skipped (API unavailable)")
        else:
            print(
                f"✅ Verified {result['verified']}/{result['total']} files "
                f"(skipped {result['skipped']} without hash)"
            )
    except CacheVerificationError as e:
        print(f"❌ {e}")
        sys.exit(1)
