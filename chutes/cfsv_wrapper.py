import os
import ctypes
import asyncio
from functools import lru_cache
from fastapi import Request


class CFSVWrapper:
    def __init__(
        self, lib_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "chutes-cfsv.so")
    ):
        self.lib = ctypes.CDLL(lib_path)

        # cfsv_challenge(base_path, salt, sparse, index_file, exclude_path, result_buf, result_buf_size)
        self.lib.cfsv_challenge.argtypes = [
            ctypes.c_char_p,  # base_path
            ctypes.c_char_p,  # salt
            ctypes.c_int,     # sparse
            ctypes.c_char_p,  # index_file
            ctypes.c_char_p,  # exclude_path
            ctypes.c_char_p,  # result_buf
            ctypes.c_size_t,  # result_buf_size
        ]
        self.lib.cfsv_challenge.restype = ctypes.c_int

        # cfsv_sizetest(test_dir, size_gib)
        self.lib.cfsv_sizetest.argtypes = [
            ctypes.c_char_p,  # test_dir
            ctypes.c_size_t,  # size_gib
        ]
        self.lib.cfsv_sizetest.restype = ctypes.c_int

        # cfsv_version()
        self.lib.cfsv_version.argtypes = []
        self.lib.cfsv_version.restype = ctypes.c_char_p

    def challenge(
        self,
        salt,
        mode="full",
        base_path="/",
        index_file="/etc/chutesfs.index",
        exclude_path="/app/chute.py",
    ):
        """
        Compute filesystem challenge hash.

        Args:
            salt: Challenge salt from validator
            mode: "sparse" or "full" (default: "full")
            base_path: Root path for file scanning (default: "/")
            index_file: Path to encrypted index file (default: "/etc/chutesfs.index")
            exclude_path: Path to exclude from hashing (default: "/app/chute.py")

        Returns:
            Hex string hash on success, None on failure
        """
        sparse = 1 if mode == "sparse" else 0
        result_buf = ctypes.create_string_buffer(65)
        ret = self.lib.cfsv_challenge(
            base_path.encode() if isinstance(base_path, str) else base_path,
            salt.encode() if isinstance(salt, str) else salt,
            sparse,
            index_file.encode() if isinstance(index_file, str) else index_file,
            exclude_path.encode() if isinstance(exclude_path, str) else exclude_path,
            result_buf,
            65,
        )
        if ret == 0:
            return result_buf.value.decode("utf-8")
        return None

    def sizetest(self, test_dir, size_gib):
        """
        Test filesystem capacity and integrity.

        Args:
            test_dir: Directory to test in
            size_gib: Size in GiB to test

        Returns:
            True on success, False on failure
        """
        ret = self.lib.cfsv_sizetest(
            test_dir.encode() if isinstance(test_dir, str) else test_dir,
            size_gib,
        )
        return ret == 0

    def version(self):
        """
        Get library version.

        Returns:
            Version string
        """
        result = self.lib.cfsv_version()
        if result:
            return result.decode("utf-8")
        return None


@lru_cache(maxsize=1)
def get_cfsv():
    """Lazily initialize CFSV wrapper (only works on Linux)."""
    return CFSVWrapper()


async def handle_challenge(request: Request):
    loop = asyncio.get_event_loop()
    salt = request.state.decrypted["salt"]
    mode = request.state.decrypted.get("mode", "full")
    exclude_path = request.state.decrypted.get("exclude_path", "/app/chute.py")
    result = await loop.run_in_executor(
        None,
        get_cfsv().challenge,
        salt,
        mode,
        "/",
        "/etc/chutesfs.index",
        exclude_path,
    )
    return {"result": result}


async def handle_sizetest(request: Request):
    loop = asyncio.get_event_loop()
    test_dir = request.state.decrypted.get("test_dir", "/tmp")
    size_gib = request.state.decrypted.get("size_gib", 10)
    result = await loop.run_in_executor(
        None,
        get_cfsv().sizetest,
        test_dir,
        size_gib,
    )
    return {"result": result}


async def handle_version(request: Request):
    return {"result": get_cfsv().version()}
