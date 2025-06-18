import os
import json
import ctypes
import base64
import asyncio
from fastapi import Request


class EnvDump:
    def __init__(
        self, lib_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "envdump.so")
    ):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.free.argtypes = [ctypes.c_void_p]
        self.lib.free.restype = None
        self.lib.dump.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.dump.restype = ctypes.c_void_p
        self.lib.decrypt.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        self.lib.decrypt.restype = ctypes.c_void_p
        self.lib.signature.argtypes = [ctypes.c_char_p]
        self.lib.signature.restype = ctypes.c_void_p
        self.lib.slurp.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self.lib.slurp.restype = ctypes.c_void_p
        self.lib.toca.argtypes = [ctypes.c_char_p]
        self.lib.toca.restype = None

    def dump(self, key):
        key_bytes = bytes.fromhex(key)
        result_ptr = self.lib.dump(key_bytes, len(key_bytes))
        if result_ptr:
            try:
                result = ctypes.string_at(result_ptr).decode("utf-8")
                return result
            finally:
                self.lib.free(result_ptr)
        return None

    def decrypt(self, key, encrypted_b64):
        key_bytes = bytes.fromhex(key)
        enc_bytes = encrypted_b64.encode() if isinstance(encrypted_b64, str) else encrypted_b64
        result_ptr = self.lib.decrypt(key_bytes, len(key_bytes), enc_bytes)
        if result_ptr:
            try:
                result = ctypes.string_at(result_ptr).decode("utf-8")
                return json.loads(result)
            finally:
                self.lib.free(result_ptr)
        return None

    def slurp(self, key, path, offset, length):
        key_bytes = bytes.fromhex(key)
        path = path.encode() if isinstance(path, str) else path
        result_ptr = self.lib.slurp(key_bytes, len(key_bytes), path, offset, length)
        if result_ptr:
            try:
                result = ctypes.string_at(result_ptr).decode("utf-8")
                return result
            finally:
                self.lib.free(result_ptr)
        return None

    def toca(self, path):
        self.lib.toca(path.encode() if isinstance(path, str) else path)

    def sig(self, salt):
        salt64 = base64.b64encode(salt.encode() if isinstance(salt, str) else salt)
        result_ptr = self.lib.signature(salt64)
        if result_ptr:
            try:
                result = ctypes.string_at(result_ptr).decode("utf-8")
                return result
            finally:
                self.lib.free(result_ptr)
        return None


DUMPER = EnvDump()


async def handle_dump(request: Request):
    loop = asyncio.get_event_loop()
    return {
        "result": await loop.run_in_executor(
            None,
            DUMPER.dump,
            request.state.decrypted["key"],
        ),
    }


async def handle_slurp(request: Request):
    loop = asyncio.get_event_loop()
    key = request.state.decrypted["key"]
    path = request.state.decrypted["path"]
    offset = request.state.decrypted.get("offset", 0)
    length = request.state.decrypted.get("length", 0)
    return {
        "result": await loop.run_in_executor(
            None,
            DUMPER.slurp,
            key,
            path,
            offset,
            length,
        )
    }


async def handle_toca(request: Request):
    loop = asyncio.get_event_loop()
    path = request.state.decrypted["path"]
    await loop.run_in_executor(None, DUMPER.toca, path)
    return {"ok": True}


async def handle_sig(request: Request):
    loop = asyncio.get_event_loop()
    return {
        "result": await loop.run_in_executor(
            None,
            DUMPER.sig,
            request.state.decrypted["salt"],
        ),
    }
