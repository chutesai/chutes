# ruff: noqa
import chutes
import chutes.chute
import chutes.chute.template
import chutes.chute.template.helpers
import chutes.chute.template.diffusion
import chutes.chute.template.vllm
import chutes.chute.template.embedding
import chutes.chute.template.sglang
import chutes.chute.base
import chutes.chute.job
import chutes.chute.node_selector
import chutes.chute.cord
import chutes.exception
import chutes._version
import chutes.entrypoint.__init__
import chutes.entrypoint._shared
import chutes.entrypoint.logger
import chutes.entrypoint.ssh
import chutes.entrypoint.build
import chutes.entrypoint.fingerprint
import chutes.entrypoint.share
import chutes.entrypoint.report
import chutes.entrypoint.api_key
import chutes.entrypoint.warmup
import chutes.entrypoint.secret
import chutes.entrypoint.deploy
import chutes.entrypoint.run
import chutes.entrypoint.register
import chutes.util
import chutes.util.context
import chutes.util.schema
import chutes.util.auth
import chutes.util.user
import chutes.config
import chutes.envcheck.core
import chutes.envdump
import chutes.constants
import chutes.metrics
import chutes.cli
import chutes.crud
import chutes.cfsv_wrapper
import chutes.image
import chutes.image.directive
import chutes.image.directive.maintainer
import chutes.image.directive.workdir
import chutes.image.directive.add
import chutes.image.directive.env
import chutes.image.directive.apt
import chutes.image.directive.base_image
import chutes.image.directive.entrypoint
import chutes.image.directive.generic_run
import chutes.image.directive.user
import chutes.image
import chutes.image.standard
import chutes.image.standard.diffusion
import chutes.image.standard.vllm
import chutes.image.standard.sglang
import os
import ctypes
import asyncio

CLIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chutes-inspecto.so"))
CLIB.get_hash.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
CLIB.get_hash.restype = ctypes.c_char_p


async def generate_hash(hash_type: str = "base", challenge: str = None):
    if not challenge or not isinstance(challenge, (str, bytes)):
        challenge = None
    if isinstance(challenge, str):
        challenge = challenge.encode()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, CLIB.get_hash, hash_type.encode(), challenge)
    if result:
        return result.decode()
    return result

if __name__ == "__main__":
    print(asyncio.run(generate_hash()))
