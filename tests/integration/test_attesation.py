import pytest
from chutes.entrypoint.verify import TeeGpuVerifier

@pytest.mark.asyncio
async def test_get_nonce():

    verifier = TeeGpuVerifier({}, "https://api.chutes.dev", {}, {})
    nonce = await verifier._get_nonce()
    assert len(nonce) == 64