from __future__ import annotations
import os
import typing as t

CANON = b"authorization"
XAPI  = b"x-api-key"
DEPRECATION_HEADERS = [
    (b"deprecation", b"true"),
    (b"sunset", b"2026-01-31"),
    (b'link', b'<https://github.com/chutesai/chutes#authentication; rel="deprecation"'),
]

def _extract_token(raw_headers: list[tuple[bytes, bytes]]) -> tuple[str | None, bool]:
    h: dict[bytes, bytes] = {}
    for k, v in raw_headers:
        h[k.lower()] = v
    alias_used = False
    v = h.get(CANON)
    if v:
        s = v.decode("latin1").strip()
        ls = s.lower()
        if ls.startswith("bearer "):
            return s[7:].strip(), False
        if ls.startswith("basic "):
            return s[6:].strip(), True
        return s, True
    v = h.get(XAPI)
    if v:
        return v.decode("latin1").strip(), True
    return None, False

class AuthNormalizationMiddleware:
    def __init__(self, app, enabled_paths_prefix: t.Tuple[str, ...] = ("/v1/",)):
        self.app = app
        self.enabled_prefixes = enabled_paths_prefix

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path") or ""
        if not any(path.startswith(p) for p in self.enabled_prefixes):
            return await self.app(scope, receive, send)
        # Global toggle
        if str(os.getenv("CHUTES_CANONICALIZE_OPENAI_AUTH", "1")).lower() not in {"1","true","yes","on"}:
            return await self.app(scope, receive, send)

        raw: list[tuple[bytes, bytes]] = list(scope.get("headers") or [])
        token, alias_used = _extract_token(raw)
        if token:
            token_bytes = f"Bearer {token}".encode("latin1")
            new_headers: list[tuple[bytes, bytes]] = []
            saw_auth = False
            for k, v in raw:
                kl = k.lower()
                if kl == CANON:
                    if not saw_auth:
                        new_headers.append((CANON, token_bytes))
                        saw_auth = True
                    continue
                if kl == XAPI:
                    continue
                new_headers.append((k, v))
            if not saw_auth:
                new_headers.append((CANON, token_bytes))
            scope = dict(scope)
            scope["headers"] = new_headers

        async def send_wrapper(message):
            if alias_used and message.get("type") == "http.response.start":
                m = dict(message)
                hdrs = list(m.get("headers") or [])
                hdrs.extend(DEPRECATION_HEADERS)
                m["headers"] = hdrs
                return await send(m)
            return await send(message)

        return await self.app(scope, receive, send_wrapper)

__all__ = ["AuthNormalizationMiddleware"]
