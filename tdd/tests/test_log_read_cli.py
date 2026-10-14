import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chutes.entrypoint.log_read import prefix_instance_logs, read_instance_logs, render_log_lines


class _Response:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def text(self):
        return self._body

    async def json(self):
        import json

        return json.loads(self._body)


class HttpSpy:
    def __init__(self, status, body):
        self._status = status
        self._body = body
        self.calls = []

    async def get(self, path, headers=None, params=None):
        self.calls.append({"path": path, "headers": headers, "params": params})
        return _Response(self._status, self._body)


def test_should_render_log_text_when_response_has_one_line():
    lines = [{"ts": "2026-07-27T00:00:00.123456789Z", "stream": "stdout", "log": "booting"}]

    rendered = render_log_lines(lines)

    assert rendered == "booting\n"


def test_should_preserve_order_when_response_has_multiple_lines():
    lines = [
        {"ts": "2026-07-27T00:00:00.000000001Z", "stream": "stdout", "log": "alpha"},
        {"ts": "2026-07-27T00:00:00.000000002Z", "stream": "stderr", "log": "beta"},
    ]

    rendered = render_log_lines(lines)

    assert rendered == "alpha\nbeta\n"


def test_should_skip_line_when_log_field_is_missing_or_null():
    lines = [
        {"ts": "2026-07-27T00:00:00.000000001Z", "stream": "stdout"},
        {"ts": "2026-07-27T00:00:00.000000002Z", "stream": "stdout", "log": None},
        {"stream": None, "log": "kept"},
        {"ts": None, "log": "also"},
    ]

    rendered = render_log_lines(lines)

    assert rendered == "kept\nalso\n"


def test_should_render_empty_text_when_response_has_no_lines():
    rendered = render_log_lines([])

    assert rendered == ""


def test_should_report_auth_error_when_response_is_401():
    client = HttpSpy(status=401, body='{"detail": "bad signature"}')

    result = asyncio.run(read_instance_logs(client, "instance-1", {"X-Chutes-Hotkey": "hk"}))

    assert result.error == "Not authorized to read logs."
    assert result.text == ""


def test_should_include_server_detail_when_response_is_403():
    detail = "You may only view logs for your own (private) chutes."
    client = HttpSpy(status=403, body=f'{{"detail": "{detail}"}}')

    result = asyncio.run(read_instance_logs(client, "instance-1", {"X-Chutes-Hotkey": "hk"}))

    assert detail in result.error
    assert result.text == ""


def test_should_include_server_detail_when_response_is_404():
    detail = "Instance not found."
    client = HttpSpy(status=404, body=f'{{"detail": "{detail}"}}')

    result = asyncio.run(read_instance_logs(client, "missing", {"X-Chutes-Hotkey": "hk"}))

    assert detail in result.error
    assert result.text == ""


def test_should_report_http_status_when_response_is_unexpected():
    client = HttpSpy(status=500, body="boom")

    result = asyncio.run(read_instance_logs(client, "inst-123", {"X-Chutes-Hotkey": "hk"}))

    assert result.text == ""
    assert result.error == "HTTP 500: boom"


def test_should_request_instance_logs_path_with_given_headers_and_no_query():
    headers = {"X-Chutes-Hotkey": "hk", "X-Chutes-Signature": "sig"}
    client = HttpSpy(status=200, body="[]")

    asyncio.run(read_instance_logs(client, "inst-123", headers))

    assert client.calls == [
        {"path": "/logs/instances/inst-123", "headers": headers, "params": None}
    ]


def test_should_return_rendered_lines_when_response_is_200():
    client = HttpSpy(
        status=200,
        body='[{"ts": "2026-07-27T00:00:00.123456789Z", "stream": "stdout", "log": "booting"}]',
    )

    result = asyncio.run(read_instance_logs(client, "inst-123", {"X-Chutes-Hotkey": "hk"}))

    assert result.error is None
    assert result.text == "booting\n"


def test_should_return_empty_text_when_body_is_not_a_list():
    client = HttpSpy(status=200, body='{"detail": "nope"}')

    result = asyncio.run(read_instance_logs(client, "inst-123", {}))

    assert result.error is None
    assert result.text == ""


def test_should_return_empty_text_when_body_is_not_json():
    client = HttpSpy(status=200, body="not-json")

    result = asyncio.run(read_instance_logs(client, "inst-123", {}))

    assert result.error is None
    assert result.text == ""


def test_should_prefix_rendered_text_with_instance_id():
    rendered = prefix_instance_logs("abcdefghijklmnop", "hello\nworld\n")

    assert rendered == "[abcdefgh] hello\n[abcdefgh] world\n"
