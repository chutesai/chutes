# User CLI log migration

Branch `vm-log-read-cli`.

## What changed

- `chutes/entrypoint/log_read.py` (added). `read_instance_logs` GETs `/logs/instances/{instance_id}` with the headers it is given and no query params. A 200 body is parsed with `orjson` and passed to `render_log_lines`, which prints each non-empty string `log` in array order and skips a missing, null, or non-string `log`. `ts` and `stream` are not printed. A non-list body or invalid JSON returns empty text and no error. 401 returns `Not authorized to read logs.` 403 and 404 return the JSON `detail`. Any other status returns `HTTP {status}: {body}`. `prefix_instance_logs` prefixes each line with the first 8 characters of the instance id. `SessionClient` adapts an aiohttp session to that call.
- `chutes/entrypoint/warmup_dashboard.py`. `_log_stream_worker` no longer opens `GET /instances/{instance_id}/logs?backfill=100`, no longer parses SSE (`data:` lines, `:` comments, single-character keepalives), and no longer retries up to 30 times. It signs with `purpose="logs"` and calls `read_instance_logs` once. An error is written to the panel in red. Success writes each rendered line. The unused `orjson` import left with the SSE parser; parsing now lives in `log_read.py`.
- `chutes/entrypoint/warmup.py`. `plain_stream_all_instances` (`chutes warmup --stream-logs`) dropped the same SSE parser and `backfill=100`. Each newly discovered instance id is fetched once, then prefixed with `prefix_instance_logs`. `stream_instance_logs` dropped the `backfill` argument and the SSE parser; it calls `read_instance_logs` once and writes stdout, or raises on `result.error`. `poll_and_stream_logs` still calls `stream_instance_logs`, so it follows that new path. Nothing in `warmup()` calls either function.
- `pyproject.toml` (added). Only `[tool.pytest.ini_options]`. Pytest's default `norecursedirs` plus `"tdd"`. No `[project]` and no `[build-system]`; `setup.py` still installs the package.
- `tdd/tests/test_log_read_cli.py` (added). Hand-written `HttpSpy`. No `conftest.py`.

## What was not changed

- `startup_logs.py` still lists `GET /encrypted_logs/{chute_id}/sessions` and downloads chunks with `purpose="chutes"`. That path is owner-only ciphertext; the Loki route is plaintext and allows sharees, support, and subnet admins.
- `_decrypt_logs.py` still decrypts those chunks locally. The new route returns plaintext, so this module is not on the warmup path.
- `chutes/util/auth.py` is unchanged. Call sites still use `sign_request(purpose="logs")`. The tests pass headers in and do not generate signatures.
- Instance discovery in `plain_stream_all_instances` still polls `GET /chutes/warmup/{name}?quick=true` and falls back to `GET /chutes/{name}`, both with `purpose="chutes"`.
- `monitor_warmup` still reads the warmup status SSE on `GET /chutes/warmup/{name}`. That stream is chute status, not pod logs.
- CLI flags `--stream-logs` and `--dashboard` are unchanged. Default warmup is still the dashboard.

## Test coverage

Tests live in `tdd/tests/test_log_read_cli.py`. They build their own spy, import nothing from `tests/`, and `tdd` is in `norecursedirs`, so a bare `pytest` does not collect them.

- `test_should_render_log_text_when_response_has_one_line` — one object renders `booting\n`.
- `test_should_preserve_order_when_response_has_multiple_lines` — two objects render `alpha` then `beta`.
- `test_should_skip_line_when_log_field_is_missing_or_null` — a missing or null `log` is omitted; a null `ts` or `stream` does not drop a present `log`.
- `test_should_render_empty_text_when_response_has_no_lines` — `[]` renders `""`.
- `test_should_report_auth_error_when_response_is_401` — error is `Not authorized to read logs.` and text is empty.
- `test_should_include_server_detail_when_response_is_403` — the error contains `You may only view logs for your own (private) chutes.`
- `test_should_include_server_detail_when_response_is_404` — the error contains `Instance not found.`
- `test_should_report_http_status_when_response_is_unexpected` — HTTP 500 with body `boom` returns `HTTP 500: boom`.
- `test_should_request_instance_logs_path_with_given_headers_and_no_query` — the spy sees `GET /logs/instances/inst-123`, the given headers, and `params is None`.
- `test_should_return_rendered_lines_when_response_is_200` — a 200 array returns `booting\n` and `error is None`.
- `test_should_return_empty_text_when_body_is_not_a_list` — a 200 JSON object returns empty text and no error.
- `test_should_return_empty_text_when_body_is_not_json` — a 200 body of `not-json` returns empty text and no error.
- `test_should_prefix_rendered_text_with_instance_id` — `abcdefghijklmnop` prefixes each line with `[abcdefgh] `.

## Depends on

`chutes-api` branch `vm-log-read-routes` must merge first. `GET /logs/instances/{instance_id}` must be deployed before this CLI change is released.
