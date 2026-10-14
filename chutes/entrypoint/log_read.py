import orjson as json


def log_read_failure(status, body):
    if status == 401:
        return "Not authorized to read logs."
    if status == 403 or status == 404:
        return json.loads(body)["detail"]
    return f"HTTP {status}: {body}"


class LogRead:
    def __init__(self, text, error):
        self.text = text
        self.error = error


def prefix_instance_logs(instance_id, rendered):
    short_id = instance_id[:8]
    return "".join(f"[{short_id}] {line}\n" for line in rendered.splitlines())


def instance_logs_path(instance_id):
    return f"/logs/instances/{instance_id}"


class SessionClient:
    """Adapt an aiohttp session to the client `read_instance_logs` calls."""

    def __init__(self, session):
        self._session = session

    async def get(self, path, headers=None, params=None):
        kwargs = {"headers": headers}
        if params is not None:
            kwargs["params"] = params
        async with self._session.get(path, **kwargs) as response:
            body = await response.text()
            return _BodyResponse(response.status, body)


class _BodyResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def text(self):
        return self._body


async def read_instance_logs(client, instance_id, headers):
    response = await client.get(instance_logs_path(instance_id), headers=headers)
    body = await response.text()
    if response.status == 200:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return LogRead("", None)
        return LogRead(render_log_lines(payload), None)
    return LogRead("", log_read_failure(response.status, body))


def render_log_lines(lines):
    if not isinstance(lines, list):
        return ""
    rendered = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        log = line.get("log")
        if isinstance(log, str) and log:
            rendered.append(f"{log}\n")
    return "".join(rendered)
