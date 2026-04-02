"""
Socket.IO client for streaming chute events from the events websocket server.
"""

from typing import Callable, Optional
from urllib.parse import urlparse

import socketio
from loguru import logger


class EventsClient:
    """
    Connects to the Socket.IO events server and filters events by chute_id.
    """

    def __init__(self, api_base_url: str, chute_id: str, callback: Callable):
        self.chute_id = chute_id
        self.callback = callback
        self.events_url = self._derive_events_url(api_base_url)
        self.sio: Optional[socketio.AsyncClient] = None
        self._connected = False

    @staticmethod
    def _derive_events_url(api_base_url: str) -> str:
        parsed = urlparse(api_base_url)
        host = parsed.hostname or ""
        # api.chutes.ai -> events.chutes.ai
        if host.startswith("api."):
            host = "events." + host[4:]
        else:
            host = "events." + host
        scheme = parsed.scheme or "https"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{scheme}://{host}{port}"

    async def connect(self):
        self.sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=0,  # infinite
            reconnection_delay=1,
            reconnection_delay_max=30,
        )

        @self.sio.on("events")
        async def on_events(data):
            try:
                event_data = data.get("data", {}) if isinstance(data, dict) else {}
                if event_data.get("chute_id") == self.chute_id:
                    await self.callback(data)
            except Exception as exc:
                logger.debug(f"Error processing event: {exc}")

        @self.sio.event
        async def connect():
            self._connected = True
            logger.debug(f"Connected to events server at {self.events_url}")

        @self.sio.event
        async def disconnect():
            self._connected = False
            logger.debug("Disconnected from events server")

        try:
            await self.sio.connect(self.events_url, transports=["websocket"])
        except Exception as exc:
            logger.debug(f"Failed to connect to events server: {exc}")
            raise

    async def wait(self):
        if self.sio:
            await self.sio.wait()

    async def disconnect(self):
        if self.sio and self._connected:
            await self.sio.disconnect()
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected
