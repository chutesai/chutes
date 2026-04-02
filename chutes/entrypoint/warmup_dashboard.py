"""
Textual TUI dashboard for chute warmup monitoring.

Shows real-time events, GPU requirements, and instance logs in a single view.
Uses ?quick=true warmup endpoint + Socket.IO events instead of long-lived SSE.
"""

import asyncio
from datetime import datetime
from typing import Optional

import aiohttp
import orjson as json
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static, RichLog
from textual.worker import Worker

from chutes.config import get_config
from chutes.entrypoint.warmup_events import EventsClient
from chutes.util.auth import sign_request


class HeaderWidget(Static):
    """Top bar showing chute name, status, GPU requirements, and bounty info."""

    def __init__(
        self, chute_name: str, node_selector: dict, initial_status: str = "COLD", **kwargs
    ):
        self.chute_name = chute_name
        self.node_selector = node_selector or {}
        self._status = initial_status
        self._bounty_info = None
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        return []

    def on_mount(self) -> None:
        self._render_header()

    def _render_header(self):
        gpu_count = self.node_selector.get("gpu_count", 1)
        include = self.node_selector.get("include", [])
        min_vram = self.node_selector.get("min_vram_gb_per_gpu")
        gpu_str = f"{gpu_count}x"
        if include:
            gpu_str += ",".join(include)
        elif min_vram:
            gpu_str += f" (>={min_vram}GB VRAM)"
        else:
            gpu_str += "GPU"

        if self._status == "HOT":
            status_badge = "[bold green] HOT [/]"
        else:
            status_badge = "[bold yellow] COLD [/]"

        bounty_str = ""
        if self._bounty_info:
            amount = self._bounty_info.get("bounty", self._bounty_info.get("amount"))
            boost = self._bounty_info.get("bounty_boost", self._bounty_info.get("boost"))
            age = self._bounty_info.get("age_seconds", 0)
            age_min = age // 60 if age else 0
            parts = []
            if amount:
                parts.append(f"bounty={amount}")
            if boost:
                parts.append(f"boost={boost:.2f}x")
            if age_min:
                parts.append(f"age={age_min}m")
            if parts:
                bounty_str = f"  [magenta]{'  '.join(parts)}[/]"

        self.update(f"[bold]{self.chute_name}[/]  {status_badge}  [dim]{gpu_str}[/]{bounty_str}")

    def set_status(self, status: str):
        self._status = status.upper()
        self._render_header()

    def set_bounty(self, bounty_info: dict):
        self._bounty_info = bounty_info
        self._render_header()


class GpuPanel(Static):
    """Shows GPU requirements and observed GPU types from events."""

    def __init__(self, node_selector: dict, **kwargs):
        self.node_selector = node_selector or {}
        self._observed_gpus: dict[str, dict] = {}
        self._instance_state: dict[str, dict] = {}
        super().__init__(**kwargs)

    def on_mount(self):
        self._refresh_content()

    def _refresh_content(self):
        gpu_count = self.node_selector.get("gpu_count", 1)
        include = self.node_selector.get("include", [])
        exclude = self.node_selector.get("exclude", [])
        min_vram = self.node_selector.get("min_vram_gb_per_gpu")

        # Left column: requirements
        parts = [f"[bold]GPUs:[/] {gpu_count}"]
        if include:
            parts.append(f"[green]{','.join(include)}[/]")
        if exclude:
            parts.append(f"[red]!{','.join(exclude)}[/]")
        if min_vram:
            parts.append(f"[cyan]>={min_vram}GB[/]")
        left = "  ".join(parts)

        # Right side: observed GPUs
        if self._observed_gpus:
            obs = []
            for gpu_type, info in sorted(self._observed_gpus.items()):
                active = info.get("active", 0)
                total = info.get("total", 0)
                obs.append(f"{gpu_type}: [green]{active}[/]/[dim]{total}[/]")
            right = "  |  [bold]Live:[/] " + "  ".join(obs)
        else:
            right = ""

        self.update(left + right)

    def track_instance(
        self, instance_id: str, gpu_type: str, gpu_count: int, active: bool, removed: bool = False
    ):
        """State-based GPU tracking per instance. GPUs are assigned at creation and never change."""
        if not gpu_type or not instance_id:
            return
        prev = self._instance_state.get(instance_id)
        if removed:
            if prev:
                entry = self._observed_gpus.get(prev["gpu_type"], {"active": 0, "total": 0})
                entry["total"] = max(0, entry["total"] - prev["gpu_count"])
                if prev["active"]:
                    entry["active"] = max(0, entry["active"] - prev["gpu_count"])
                del self._instance_state[instance_id]
        elif prev:
            # Already tracked — just update active flag.
            if active != prev["active"]:
                entry = self._observed_gpus.get(prev["gpu_type"], {"active": 0, "total": 0})
                if active:
                    entry["active"] += prev["gpu_count"]
                else:
                    entry["active"] = max(0, entry["active"] - prev["gpu_count"])
                prev["active"] = active
        else:
            # First time seeing this instance.
            if gpu_type not in self._observed_gpus:
                self._observed_gpus[gpu_type] = {"active": 0, "total": 0}
            entry = self._observed_gpus[gpu_type]
            entry["total"] += gpu_count
            if active:
                entry["active"] += gpu_count
            self._instance_state[instance_id] = {
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "active": active,
            }
        self._refresh_content()


class EventsPanel(RichLog):
    """Scrolling event log with color-coded entries."""

    def __init__(self, **kwargs):
        super().__init__(markup=True, **kwargs)

    def add_event(self, reason: str, message: str):
        now = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "instance_hot": "green",
            "instance_activated": "green",
            "instance_created": "yellow",
            "launch_token_created": "yellow",
            "instance_deleted": "red",
            "instance_disabled": "red",
            "instance_verified": "cyan",
            "bounty_change": "magenta",
            "warmup_hot": "green",
            "warmup_cold": "yellow",
            "info": "blue",
            "error": "red",
        }
        color = color_map.get(reason, "white")
        self.write(f"[dim]{now}[/] [{color}]{reason}[/] {message}")


class LogPanel(RichLog):
    """Per-instance log panel with title border."""

    MAX_LINES = 1000

    def __init__(self, instance_id: str, **kwargs):
        self.instance_id = instance_id
        short_id = instance_id[:8] if len(instance_id) > 8 else instance_id
        super().__init__(
            max_lines=self.MAX_LINES,
            wrap=True,
            id=f"log-{instance_id}",
            classes="log-panel",
            **kwargs,
        )
        self.border_title = f"Instance {short_id}"


class WarmupDashboard(App):
    """Textual TUI app for monitoring chute warmup."""

    CSS_PATH = "warmup_dashboard.tcss"
    BINDINGS = [("q", "quit", "Quit"), ("escape", "quit", "Quit")]

    # How often to re-trigger warmup (bounty refresh) while cold
    WARMUP_POLL_INTERVAL = 30.0

    def __init__(
        self,
        chute_name: str,
        chute_id: str,
        node_selector: dict,
        existing_instances: list,
        config=None,
    ):
        super().__init__()
        self.chute_name = chute_name
        self.chute_id = chute_id
        self.node_selector = node_selector or {}
        self.existing_instances = existing_instances or []
        self.config = config or get_config()
        self._log_panels: dict[str, LogPanel] = {}
        self._log_workers: dict[str, Worker] = {}
        self._events_client: Optional[EventsClient] = None
        self._is_hot = any(inst.get("active") for inst in self.existing_instances)

    def compose(self) -> ComposeResult:
        initial_status = "HOT" if self._is_hot else "COLD"
        yield HeaderWidget(
            self.chute_name, self.node_selector, initial_status=initial_status, id="header-widget"
        )
        with Container(id="main-area"):
            yield GpuPanel(self.node_selector, id="gpu-panel")
            yield EventsPanel(id="events-panel", max_lines=500)
            yield Container(id="log-grid")
        yield Static("Press [bold]q[/] to quit", id="status-bar")

    def on_mount(self) -> None:
        # Fire quick warmup (triggers bounty, gets initial status)
        self.run_worker(self._warmup_poll_worker(), exclusive=False, group="warmup")
        # Start events Socket.IO client
        self.run_worker(self._events_worker(), exclusive=False, group="events")
        # Add log panels for existing instances
        for inst in self.existing_instances:
            instance_id = inst.get("instance_id")
            if instance_id:
                self._add_log_panel(instance_id)
        self._update_status_bar()

    def _add_log_panel(self, instance_id: str):
        if instance_id in self._log_panels:
            return
        panel = LogPanel(instance_id)
        self._log_panels[instance_id] = panel
        grid = self.query_one("#log-grid")
        grid.mount(panel)
        self._reflow_grid()
        worker = self.run_worker(
            self._log_stream_worker(instance_id), exclusive=False, group="logs"
        )
        self._log_workers[instance_id] = worker
        self._update_status_bar()

    def _remove_log_panel(self, instance_id: str):
        if instance_id not in self._log_panels:
            return
        panel = self._log_panels.pop(instance_id)
        worker = self._log_workers.pop(instance_id, None)
        if worker:
            worker.cancel()
        panel.remove()
        self._reflow_grid()
        self._update_status_bar()

    def _reflow_grid(self):
        count = len(self._log_panels)
        grid = self.query_one("#log-grid")
        if count <= 1:
            cols = 1
        elif count <= 2:
            cols = 2
        elif count <= 4:
            cols = 2
        else:
            cols = 3
        grid.styles.grid_size_columns = cols

    def _update_status_bar(self):
        bar = self.query_one("#status-bar", Static)
        count = len(self._log_panels)
        hot_str = "[green]HOT[/]" if self._is_hot else "[yellow]COLD[/]"
        bar.update(
            f"Press [bold]q[/] to quit  |  "
            f"Status: {hot_str}  |  "
            f"Instances: [bold]{count}[/]  |  "
            f"Chute: [dim]{self.chute_id[:12]}...[/]"
        )

    async def _quick_warmup(self) -> Optional[dict]:
        """Call the quick warmup endpoint once. Returns response dict or None."""
        headers, _ = sign_request(purpose="chutes")
        try:
            async with aiohttp.ClientSession(base_url=self.config.generic.api_base_url) as session:
                async with session.get(
                    f"/chutes/warmup/{self.chute_name}",
                    headers=headers,
                    params={"quick": "true"},
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return None
        except Exception:
            return None

    async def _warmup_poll_worker(self):
        """Periodically call quick warmup to trigger bounty and check status."""
        header_widget = self.query_one(HeaderWidget)
        events_panel = self.query_one("#events-panel", EventsPanel)

        try:
            while True:
                result = await self._quick_warmup()
                if result:
                    chute_status = result.get("status", "cold")
                    instance_count = result.get("instance_count", 0)
                    bounty = result.get("bounty")

                    header_widget.set_status(chute_status)
                    if bounty:
                        header_widget.set_bounty(bounty)

                    if chute_status == "hot":
                        if not self._is_hot:
                            self._is_hot = True
                            events_panel.add_event(
                                "warmup_hot",
                                f"Chute is hot, {instance_count} instance(s) available",
                            )
                            self._update_status_bar()
                    else:
                        if self._is_hot:
                            self._is_hot = False
                            self._update_status_bar()
                        bounty_msg = ""
                        if bounty:
                            amount = bounty.get("amount", 0)
                            boost = bounty.get("boost", 1.0)
                            bounty_msg = f" (bounty={amount}, boost={boost:.2f}x)"
                        events_panel.add_event(
                            "warmup_cold",
                            f"Waiting for instances...{bounty_msg}",
                        )

                await asyncio.sleep(self.WARMUP_POLL_INTERVAL)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            events_panel.add_event("error", f"Warmup poll error: {exc}")

    async def _events_worker(self):
        """Connect to Socket.IO events server and dispatch events."""
        events_panel = self.query_one("#events-panel", EventsPanel)
        gpu_panel = self.query_one("#gpu-panel", GpuPanel)
        header_widget = self.query_one(HeaderWidget)

        async def on_event(data):
            reason = data.get("reason", "")
            message = data.get("message", "")
            event_data = data.get("data", {})

            events_panel.add_event(reason, message)

            gpu_type = event_data.get("gpu_model_name")
            gpu_count = event_data.get("gpu_count")
            instance_id = event_data.get("instance_id")

            if reason == "bounty_change":
                header_widget.set_bounty(event_data)
                return

            if reason == "instance_created" and instance_id:
                if gpu_type and gpu_count:
                    gpu_panel.track_instance(instance_id, gpu_type, gpu_count, active=False)
                self._add_log_panel(instance_id)

            elif reason == "instance_hot":
                # instance_hot = verified but not yet active, treat same as created
                if instance_id and gpu_type and gpu_count:
                    gpu_panel.track_instance(instance_id, gpu_type, gpu_count, active=False)

            elif reason == "instance_activated" and instance_id:
                if gpu_type and gpu_count:
                    gpu_panel.track_instance(instance_id, gpu_type, gpu_count, active=True)
                self._is_hot = True
                header_widget.set_status("HOT")
                self._update_status_bar()

            elif reason == "instance_deleted" and instance_id:
                if gpu_type and gpu_count:
                    gpu_panel.track_instance(
                        instance_id, gpu_type, gpu_count, active=False, removed=True
                    )
                self._remove_log_panel(instance_id)

            elif reason == "instance_disabled" and instance_id:
                if gpu_type and gpu_count:
                    gpu_panel.track_instance(instance_id, gpu_type, gpu_count, active=False)

        try:
            self._events_client = EventsClient(
                self.config.generic.api_base_url, self.chute_id, on_event
            )
            await self._events_client.connect()
            events_panel.add_event("info", "Connected to events server")
            await self._events_client.wait()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            events_panel.add_event(
                "error", f"Events connection failed: {exc} — events will not stream"
            )

    async def _log_stream_worker(self, instance_id: str):
        """Stream logs from a single instance."""
        panel = self._log_panels.get(instance_id)
        if not panel:
            return
        headers, _ = sign_request(purpose="logs")
        try:
            async with aiohttp.ClientSession(base_url=self.config.generic.api_base_url) as session:
                async with session.get(
                    f"/instances/{instance_id}/logs",
                    headers=headers,
                    params={"backfill": "100"},
                ) as response:
                    if response.status != 200:
                        panel.write(f"[red]Log stream failed: {response.status}[/]")
                        return
                    buffer = b""
                    async for chunk in response.content.iter_any():
                        if not chunk:
                            continue
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            if not line.strip():
                                continue
                            if line.startswith(b":"):
                                continue
                            if line.startswith(b"data: "):
                                data_content = line[6:].strip()
                                if not data_content:
                                    continue
                                try:
                                    data = json.loads(data_content)
                                    log_msg = data.get("log", "")
                                    if log_msg and log_msg.strip() and len(log_msg.strip()) > 1:
                                        panel.write(log_msg.rstrip())
                                except Exception:
                                    pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if instance_id in self._log_panels:
                panel.write(f"[red]Stream ended: {exc}[/]")

    async def action_quit(self) -> None:
        if self._events_client:
            try:
                await self._events_client.disconnect()
            except Exception:
                pass
        self.exit()
