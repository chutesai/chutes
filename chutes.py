#!/usr/bin/env python

import typer
from chutes.entrypoint.api_key import app as api_key_app

COMMAND_MAP = {
    "build": "build_image",
    "deploy": "deploy_chute",
    "run": "run_chute",
    "report": "report_invocation",
    "register": "register",
    "api_key": "create_api_key",
}

app = typer.Typer()
app.add_typer(api_key_app, name="api_key")

if __name__ == "__main__":
    app()
