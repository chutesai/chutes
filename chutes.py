#!/usr/bin/env python

import typer
from chutes.entrypoint.api_key import create_api_key
from chutes.entrypoint.register import register
from chutes.entrypoint.report import report_invocation
COMMAND_MAP = {
    "build": "build_image",
    "deploy": "deploy_chute",
    "run": "run_chute",
    "report": "report_invocation",
    "register": "register",
    "api_key": "create_api_key",
}
# app.add_typer(api_key_app, name="api", help="bye")

app = typer.Typer(no_args_is_help=True)


app.command(name="register", help="Create an account with the chutes run platform!")(
    register
)
app.command(help="Create an API key for the chutes run platform!", no_args_is_help=True)(
    create_api_key
)
app.command(help="Report an invocation!", no_args_is_help=True)(report_invocation)


if __name__ == "__main__":
    app()
