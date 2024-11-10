#!/usr/bin/env python

import typer
from chutes.entrypoint.api_key import create_api_key
from chutes.entrypoint.deploy import deploy_chute
from chutes.entrypoint.register import register
from chutes.entrypoint.build import build_image
from chutes.entrypoint.report import report_invocation
from chutes.entrypoint.run import run_chute


app = typer.Typer(no_args_is_help=True)


app.command(name="register", help="Create an account with the chutes run platform!")(
    register
)
app.command(
    help="Create an API key for the chutes run platform!", no_args_is_help=True
)(create_api_key)
app.command(help="Report an invocation!", no_args_is_help=True)(report_invocation)
app.command(help="Run a chute!", no_args_is_help=True)(run_chute)
app.command(help="Deploy a chute!", no_args_is_help=True)(deploy_chute)
app.command(help="Build an image!", no_args_is_help=True)(build_image)
if __name__ == "__main__":
    app()
