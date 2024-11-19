#!/usr/bin/env python

import typer
from chutes.entrypoint.api_key import create_api_key
from chutes.entrypoint.deploy import deploy_chute
from chutes.entrypoint.register import register
from chutes.entrypoint.build import build_image
from chutes.entrypoint.report import report_invocation
from chutes.entrypoint.run import run_chute
from chutes.crud import chutes_app, images_app, api_keys_app

app = typer.Typer(no_args_is_help=True)


app.command(name="register", help="Create an account with the chutes run platform!")(register)

app.command(help="Report an invocation!", no_args_is_help=True, name="report")(report_invocation)
app.command(help="Run a chute!", no_args_is_help=True, name="run")(run_chute)
app.command(help="Deploy a chute!", no_args_is_help=True, name="deploy")(deploy_chute)
app.command(help="Build an image!", no_args_is_help=True, name="build")(build_image)

# Chutes
app.add_typer(chutes_app, name="chutes")

# Images
app.add_typer(images_app, name="images")

# API Keys
api_keys_app.command(
    help="Create an API key for the chutes run platform!",
    no_args_is_help=True,
    name="create",
)(create_api_key)
app.add_typer(api_keys_app)

if __name__ == "__main__":
    app()
