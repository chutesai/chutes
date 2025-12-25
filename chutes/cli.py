#!/usr/bin/env python
"""
Chutes CLI - Command-line interface for the Chutes platform.

This module provides the main CLI entry point for managing chutes, images,
API keys, secrets, and other platform resources.
"""

import typer
from loguru import logger
from chutes.entrypoint.api_key import create_api_key
from chutes.entrypoint.deploy import deploy_chute
from chutes.entrypoint.register import register
from chutes.entrypoint.login import login
from chutes.entrypoint.build import build_image
from chutes.entrypoint.report import report_invocation
from chutes.entrypoint.run import run_chute
from chutes.entrypoint.fingerprint import change_fingerprint
from chutes.entrypoint.share import share_chute
from chutes.entrypoint.warmup import warmup_chute
from chutes.entrypoint.secret import create_secret
from chutes.crud import chutes_app, images_app, api_keys_app, secrets_app
from chutes.exception import ChutesError, UserAbortedError

app = typer.Typer(no_args_is_help=True)


def handle_errors(func):
    """
    Decorator to handle common exceptions in CLI commands.
    
    Converts ChutesError exceptions into clean user-facing messages
    and proper exit codes.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except UserAbortedError as e:
            logger.info(str(e))
            raise typer.Exit(0)  # Clean exit, user cancelled
        except ChutesError as e:
            logger.error(str(e))
            raise typer.Exit(1)
        except KeyboardInterrupt:
            logger.info("Operation interrupted by user")
            raise typer.Exit(130)  # Standard SIGINT exit code
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise typer.Exit(1)
    return wrapper


# Register commands with error handling
app.command(name="register", help="Create an account with the chutes run platform!")(
    handle_errors(register)
)
app.command(name="login", help="Login to the chutes platform via browser-based authentication.")(
    handle_errors(login)
)
app.command(help="Change your fingerprint!", no_args_is_help=True, name="refinger")(
    handle_errors(change_fingerprint)
)
app.command(help="Report an invocation!", no_args_is_help=True, name="report")(
    handle_errors(report_invocation)
)
app.command(help="Run a chute!", no_args_is_help=True, name="run")(
    handle_errors(run_chute)
)
app.command(help="Deploy a chute!", no_args_is_help=True, name="deploy")(
    handle_errors(deploy_chute)
)
app.command(help="Build an image!", no_args_is_help=True, name="build")(
    handle_errors(build_image)
)
app.command(help="Share a chute!", no_args_is_help=True, name="share")(
    handle_errors(share_chute)
)
app.command(help="Warm up a chute!", no_args_is_help=True, name="warmup")(
    handle_errors(warmup_chute)
)

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

# Secrets
secrets_app.command(
    help="Create a secret!",
    no_args_is_help=True,
    name="create",
)(create_secret)
app.add_typer(secrets_app)

if __name__ == "__main__":
    app()
