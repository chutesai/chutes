import asyncio
import typer
from loguru import logger
from chutes.entrypoint._shared import upload_logo


def upload_logo_cmd(
    logo: str = typer.Argument(
        ...,
        help="Path to the logo image file to upload",
    ),
):
    """
    Upload a logo and return the logo ID.
    """

    async def _upload():
        logo_id = await upload_logo(logo)
        logger.success(f"Logo uploaded successfully: logo_id={logo_id}")

    asyncio.run(_upload())
