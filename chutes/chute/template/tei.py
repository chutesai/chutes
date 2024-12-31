import asyncio
import socket
import time
from loguru import logger
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Callable
from chutes.image import Image
from chutes.image.standard.tei import TEI
from chutes.chute import Chute, ChutePack, NodeSelector


class EmbeddingRequest(BaseModel):
    inputs: Union[str, List[str]] = Field(
        ..., description="String or list of strings to generate embeddings for"
    )


class EmbeddingData(BaseModel):
    embedding: List[float] = Field(..., description="Vector embedding")
    index: int = Field(..., description="Index of the input text")


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData] = Field(..., description="List of embeddings")


class RerankRequest(BaseModel):
    query: str = Field(..., description="Query text to compare against candidate texts")
    texts: List[str] = Field(..., description="List of texts to be ranked against the query")
    raw_scores: Optional[bool] = Field(False, description="Whether to return raw similarity scores")


class RankResult(BaseModel):
    index: int = Field(..., description="Original index of the text")
    text: str = Field(..., description="Original text")
    relevance_score: float = Field(..., description="Similarity score with the query")


class RerankResponse(BaseModel):
    results: List[RankResult] = Field(..., description="Ranked results sorted by relevance score")


class PredictRequest(BaseModel):
    inputs: Union[List[str], List[List[str]]] = Field(
        ..., description="Single text or batch of texts for classification"
    )


class PredictionResult(BaseModel):
    label: str = Field(..., description="Predicted class label")
    score: float = Field(..., description="Confidence score for the prediction")


class PredictResponse(BaseModel):
    predictions: List[List[PredictionResult]] = Field(
        ..., description="Predictions for each input text"
    )


class TEIChute(ChutePack):
    embed: Optional[Callable] = None
    rerank: Optional[Callable] = None
    predict: Optional[Callable] = None


def build_tei_chute(
    username: str,
    model_name: str,
    endpoints: list[str],
    node_selector: NodeSelector,
    image: str | Image = TEI,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 32,
    revision: Optional[str] = None,
):
    chute = Chute(
        username=username,
        name=model_name,
        tagline=tagline,
        readme=readme,
        image=image,
        node_selector=node_selector,
        concurrency=concurrency,
        standard_template="tei",
    )

    async def _check_tcp_port():
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", 8881)
            writer.close()
            await writer.wait_closed()
            return True
        except (ConnectionRefusedError, socket.gaierror):
            return False

    @chute.on_startup()
    async def start_tei(self):
        nonlocal model_name, revision
        cmd = [
            "text-embeddings-router",
            "--model-id",
            model_name,
        ]
        if revision:
            cmd += ["--revision", revision]
        cmd += ["--port", "8881"]
        self.process = await asyncio.create_subprocess_exec(*cmd, stdout=None, stderr=None)
        started_at = time.time()
        self.running = False
        while (delta := time.time() - started_at) <= 600:
            if await _check_tcp_port():
                logger.success(f"TEI server started successfully after {int(delta)} seconds")
                self.running = True
                break
            logger.info(f"TEI server still not running after {int(delta)} seconds...")
            await asyncio.sleep(5)
        if not self.running:
            raise RuntimeError(f"TEI server failed to start after {int(delta)} seconds!")

    cords = {}
    if "embed" in endpoints:

        @chute.cord(
            public_api_path="/embed",
            method="POST",
            input_schema=EmbeddingRequest,
            passthrough=True,
            passthrough_path="/embed",
            passthrough_port=8881,
        )
        async def embed(data) -> EmbeddingData:
            return data

        cords["embed"] = embed

    if "rerank" in endpoints:

        @chute.cord(
            public_api_path="/rerank",
            method="POST",
            input_schema=RerankRequest,
            passthrough=True,
            passthrough_path="/rerank",
            passthrough_port=8881,
        )
        async def rerank(data) -> RerankResponse:
            return data

        cords["rerank"] = rerank

    if "predict" in endpoints:

        @chute.cord(
            public_api_path="/predict",
            method="POST",
            input_schema=PredictRequest,
            passthrough=True,
            passthrough_path="/predict",
            passthrough_port=8881,
        )
        async def predict(data) -> PredictResponse:
            return data

        cords["predict"] = predict

    return TEIChute(chute=chute, **cords)
