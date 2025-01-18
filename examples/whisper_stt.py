import tempfile
import base64
from pydantic import BaseModel, Field
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="chutes",
        name="whisper-large-v3",
        tag="0.0.1",
        readme="## Speech to text with whisper-large-v3",
    )
    .from_base("parachutes/base-python:3.12.7")
    .set_user("root")
    .run_command("apt update && apt -y install ffmpeg")
    .set_user("chutes")
    .run_command("pip install transformers datasets[audio] accelerate")
    .add("warmup.wav", "/app/warmup.wav")
)

chute = Chute(
    username="chutes",
    name="whisper-large-v3",
    tagline="Speech to text with translation support using whisper-large-v3",
    readme="Whisper is a state-of-the-art model for automatic speech recognition (ASR) and speech translation, proposed in the paper Robust Speech Recognition via Large-Scale Weak Supervision by Alec Radford et al. from OpenAI. Trained on >5M hours of labeled data, Whisper demonstrates a strong ability to generalise to many datasets and domains in a zero-shot setting.",
    image=image,
    node_selector=NodeSelector(gpu_count=1),
)


class TranscriptionArgs(BaseModel):
    audio_b64: str
    language: str = Field(None, description="Also translate the text to this specified language.")


def format_chunks(chunks):
    return [
        {
            "start": chunk["timestamp"][0],
            "end": chunk["timestamp"][0],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]


@chute.on_startup()
async def initialize(self):
    """
    Load the model and all voice packs.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import torch

    self.torch = torch
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16, use_safetensors=True
    )
    model.to("cuda")
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    self.pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda",
    )
    _ = self.pipeline("/app/warmup.wav")


@chute.cord(
    public_api_path="/transcribe",
    public_api_method="POST",
    stream=False,
)
async def transcribe(self, args: TranscriptionArgs) -> str:
    """
    Transcribe (with automatic language detection).
    """
    with tempfile.NamedTemporaryFile(mode="wb") as tmpfile:
        tmpfile.write(base64.b64decode(args.audio_b64))
        tmpfile.flush()
        kwargs = {}
        if args.language:
            kwargs.update(
                {
                    "language": args.language,
                    "task": "translate",
                }
            )
        return format_chunks(
            self.pipeline([tmpfile.name], return_timestamps=True, generate_kwargs=kwargs)[0]["chunks"]
        )
