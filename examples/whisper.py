import tempfile
import base64
from pydantic import BaseModel, Field
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

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


from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=torch.float16, use_safetensors=True
)
model.to("cuda")
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device="cuda",
)
#_ = pipeline("/app/warmup.wav")

def transcribe(args: TranscriptionArgs) -> str:
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
        return format_chunks(pipeline([tmpfile.name], return_timestamps=True, generate_kwargs=kwargs)[0]["chunks"])

print(transcribe(TranscriptionArgs(audio_b64=base64.b64encode(open("warmup.wav", "rb").read()))))
