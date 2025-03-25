import os
import uuid
import pybase64 as base64
import tempfile
from enum import Enum
from typing import Optional
from loguru import logger
from io import BytesIO
from fastapi import HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="chutes",
        name="spark-tts",
        tag="0.0.1",
        readme="## Text-to-speech using SparkAudio/Spark-TTS-0.5B",
    )
    .from_base("parachutes/base-python:3.12.9")
    .run_command("git clone https://github.com/SparkAudio/Spark-TTS.git")
    .run_command("pip install -r Spark-TTS/requirements.txt")
    .run_command("mv -f Spark-TTS/cli/SparkTTS.py /app/")
    .add("cortana.wav", "/app/warmup_audio.wav")
    .run_command("pip install pybase64")
    .with_env("PYTHONPATH", "/app/Spark-TTS")
)

chute = Chute(
    username="chutes",
    name="spark-tts",
    tagline="Text-to-speech with SparkAudio/Spark-TTS-0.5B, and optional transcription with whisper.",
    readme="Text-to-speech with SparkAudio/Spark-TTS-0.5B, and optional transcription with whisper.",
    image=image,
    node_selector=NodeSelector(gpu_count=1),
)


class SpeechOption(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# Spark-TTS supports two genders only so blame them if you don't like it.
class Gender(Enum):
    MALE = "male"
    FEMALE = "female"


class InputArgs(BaseModel):
    text: str
    sample_audio_b64: Optional[str] = None
    sample_audio_text: Optional[str] = None
    pitch: Optional[SpeechOption] = None
    speed: Optional[SpeechOption] = None
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    gender: Optional[Gender] = None


def transcribe(self, audio_path):
    """
    Use whisper to transcribe input audio.
    """
    chunks = self.whisper_pipeline([audio_path], return_timestamps=True)[0]["chunks"]
    return " ".join([chunk["text"] for chunk in chunks])


def load_audio(self, audio_b64):
    """
    Convert the input base64 to wav and make sure torchaudio can load it.
    """
    try:
        audio_bytes = BytesIO(base64.b64decode(audio_b64))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes.getvalue())
            temp_path = temp_file.name
        waveform, sample_rate = self.torchaudio.load(temp_path)
        return temp_path
    except Exception as exc:
        logger.error(f"Error loading audio: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input audio_b64 provided: {exc}",
        )


@chute.on_startup()
async def initialize(self):
    """
    Initialize the model.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from huggingface_hub import snapshot_download
    from SparkTTS import SparkTTS
    import torchaudio
    import torch
    import soundfile

    # Make sure the model is downloaded and do a warmup pass.
    revision = "642071559bfc6346c2359d19dcb6be3f9dd8a05d"
    path = snapshot_download("SparkAudio/Spark-TTS-0.5B", revision=revision)
    self.model = SparkTTS(path, "cuda")

    # Load whisper to automatically transcribe audio samples if sample text is not provided.
    self.whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16, use_safetensors=True
    )
    self.whisper.to("cuda")
    self.whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    self.whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=self.whisper,
        tokenizer=self.whisper_processor.tokenizer,
        feature_extractor=self.whisper_processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda",
    )

    # Warmup.
    with torch.no_grad():
        text = transcribe(self, "/app/warmup_audio.wav")
        _ = self.model.inference(
            "Warming up Spark-TTS.",
            "/app/warmup_audio.wav",
            prompt_text=text,
            gender="female",
            pitch=SpeechOption.MODERATE.value,
            speed=SpeechOption.MODERATE.value,
        )

    self.torchaudio = torchaudio
    self.torch = torch
    self.soundfile = soundfile


@chute.cord(
    public_api_path="/speak",
    public_api_method="POST",
    stream=False,
    output_content_type="audio/wav",
)
async def speak(self, args: InputArgs) -> Response:
    """
    Perform text to speech, with optional voice cloning.
    """
    # Extract input sample + transcribe if text wasn't provided.
    input_path = None
    if args.sample_audio_b64:
        input_path = load_audio(self, args.sample_audio_b64)
        if not args.sample_audio_text:
            try:
                args.sample_audio_text = transcribe(self, input_path)
            except:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid input audio_b64 provided: {exc}",
                )
    else:
        args.sample_audio_text = None

    # Generate the audio.
    output_path = f"/tmp/{uuid.uuid4()}.wav"
    try:
        generate_args = args.model_dump()
        generate_args.pop("sample_audio_b64", None)
        generate_args.pop("sample_audio_text", None)
        for key in ("pitch", "speed", "gender"):
            value = generate_args.get(key)
            if key in generate_args and not value:
                generate_args.pop(key)
            if isinstance(value, Enum):
                generate_args[key] = value.value
        if input_path:
            generate_args.update(
                {"prompt_speech_path": input_path, "prompt_text": args.sample_audio_text}
            )
            # These seem to prevent cloning.
            for key in ("pitch", "pseed", "gender"):
                generate_args.pop(key, None)
        text = generate_args.pop("text")
        with self.torch.no_grad():
            output = self.model.inference(text, **generate_args)
            self.soundfile.write(output_path, output, samplerate=16000)
        with open(output_path, "rb") as infile:
            return Response(
                content=infile.read(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename={uuid.uuid4()}.wav",
                },
            )
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
