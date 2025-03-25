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
from pydantic import BaseModel
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="chutes",
        name="orpheus-tts",
        tag="0.0.3",
        readme="## Text-to-speech using canopyai/Orpheus-TTS",
    )
    .from_base("parachutes/base-python:3.12.9")
    .run_command("pip install orpheus-speech vllm==0.7.3")
    .add("cortana.wav", "/app/warmup_audio.wav")
    .run_command("pip install pybase64 huggingface-hub")
    .run_command("perl -pi -e 's/engine_args = AsyncEngineArgs\\(/engine_args = AsyncEngineArgs\\(\\n            max_model_len=100000,/' /home/chutes/.local/lib/python3.12/site-packages/orpheus_tts/engine_class.py")
    .run_command("perl -pi -e 's/print\\(prompt/#/g' /home/chutes/.local/lib/python3.12/site-packages/orpheus_tts/engine_class.py")
)

chute = Chute(
    username="chutes",
    name="orpheus-tts",
    tagline="Text-to-speech with canopylabs/orpheus-3b-0.1-ft and optional transcription with whisper.",
    readme="Text-to-speech with canopylabs/orpheus-3b-0.1-ft and optional transcription with whisper.",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24),
)


class Voice(Enum):
    Tara = "tara"
    Leah = "leah"
    Jess = "jess"
    Leo = "leo"
    Dan = "dan"
    Mia = "mia"
    Zac = "zac"
    Zoe = "zoe"


class InputArgs(BaseModel):
    prompt: str
    # sample_audio_b64: Optional[str] = None
    # sample_audio_text: Optional[str] = None
    voice: Optional[Voice] = Voice.Tara


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
    from orpheus_tts import OrpheusModel
    import torchaudio
    import torch
    import wave
    from huggingface_hub import snapshot_download

    # Make sure the model is downloaded and do a warmup pass.
    revision = "07662d0d527654e72755adf65fa5428356adf1db"
    snapshot_download("chutesai/orpheus-3b-0.1-ft", revision=revision)
    self.model = OrpheusModel(model_name="chutesai/orpheus-3b-0.1-ft")

    # TODO: re-enable transcription when voice cloning is functional.
    # self.whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
    #    "openai/whisper-large-v3", torch_dtype=torch.float16, use_safetensors=True
    # )
    # self.whisper.to("cuda")
    # self.whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    # self.whisper_pipeline = pipeline(
    #    "automatic-speech-recognition",
    #    model=self.whisper,
    #    tokenizer=self.whisper_processor.tokenizer,
    #    feature_extractor=self.whisper_processor.feature_extractor,
    #    torch_dtype=torch.float16,
    #    device="cuda",
    # )

    # Warmup.
    with torch.no_grad():
        prompt = "Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot."
        syn_tokens = self.model.generate_speech(
            prompt=prompt,
            voice="tara",
        )
        with wave.open("/app/warmup.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            total_frames = 0
            chunk_counter = 0
            for audio_chunk in syn_tokens:
                chunk_counter += 1
                frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
                wf.writeframes(audio_chunk)
            print("Successully warmed up TTS")

        # TODO: once fixed, enable voice cloning and allow transcription instead of manual annotation of text.
        # text = transcribe(self, "/app/warmup_audio.wav")

    self.torchaudio = torchaudio
    self.torch = torch
    self.wave = wave


@chute.cord(
    public_api_path="/speak",
    public_api_method="POST",
    stream=False,
    output_content_type="audio/wav",
)
async def speak(self, args: InputArgs) -> Response:
    """
    Perform text to speech.
    """

    # TODO: when cloning works, extract input sample + transcribe if text wasn't provided.
    # input_path = None
    # if args.sample_audio_b64:
    #    input_path = load_audio(self, args.sample_audio_b64)
    #    if not args.sample_audio_text:
    #        try:
    #            args.sample_audio_text = transcribe(self, input_path)
    #        except:
    #            raise HTTPException(
    #                status_code=status.HTTP_400_BAD_REQUEST,
    #                detail=f"Invalid input audio_b64 provided: {exc}",
    #            )
    # else:
    #    args.sample_audio_text = None

    # Generate the audio.
    output_path = f"/tmp/{uuid.uuid4()}.wav"
    try:
        generate_args = args.model_dump()
        # generate_args.pop("sample_audio_b64", None)
        # generate_args.pop("sample_audio_text", None)
        # if input_path:
        #    generate_args.update(
        #        {"prompt_speech_path": input_path, "prompt_text": args.sample_audio_text}
        #    )
        with self.torch.no_grad():
            syn_tokens = self.model.generate_speech(**generate_args)
            with self.wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                total_frames = 0
                chunk_counter = 0
                for audio_chunk in syn_tokens:
                    chunk_counter += 1
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_chunk)
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
        # if input_path and os.path.exists(input_path):
        #    os.remove(input_path)
