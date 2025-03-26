import uuid
from io import BytesIO
from enum import Enum
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="chutes",
        name="kokoro-82m",
        tag="0.0.2",
        readme="## Text-to-speech using hexgrade/Kokoro-82M",
    )
    .from_base("parachutes/base-python:3.12.7")
    .set_user("root")
    .run_command("apt update")
    .apt_install(["espeak-ng", "git-lfs"])
    .set_user("chutes")
    .run_command("pip install phonemizer scipy munch torch transformers")
    .run_command("git lfs install")
    .run_command("git clone https://huggingface.co/hexgrad/Kokoro-82M")
    .run_command("mv -f Kokoro-82M/* /app/")
)

chute = Chute(
    username="chutes",
    name="Kokoro-82M",
    tagline="Text-to-speech with hexgrad/Kokoro-82M",
    readme="Kokoro is a frontier TTS model for its size of 82 million parameters (text in/audio out).",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gpu_per_gpu=48),
)


class VoicePack(str, Enum):
    DEFAULT = "af"
    BELLA = "af_bella"
    SARAH = "af_sarah"
    ADAM = "am_adam"
    MICHAEL = "am_michael"
    EMMA = "bf_emma"
    ISABELLA = "bf_isabella"
    GEORGE = "bm_george"
    LEWIS = "bm_lewis"
    NICOLE = "af_nicole"
    SKY = "af_sky"


class InputArgs(BaseModel):
    text: str
    voice: VoicePack = Field(
        default=VoicePack.DEFAULT, description="Voice pack selection for text-to-speech"
    )


@chute.on_startup()
async def initialize(self):
    """
    Load the model and all voice packs.
    """
    from models import build_model
    import torch
    import wave
    import numpy as np
    from kokoro import generate

    self.torch = torch
    self.wave = wave
    self.np = np
    self.generate = generate

    # Load the model.
    self.model = build_model("kokoro-v0_19.pth", "cuda")

    # Load all the voice packs.
    self.voice_packs = {}
    for voice_id in VoicePack:
        self.voice_packs[voice_id.value] = torch.load(
            f"voices/{voice_id.value}.pt", weights_only=True
        ).to("cuda")
        _, _ = generate(
            self.model, "warming up", self.voice_packs[voice_id.value], lang=voice_id.value[0]
        )


@chute.cord(
    public_api_path="/speak",
    public_api_method="POST",
    stream=False,
    output_content_type="audio/wav",
)
async def speak(self, args: InputArgs) -> StreamingResponse:
    """
    Generate SSE audio chunks from input text.
    """
    audio_data, out_ps = self.generate(
        self.model, args.text, self.voice_packs[args.voice.value], lang=args.voice.value[0]
    )
    buffer = BytesIO()
    audio_int16 = (audio_data * 32768).astype(self.np.int16)
    with self.wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    filename = f"{str(uuid.uuid4())}.wav"
    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
