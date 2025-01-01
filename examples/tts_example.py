import json
import base64
import tempfile
from loguru import logger
from pydantic import BaseModel, Field
from typing import Optional
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="chutes",
        name="f5-tts",
        tag="0.0.1",
        readme="## Text-to-speech using F5-TTS",
    )
    .from_base("parachutes/base-python:3.12.7")
    .add("cortana.mp3", "/app/cortana.mp3")
    .run_command("pip install torch transformers")
    .run_command("pip install git+https://github.com/SWivid/F5-TTS.git")
    .set_user("root")
    .run_command("apt update")
    .apt_install("ffmpeg")
    .set_user("chutes")
)

chute = Chute(
    username="chutes",
    name="f5-tts",
    tagline="Text-to-speech with F5-TTS",
    readme="## F5-TTS: Diffusion Transformer with ConvNeXt V2, faster trained and inference.",
    image=image,
    node_selector=NodeSelector(gpu_count=1),
)


class InputArgs(BaseModel):
    text: str
    chunk_duration: Optional[float] = Field(default=5.0, ge=0.5, le=15.0)
    ref_audio_b64: Optional[str] = None
    ref_audio_text: Optional[str] = None


class StreamChunk(BaseModel):
    audio_b64: str
    timestamp: float
    is_final: bool


@chute.on_startup()
async def initialize(self):
    """
    Load the model, reference audio, etc.
    """
    import torch
    import torchaudio
    from cached_path import cached_path
    from f5_tts.infer.utils_infer import (
        infer_batch_process,
        preprocess_ref_audio_text,
        load_vocoder,
        load_model,
    )
    from f5_tts.model.backbones.dit import DiT

    self.device = "cuda"
    self.torch = torch
    self.torchaudio = torchaudio
    self.preprocess = preprocess_ref_audio_text
    self.infer = infer_batch_process
    self.model = load_model(
        model_cls=DiT,
        model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors")),
        mel_spec_type="vocos",
        vocab_file="/home/chutes/.local/lib/python3.12/site-packages/f5_tts/infer/examples/vocab.txt",
        ode_method="euler",
        use_ema=True,
        device=self.device,
    ).to(self.device)
    self.vocoder = load_vocoder(is_local=False)

    # Reference audio and text
    ref_audio = "/app/cortana.mp3"
    ref_text = "Ah, now I see. There's a submerged section that connects these towers to the outlying structures."
    self.default_ref_audio, self.default_ref_text = preprocess_ref_audio_text(ref_audio, ref_text)

    self.default_audio_obj, self.default_sr = torchaudio.load(ref_audio)
    infer_batch_process(
        (self.default_audio_obj, self.default_sr),
        ref_text,
        ["Warm-up text."],
        self.model,
        self.vocoder,
        device=self.device,
    )


@chute.cord(public_api_path="/speak", public_api_method="POST", stream=True)
async def speak(self, args: InputArgs) -> StreamChunk:
    """
    Generate SSE audio chunks from input text.
    """
    ref_audio, ref_text, audio, sr = (
        self.default_ref_audio,
        self.default_ref_text,
        self.default_audio_obj,
        self.default_sr,
    )
    if args.ref_audio_b64:
        try:
            with tempfile.NamedTemporaryFile() as tmpfile:
                tmpfile.write(base64.b64decode(args.ref_audio_b64))
                tmpfile.flush()
                tmpfile.seek(0)
                ref_audio, ref_text = self.preprocess(tmpfile.name, args.ref_audio_text or "")
                audio, sr = self.torchaudio.load(tmpfile.name)
        except Exception as exc:
            logger.error(f"Error loading reference audio, reverting to default: {exc}")

    with self.torch.no_grad():
        audio_data, sample_rate, _ = self.infer(
            (audio, sr), ref_text, [args.text], self.model, self.vocoder, device=self.device
        )
        chunk_size = int(sample_rate * args.chunk_duration)
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            if len(chunk) > 0:
                chunk_bytes = chunk.tobytes()
                chunk_b64 = base64.b64encode(chunk_bytes).decode("utf-8")
                chunk_data = {
                    "audio": chunk_b64,
                    "timestamp": i / sample_rate,
                    "is_final": (i + chunk_size) >= len(audio_data),
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
