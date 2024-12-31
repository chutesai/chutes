import base64
import traceback
from io import BytesIO
from pydantic import BaseModel
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="chutes",
        name="nfsw-classifier",
        tag="0.0.1",
        readme="## Image classifier to detect NSFW via Falconsai/nsfw_image_detection",
    )
    .from_base("parachutes/base-python:3.12.7")
    .add("parachute.png", "/app/parachute.png")
    .run_command("pip install torch transformers ")
    .run_command("pip install Pillow")
    .run_command("pip install detoxify")
)

chute = Chute(
    username="chutes",
    name="nsfw-classifier",
    readme="## NSFW Classifier\n\nThis is an NSFW classifier for both images (via Falconsai/nsfw_image_detection) and text (via detoxify)",
    image=image,
    node_selector=NodeSelector(gpu_count=1),
)


class ImageArgs(BaseModel):
    image_b64: str


class TextArgs(BaseModel):
    text: str


class ImageClassification(BaseModel):
    label: str
    confidence: float


class TextClassification(BaseModel):
    label: str
    scores: dict[str, float]


@chute.on_startup()
async def initialize(self):
    """
    Load the classification pipeline and perform a warmup.
    """
    from transformers import AutoModelForImageClassification, ViTImageProcessor
    from PIL import Image as IMG
    import torch
    from detoxify import Detoxify

    torch.set_float32_matmul_precision("high")
    model_name = "Falconsai/nsfw_image_detection"
    model = AutoModelForImageClassification.from_pretrained(model_name).to("cuda")
    self.torch = torch
    self.IMG = IMG
    self.model = torch.compile(model, mode="reduce-overhead")
    self.processor = ViTImageProcessor.from_pretrained(model_name)
    self.text_detector = Detoxify("multilingual")

    # Warmup...
    print("Warming up model...")
    image = IMG.open("/app/parachute.png").convert("RGB")
    with torch.no_grad():
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        _ = self.model(**inputs)
    print("Initialization complete, ready to classify.")


@chute.cord(public_api_path="/image")
async def classify_image(self, args: ImageArgs) -> ImageClassification:
    """
    Perform classification against an image.
    """
    try:
        image_bytes = base64.b64decode(args.image_b64)
        image = self.IMG.open(BytesIO(image_bytes)).convert("RGB")
        with self.torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to("cuda")
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = self.torch.nn.functional.softmax(logits, dim=-1)
        predicted_idx = logits.argmax(-1).item()
        confidence = probs[0][predicted_idx].item()
        return ImageClassification(
            label=self.model.config.id2label[predicted_idx],
            confidence=confidence,
        )
    except Exception as exc:
        print(f"Failed to perform classification: {exc}\n{traceback.format_exc()}")
        ImageClassification(
            label="error",
            confidence=1.0,
        )


@chute.cord(public_api_path="/text")
async def classify_text(self, args: TextArgs) -> TextClassification:
    """
    Classify text with detoxify.
    """
    results = self.text_detector.predict(args.text)
    max_class = max(results.items(), key=lambda x: x[1])
    return TextClassification(
        label=max_class[0] if max_class[1] > 0.5 else "normal",
        scores={clazz: float(score) for clazz, score in results.items()},
    )
