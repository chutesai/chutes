from chutes.chute.template.tei import build_tei_chute
from chutes.chute import NodeSelector

# To build a similar image:
# from chutes.image import Image
# image = (
#     Image(
#         username="chutes", name="tei", tag="1.6.0"
#     )
#     .from_base("parachutes/tei-base:1.6.0")
#     .set_workdir("/app")
#     .with_env("PATH", "/app/text-embeddings-inference/target/release:$PATH")
# )

chute = build_tei_chute(
    username="chutes",
    model_name="BAAI/bge-large-en-v1.5",
    node_selector=NodeSelector(num_gpus=1),
    image="chutes/tei:1.6.0",
    revision="refs/pr/5",
)
