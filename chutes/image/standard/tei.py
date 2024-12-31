TEI = "chutes/tei:1.6.0"

# The "parachutes/tei:1.6.0" base image dockerfile is as follows:
# FROM ghcr.io/huggingface/text-embeddings-inference:1.6 AS tei
# FROM parachutes/base-python:3.12.7
# COPY --from=tei /usr/local/bin/text-embeddings-router /usr/local/bin/text-embeddings-router

# That image is then used when creating the chutes image (which is the same, plus chutes SDK):
# from chutes.image import Image
# image = (
#     Image(
#         username="chutes", name="tei", tag="1.6.0"
#     )
#     .from_base("parachutes/tei:1.6.0")
# )
