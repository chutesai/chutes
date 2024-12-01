# Chutes!

This package provides the command line interface and development kit for use with the chutes.ai platform.

## Glossary

Before getting into the weeds, it might be useful to understand the terminology.

### image

Images are simply docker images that all chutes (applications) will run on within the platform.

Images must meet two requirements:
- Containt a cuda installation, preferably version 12.2-12.6
- Contain a python 3.10+ installation, where `python` and `pip` are contained within the executable path `PATH`

### chute

A chute is essentially an application that runs on top of an image, within the platform.  Think of a chute as a single FastAPI application.

### cord

A cord is a single function within the chute.  In the FastAPI analogy, this would be a single route & method.

### graval

GraVal is the graphics card validation library used to help ensure the GPUs that miners claim to be running are authentic/correct.
The library performs VRAM capacity checks, matrix multiplications seeded by device information, etc.

You don't really need to know anything about graval, except that it runs as middleware within the chute to decrypt traffic from the validator and perform additional validation steps (filesystem checks, device info challenges, pings, etc.)

## Register

Currently, to become a user on the chutes platform, you must have a bittensor wallet and hotkey, as authentication is perform via bittensor hotkey signatures.
Once you are registered, you can create API keys that can be used with a simple "Authorization" header in your requests.

If you don't already have a wallet, you can create one by installing one of two packages:
1. `bittensor-wallet` in an unsensible twist of fate, you would need to first install rust to use this library, which is insane `brew install rust`
2. `bittensor<8` the good ole faithful all-in-one-package that doesn't require rust

Then, create a coldkey and hotkey according to the library you installed, e.g. with `pip install 'bittensor<8'`, you'd just run:
```bash
btcli wallet new_coldkey --n_words 24 --wallet.name chutes
btcli wallet new_hotkey --wallet.name chutes --n_words 24 --wallet.hotkey chuteshk
```

Once you have your hotkey, just run:
```bash
chutes register
```

To use the development environment, simply set the `CHUTES_API_URL` environment variable accordingly, e.g.:
```bash
CHUTES_API_URL=https://api.chutes.dev chutes register
```

Once you've completed the registration process, you'll have a file in `~/.chutes/config.ini` which contains the configuration for using chutes.

## Building an image

The first step in getting an application onto the chutes platform is to build an image.
This SDK includes an image creation helper library as well, and we have a recommended base image which includes python 3.12.7 and all necessary cuda packages: `parachutes/base-python:3.12.7`

Here is an entire chutes application, which has an image that includes `vllm` -- let's store it in `llama1b.py`:

```python
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute
from chutes.image import Image

image = (
    Image(username="chutes", name="vllm", tag="0.6.3", readme="## vLLM - fast, flexible llm inference")
    .from_base("parachutes/base-python:3.12.7")
    .run_command("pip install --no-cache 'vllm<0.6.4' wheel packaging")
    .run_command("pip install --no-cache flash-attn")
    .run_command("pip uninstall -y xformers")
)

chute = build_vllm_chute(
    username="chutes",
    readme="## Meta Llama 3.2 1B Instruct\n### Hello.",
    model_name="unsloth/Llama-3.2-1B-Instruct",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
    ),
)
```

The `chutes.image.Image` class includes many helper directives for environment variables, adding files, installing python from source, etc.

To build this image, you can use the chutes CLI:
```bash
chutes build llama1b:chute --public --wait --debug
```

Explanation of the flags:
- `--public` means we want this image to be public/available for ANY user to use -- use with care but we do like public/open source things!
- `--wait` means we want to stream the docker build logs back to the command line.  All image builds occur remotely on our platform, so without the `--wait` flag you just have to wait for the image to become available, whereas with this flag you can see real-time logs/status.
- `--debug` additional debug logging

## Deploying a chute

Once you have an image that is built and pushed and ready for use (see above), you can deploy applications on top of those.

To use the same example `llama1b.py` file outlined in the image building section above, we can deploy the llama-3.2-1b-instruct model with:
```bash
chutes deploy llama1b:chute --public
```

Be sure to carefully craft the `node_selector` option within the chute, to ensure the code runs on GPUs appropriate to the task.
```python
node_selector=NodeSelector(
    gpu_count=1,
    # All options.
    # gpu_count: int = Field(1, ge=1, le=8)
    # min_vram_gb_per_gpu: int = Field(16, ge=16, le=80)
    # require_sxm: bool = False
    # include: Optional[List[str]] = None
    # exclude: Optional[List[str]] = None
),
```

The most important fields are `gpu_count` and `min_vram_gb_per_gpu`.  If you wish to include specific GPUs, you can do so, where the `include` (or `exclude`) fields are the short identifier per model, e.g. `"a6000"`, `"a100"`, etc.  [All supported GPUs and their short identifiers](https://github.com/rayonlabs/chutes-api/blob/c0df10cff794c17684be9cf1111c00d84eb015b0/api/gpu.py#L17)

## Building custom/non-vllm chutes

Chutes are in fact completely arbitrary, so you can customize to your heart's content.

Here's an example chute showing some of this functionality:
```python
import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(username="chutes", name="base-python", tag="3.12.7", readme="## Base python+cuda image for chutes")
    .from_base("parachutes/base-python:3.12.7")
)

chute = Chute(
    username="test",
    name="example",
    readme="## Example Chute\n\n### Foo.\n\n```python\nprint('foo')```",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        # All options.
        # gpu_count: int = Field(1, ge=1, le=8)
        # min_vram_gb_per_gpu: int = Field(16, ge=16, le=80)
        # require_sxm: bool = False
        # include: Optional[List[str]] = None
        # exclude: Optional[List[str]] = None
    ),
)


class MicroArgs(BaseModel):
    foo: str = Field(..., max_length=100)
    bar: int = Field(0, gte=0, lte=100)
    baz: bool = False


class FullArgs(MicroArgs):
    bunny: Optional[str] = None
    giraffe: Optional[bool] = False
    zebra: Optional[int] = None


class ExampleOutput(BaseModel):
    foo: str
    bar: str
    baz: Optional[str]


@chute.on_startup()
async def initialize(self):
    self.billygoat = "billy"
    print("Inside the startup function!")


@chute.cord(minimal_input_schema=MicroArgs)
async def echo(input_args: FullArgs) -> str:
    return f"{chute.billygoat} says: {input_args}"


@chute.cord()
async def complex(input_args: MicroArgs) -> ExampleOutput:
    return ExampleOutput(foo=input_args.foo, bar=input_args.bar, baz=input_args.baz)


@chute.cord(
    output_content_type="image/png",
    public_api_path="/image",
    public_api_method="GET",
)
async def image() -> FileResponse:
    return FileResponse("parachute.png", media_type="image/png")


async def main():
    print(await echo("bar"))

if __name__ == "__main__":
    asyncio.run(main())
```

The main thing to notice here are the various the `@chute.cord(..)` decorators and `@chute.on_startup()` decorator.

Any code within the `@chute.on_startup()` decorated function(s) are executed when the application starts on the miner, it does not run in the local/client context.

Any function that you decorate with `@chute.cord()` becomes a function that runs within the chute, i.e. not locally - it's executed on the miners' hardware.

It is very important to give type hints to the functions, because the system will automatically generate OpenAPI schemas for each function for use with the public/hostname based API using API keys instead of requiring the chutes SDK to execute.

For a cord to be available from the public, subdomain based API, you need to specify `public_api_path` and `public_api_method`, and if the return content type is anything other than `application/json`, you'll want to specify that as well.

You can also spin up completely arbitrary webservers and do "passthrough" cords which pass along the request to the underlying webserver. This would be useful for things like using a webserver written in a different programming language, for example.

To see an example of passthrough functions and more complex functionality, see the [vllm template chute/helper](https://github.com/rayonlabs/chutes/blob/main/chutes/chute/template/vllm.py)

## Local testing

If you'd like to test your image/chute before actually deploying onto the platform, you can build the images with `--local`, then run in dev mode:
```bash
chutes build llama1b:chute --local
```

Then, you can start a container with that image:
```bash
docker run --rm -it -e CHUTES_EXECUTIION_CONTEXT=REMOTE -p 8000:8000 vllm:0.6.3 chutes run llama1b:chute --port 8000 --dev
```

Then, you can actually invoke your functions by setting the `CHUTES_DEV_URL` environment variable, e.g., supposing you add the following to `llama1b.py`:
```python
async def main():
    request = {
        "json": {
            "model": "unsloth/Llama-3.2-1B-Instruct",
            "messages": [{"role": "user", "content": "Give me a spicy mayo recipe."}],
            "temperature": 0.7,
            "seed": 42,
            "max_tokens": 3,
            "stream": True,
            "logprobs": True,
        }
    }
    async for data in chute.chat_stream(**request):
        if not data:
            continue
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

You can call the `chute.chat_stream(..)` function running on your local instance with:
```bash
CHUTES_DEV_URL=http://127.0.0.1:8000 python llama1b.py
```
