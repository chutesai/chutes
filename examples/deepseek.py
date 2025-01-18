from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute
from chutes.image import Image

image = (
    Image(
        username="chutes", name="vllm", tag="0.6.6.post1", readme="## vLLM - fast, flexible llm inference"
    )
    .from_base("parachutes/base-python:3.12.7")
    .run_command("pip install --no-cache 'vllm==0.6.6.post1' wheel packaging")
    .run_command("pip install --no-cache flash-attn")
    .run_command("pip uninstall -y xformers")
)

readme = """We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks."""

chute = build_vllm_chute(
    username="chutes",
    readme=readme,
    model_name="deepseek-ai/DeepSeek-V3",
    image=image,
    # The smallest supported GPU on chutes is 16GB VRAM, so we don't need
    # any significant detail here in the node selector, any node should work.
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=140,
    ),
    engine_args=dict(
        max_model_len=24000,
        num_scheduler_steps=1,
        trust_remote_code=True,
    ),
)
