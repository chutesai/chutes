from pydantic import BaseModel, Field
from typing import List


class NodeSelector(BaseModel):
    gpu_count: int = Field(1, ge=1, le=8)
    min_vram_gb_per_gpu: int = Field(16, ge=16, le=80)
    exclude: List[str] = None
    include: List[str] = None
    require_sxm: bool = False
