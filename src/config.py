from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TranscribeConfig:
    model: str = "base"
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    compute_type: str = "int8"    # "int8" | "float16" | "float32"
    language: Optional[str] = None
    output_formats: list[str] = field(default_factory=lambda: ["txt"])
    model_cache_dir: str = ".models"
