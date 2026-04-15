from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    # Models
    first_pass_model: str = "distil-large-v3"
    yellow_pass_model: str = "turbo"
    red_pass_model: str = "large-v3"

    # Scoring thresholds
    yellow_logprob: float = -0.6
    red_logprob: float = -1.0
    yellow_no_speech: float = 0.3
    red_no_speech: float = 0.6
    yellow_compression_high: float = 2.0
    red_compression_high: float = 2.5
    yellow_compression_low: float = 0.8
    red_compression_low: float = 0.5
    yellow_repeat_words: int = 3
    red_repeat_words: int = 5

    # Pipeline flags
    enable_large_pass: bool = False
    vad: bool = True
    beam_size: int = 5
    word_timestamps: bool = False
    export_srt: bool = False
    export_vtt: bool = False
    dry_run: bool = False
    verbose: bool = False

    # Hardware
    device: str = "auto"
    compute_type: str = "int8"
    model_cache_dir: str = ".models"

    # Language
    language: Optional[str] = None

    # Output
    output_dir: Optional[str] = None
