from pathlib import Path

from src.config import TranscribeConfig
from src.output import Segment

_MODEL_MAP: dict[str, str] = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


class MlxWhisperTranscriber:
    def __init__(self, config: TranscribeConfig):
        self._hf_model = _MODEL_MAP[config.model]
        self._language = config.language
        self._model_name = config.model

    def transcribe(self, audio_path: Path) -> list[Segment]:
        import mlx_whisper
        print(f"  [mlx] model={self._model_name} language={self._language or 'auto'}", flush=True)
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self._hf_model,
            language=self._language,
        )
        segments = []
        for s in result.get("segments", []):
            segments.append(Segment(start=s["start"], end=s["end"], text=s["text"]))
        return segments

    @property
    def engine_name(self) -> str:
        return "mlx-whisper"
