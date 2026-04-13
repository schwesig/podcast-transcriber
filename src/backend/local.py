import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from faster_whisper import WhisperModel

from src.audio import get_audio_duration
from src.config import TranscribeConfig
from src.output import Segment

@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[Segment]: ...

class LocalWhisperTranscriber:
    def __init__(self, config: TranscribeConfig):
        device = config.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._model = WhisperModel(
            config.model,
            device=device,
            compute_type=config.compute_type,
            download_root=config.model_cache_dir,
        )
        self._language = config.language

    def transcribe(self, audio_path: Path) -> list[Segment]:
        segments_iter, _ = self._model.transcribe(
            str(audio_path),
            language=self._language,
            beam_size=5,
        )
        duration = get_audio_duration(audio_path)
        segments = []
        for s in segments_iter:
            segments.append(Segment(start=s.start, end=s.end, text=s.text))
            if duration > 0:
                pct = min(int(s.end / duration * 100), 100)
                print(f"\r  Transcribing... {pct}%", end="", flush=True)
        if duration > 0:
            print(f"\r  Transcribing... 100%")
        return segments
