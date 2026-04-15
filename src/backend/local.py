import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from faster_whisper import WhisperModel

from src.audio import get_audio_duration
from src.config import TranscribeConfig
from src.output import Segment, RichSegment

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
        self._model_name = config.model

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

    def transcribe_rich(
        self,
        audio_path: Path,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        language: str | None = None,
    ) -> list[RichSegment]:
        lang = language or self._language
        segments_iter, _ = self._model.transcribe(
            str(audio_path),
            language=lang,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )
        duration = get_audio_duration(audio_path)
        results = []
        for s in segments_iter:
            results.append(RichSegment(
                start=s.start,
                end=s.end,
                text=s.text,
                model_used=self._model_name,
                difficulty="green",
                reason_flags=[],
                original_text=None,
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
                compression_ratio=s.compression_ratio,
            ))
            if duration > 0:
                pct = min(int(s.end / duration * 100), 100)
                print(f"\r  Transcribing... {pct}%", end="", flush=True)
        if duration > 0:
            print(f"\r  Transcribing... 100%")
        return results
