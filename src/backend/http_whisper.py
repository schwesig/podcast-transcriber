import os
from pathlib import Path

import httpx

from src.audio import get_audio_duration
from src.config import TranscribeConfig
from src.output import Segment, RichSegment

class HttpWhisperTranscriber:
    """Transcribes via a remote OpenAI-compatible /v1/audio/transcriptions endpoint."""

    def __init__(self, config: TranscribeConfig):
        url = (
            os.environ.get("WHISPER_SERVER_URL")
            or getattr(config, "whisper_server_url", None)
        )
        if not url:
            raise ValueError(
                "HTTP Whisper backend requires a server URL. "
                "Set WHISPER_SERVER_URL env var or whisper_server_url= in feeds.txt."
            )
        self._url = url.rstrip("/")
        self._language = config.language
        self._model_name = config.model or "whisper-1"
        self._timeout = 600.0  # long episodes can take a while

    @property
    def engine_name(self) -> str:
        return "http-whisper"

    def transcribe(self, audio_path: Path) -> list[Segment]:
        return self._call(audio_path)

    def transcribe_rich(
        self,
        audio_path: Path,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        language: str | None = None,
    ) -> list[RichSegment]:
        segments = self._call(audio_path, language=language)
        return [
            RichSegment(
                start=s.start,
                end=s.end,
                text=s.text,
                model_used=self._model_name,
                difficulty="green",
                reason_flags=[],
                original_text=None,
                avg_logprob=None,
                no_speech_prob=None,
                compression_ratio=None,
            )
            for s in segments
        ]

    def _call(self, audio_path: Path, language: str | None = None) -> list[Segment]:
        lang = language or self._language
        duration = get_audio_duration(audio_path)
        print(f"  [http-whisper] POST {self._url}/v1/audio/transcriptions ({audio_path.name})")

        with open(audio_path, "rb") as fh:
            data = {"model": self._model_name, "response_format": "verbose_json"}
            if lang:
                data["language"] = lang
            resp = httpx.post(
                f"{self._url}/v1/audio/transcriptions",
                files={"file": (audio_path.name, fh, "audio/mpeg")},
                data=data,
                timeout=self._timeout,
            )

        resp.raise_for_status()
        body = resp.json()

        # verbose_json returns {"segments": [{start, end, text, ...}]}
        raw_segments = body.get("segments") or []
        if not raw_segments:
            # fallback: server returned plain text only
            text = body.get("text", "")
            return [Segment(start=0.0, end=duration, text=text)] if text else []

        return [
            Segment(start=s["start"], end=s["end"], text=s["text"])
            for s in raw_segments
        ]
