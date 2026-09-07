from pathlib import Path

from src.config import TranscribeConfig
from src.output import Segment, RichSegment

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

    def _run(self, audio_path: Path, language: str | None) -> dict:
        """Transcribe in a worker thread, printing elapsed time while it runs."""
        import threading
        import time
        import mlx_whisper

        print(f"  [mlx] model={self._model_name} language={language or 'auto'}", flush=True)

        result_holder: list = []
        error_holder: list = []
        done = threading.Event()

        def _work():
            try:
                result_holder.append(mlx_whisper.transcribe(
                    str(audio_path),
                    path_or_hf_repo=self._hf_model,
                    language=language,
                ))
            except BaseException as exc:  # surfaced after join, never swallowed
                error_holder.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=_work, daemon=True)
        t0 = time.monotonic()
        t.start()
        while not done.wait(timeout=1.0):
            elapsed = int(time.monotonic() - t0)
            print(f"\r  Transcribing... {elapsed}s", end="", flush=True)
        elapsed = int(time.monotonic() - t0)
        print(f"\r  Transcribing... {elapsed}s")

        if error_holder:
            raise error_holder[0]
        return result_holder[0]

    def transcribe(self, audio_path: Path) -> list[Segment]:
        result = self._run(audio_path, self._language)
        return [
            Segment(start=s["start"], end=s["end"], text=s["text"])
            for s in result.get("segments", [])
        ]

    def transcribe_rich(
        self,
        audio_path: Path,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        language: str | None = None,
    ) -> list[RichSegment]:
        """Rich transcription for the multi-pass pipeline.

        mlx-whisper exposes no VAD and no beam search, so beam_size, vad_filter
        and word_timestamps are accepted for interface parity and ignored. The
        quality metrics scorer.py needs (avg_logprob, no_speech_prob,
        compression_ratio) are present in mlx-whisper segments, but are reported
        per decoding window rather than per segment. Segments sharing a window
        score identically, so difficulty detection is coarser than with
        faster-whisper.
        """
        result = self._run(audio_path, language or self._language)
        return [
            RichSegment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                model_used=self._model_name,
                difficulty="green",  # populated by scorer.py
                reason_flags=[],
                original_text=None,
                avg_logprob=s.get("avg_logprob"),
                no_speech_prob=s.get("no_speech_prob"),
                compression_ratio=s.get("compression_ratio"),
            )
            for s in result.get("segments", [])
        ]

    @property
    def engine_name(self) -> str:
        return "mlx-whisper"
