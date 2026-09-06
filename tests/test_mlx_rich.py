"""Tests for MlxWhisperTranscriber.transcribe_rich and pipeline backend routing."""
from pathlib import Path
from unittest.mock import patch

from src.backend.mlx_whisper import MlxWhisperTranscriber
from src.config import TranscribeConfig
from src.output import RichSegment
from src.pipeline.config import PipelineConfig
from src.pipeline.stages import _make_transcriber

# shape mirrors a real mlx_whisper.transcribe() result
_MLX_RESULT = {
    "text": " Hello world.",
    "language": "en",
    "segments": [
        {
            "id": 0, "seek": 0, "start": 0.0, "end": 3.5, "text": " Hello world.",
            "tokens": [50364, 2425], "temperature": 0.0,
            "avg_logprob": -0.32, "compression_ratio": 1.09, "no_speech_prob": 0.004,
        }
    ],
}


def _transcriber(model: str = "small") -> MlxWhisperTranscriber:
    return MlxWhisperTranscriber(TranscribeConfig(model=model, backend="mlx"))


def test_transcribe_rich_returns_rich_segments():
    t = _transcriber()
    with patch.object(MlxWhisperTranscriber, "_run", return_value=_MLX_RESULT):
        segments = t.transcribe_rich(Path("dummy.wav"))
    assert len(segments) == 1
    seg = segments[0]
    assert isinstance(seg, RichSegment)
    assert seg.text == " Hello world."
    assert seg.model_used == "small"
    assert seg.difficulty == "green"
    assert seg.reason_flags == []
    assert seg.original_text is None


def test_transcribe_rich_carries_scorer_metrics():
    """scorer.py needs these three — they must survive the mlx conversion."""
    t = _transcriber()
    with patch.object(MlxWhisperTranscriber, "_run", return_value=_MLX_RESULT):
        seg = t.transcribe_rich(Path("dummy.wav"))[0]
    assert seg.avg_logprob == -0.32
    assert seg.compression_ratio == 1.09
    assert seg.no_speech_prob == 0.004


def test_transcribe_rich_tolerates_missing_metrics():
    """Older mlx builds may omit metrics; None is valid for RichSegment."""
    bare = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
    t = _transcriber()
    with patch.object(MlxWhisperTranscriber, "_run", return_value=bare):
        seg = t.transcribe_rich(Path("dummy.wav"))[0]
    assert seg.avg_logprob is None
    assert seg.no_speech_prob is None
    assert seg.compression_ratio is None


def test_run_reraises_worker_exception():
    """A failure inside the worker thread must surface, not become IndexError."""
    t = _transcriber()
    with patch("mlx_whisper.transcribe", side_effect=RuntimeError("boom")):
        try:
            t.transcribe(Path("dummy.wav"))
        except RuntimeError as exc:
            assert str(exc) == "boom"
        else:
            raise AssertionError("expected RuntimeError to propagate")


def test_pipeline_honours_backend_setting():
    """PipelineConfig.backend must reach the factory (was hardcoded to local)."""
    cfg = PipelineConfig(backend="faster-whisper")
    with patch("src.pipeline.stages.get_transcriber") as factory:
        _make_transcriber("small", cfg)
    assert factory.call_args.args[0].backend == "faster-whisper"


def test_pipeline_passes_model_and_language():
    cfg = PipelineConfig(backend="mlx", language="de")
    with patch("src.pipeline.stages.get_transcriber") as factory:
        _make_transcriber("medium", cfg)
    tc = factory.call_args.args[0]
    assert tc.model == "medium"
    assert tc.language == "de"
    assert tc.backend == "mlx"
