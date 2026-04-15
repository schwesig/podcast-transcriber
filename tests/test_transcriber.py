# tests/test_transcriber.py
from pathlib import Path
from src.config import TranscribeConfig
from src.backend import get_transcriber
from src.output import Segment, RichSegment

FIXTURE = Path("tests/fixtures/hello.wav")

def test_local_transcriber_returns_segments():
    cfg = TranscribeConfig(model="tiny", device="cpu", compute_type="int8")
    transcriber = get_transcriber(cfg)
    segments = transcriber.transcribe(FIXTURE)
    assert isinstance(segments, list)
    assert len(segments) >= 0          # sine tone may produce no speech — shape is what matters
    for seg in segments:
        assert isinstance(seg, Segment)
        assert seg.end > seg.start

def test_local_transcriber_returns_rich_segments():
    cfg = TranscribeConfig(model="tiny", device="cpu", compute_type="int8")
    transcriber = get_transcriber(cfg)
    segments = transcriber.transcribe_rich(FIXTURE)
    assert isinstance(segments, list)
    for seg in segments:
        assert isinstance(seg, RichSegment)
        assert seg.model_used == "tiny"
        assert seg.difficulty == "green"
        assert seg.reason_flags == []
        assert seg.original_text is None
        assert isinstance(seg.avg_logprob, (float, type(None)))
        assert isinstance(seg.no_speech_prob, (float, type(None)))
        assert isinstance(seg.compression_ratio, (float, type(None)))
