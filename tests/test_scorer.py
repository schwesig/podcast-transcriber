import pytest
from src.output import RichSegment
from src.pipeline.config import PipelineConfig
from src.pipeline.scorer import score_segment, HALLUCINATION_PHRASES


def _seg(**kwargs) -> RichSegment:
    defaults = dict(
        start=0.0, end=2.0, text="Hello world this is normal speech.",
        model_used="distil-large-v3", difficulty="green", reason_flags=[],
        original_text=None, avg_logprob=-0.3, no_speech_prob=0.1,
        compression_ratio=1.2,
    )
    defaults.update(kwargs)
    return RichSegment(**defaults)


cfg = PipelineConfig()


def test_green_segment():
    seg = _seg()
    result = score_segment(seg, cfg)
    assert result.difficulty == "green"
    assert result.reason_flags == []


def test_yellow_low_logprob():
    seg = _seg(avg_logprob=-0.7)
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "low_logprob" in result.reason_flags


def test_red_very_low_logprob():
    seg = _seg(avg_logprob=-1.1)
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"
    assert "low_logprob" in result.reason_flags


def test_yellow_no_speech():
    seg = _seg(no_speech_prob=0.4)
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "no_speech" in result.reason_flags


def test_red_no_speech():
    seg = _seg(no_speech_prob=0.7)
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"
    assert "no_speech" in result.reason_flags


def test_yellow_high_compression():
    seg = _seg(compression_ratio=2.1)
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "compression_ratio" in result.reason_flags


def test_red_high_compression():
    seg = _seg(compression_ratio=2.6)
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"
    assert "compression_ratio" in result.reason_flags


def test_yellow_repeated_words():
    seg = _seg(text="the the the is something")
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "repeated_words" in result.reason_flags


def test_red_repeated_words():
    seg = _seg(text="the the the the the is something")
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"
    assert "repeated_words" in result.reason_flags


def test_yellow_short_segment():
    seg = _seg(text="OK")
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "short_segment" in result.reason_flags


def test_yellow_long_segment():
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    long_text = " ".join((words * 15)[:65])
    seg = _seg(text=long_text)
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "long_segment" in result.reason_flags


def test_red_hallucination_phrase():
    seg = _seg(text="Thanks for watching!")
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"
    assert "hallucination_phrase" in result.reason_flags


def test_red_wins_over_yellow():
    seg = _seg(avg_logprob=-0.7, no_speech_prob=0.7)
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"


def test_multiple_flags_accumulated():
    seg = _seg(avg_logprob=-0.7, no_speech_prob=0.4)
    result = score_segment(seg, cfg)
    assert "low_logprob" in result.reason_flags
    assert "no_speech" in result.reason_flags
