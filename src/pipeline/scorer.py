from dataclasses import replace
from src.output import RichSegment
from src.pipeline.config import PipelineConfig


HALLUCINATION_PHRASES = [
    "thanks for watching",
    "subscribe",
    "like and subscribe",
    "vielen dank fürs zuschauen",
    "don't forget to subscribe",
    "see you in the next video",
    "bye bye",
    "please subscribe",
]


def _max_consecutive_repeated_words(text: str) -> int:
    words = text.lower().split()
    if not words:
        return 0
    max_run = 1
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run


def score_segment(seg: RichSegment, cfg: PipelineConfig) -> RichSegment:
    """Classify segment difficulty and populate reason_flags. Returns new RichSegment."""
    flags: list[str] = []
    red_flags: list[str] = []

    text = seg.text.strip()
    word_count = len(text.split())

    # --- logprob ---
    if seg.avg_logprob is not None:
        if seg.avg_logprob < cfg.red_logprob:
            red_flags.append("low_logprob")
        elif seg.avg_logprob < cfg.yellow_logprob:
            flags.append("low_logprob")

    # --- no_speech_prob ---
    if seg.no_speech_prob is not None:
        if seg.no_speech_prob > cfg.red_no_speech:
            red_flags.append("no_speech")
        elif seg.no_speech_prob > cfg.yellow_no_speech:
            flags.append("no_speech")

    # --- compression_ratio ---
    if seg.compression_ratio is not None:
        cr = seg.compression_ratio
        if cr > cfg.red_compression_high or cr < cfg.red_compression_low:
            red_flags.append("compression_ratio")
        elif cr > cfg.yellow_compression_high or cr < cfg.yellow_compression_low:
            flags.append("compression_ratio")

    # --- repeated words ---
    max_run = _max_consecutive_repeated_words(text)
    if max_run >= cfg.red_repeat_words:
        red_flags.append("repeated_words")
    elif max_run >= cfg.yellow_repeat_words:
        flags.append("repeated_words")

    # --- segment length ---
    if word_count < 2:
        flags.append("short_segment")
    elif word_count > 60:
        flags.append("long_segment")

    # --- hallucination phrases ---
    text_lower = text.lower()
    for phrase in HALLUCINATION_PHRASES:
        if phrase in text_lower:
            red_flags.append("hallucination_phrase")
            break

    all_flags = flags + red_flags
    if red_flags:
        difficulty = "red"
    elif flags:
        difficulty = "yellow"
    else:
        difficulty = "green"

    return replace(seg, difficulty=difficulty, reason_flags=all_flags)
