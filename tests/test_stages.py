from pathlib import Path
from unittest.mock import patch
from src.output import RichSegment
from src.pipeline.config import PipelineConfig
from src.pipeline.stages import run_pipeline


FIXTURE = Path("tests/fixtures/hello.wav")


def _make_rich_seg(text="hello", difficulty="green", logprob=-0.3, no_speech=0.05):
    return RichSegment(
        start=0.0, end=1.0, text=text,
        model_used="distil-large-v3", difficulty=difficulty,
        reason_flags=[], original_text=None,
        avg_logprob=logprob, no_speech_prob=no_speech, compression_ratio=1.2,
    )


def test_run_pipeline_dry_run(tmp_path, capsys):
    cfg = PipelineConfig(dry_run=True, output_dir=str(tmp_path))
    with patch("src.pipeline.stages._transcribe_file") as mock_t:
        mock_t.return_value = [_make_rich_seg("hello world")]
        run_pipeline(FIXTURE, cfg)
    captured = capsys.readouterr()
    assert "green" in captured.out
    # no output files written in dry-run
    assert not list(tmp_path.glob("*.txt"))


def test_run_pipeline_writes_txt(tmp_path):
    cfg = PipelineConfig(
        first_pass_model="tiny",
        device="cpu",
        compute_type="int8",
        output_dir=str(tmp_path),
        vad=False,
    )
    run_pipeline(FIXTURE, cfg)
    txt_files = list(tmp_path.glob("*.txt"))
    assert len(txt_files) == 1


def test_run_pipeline_writes_json(tmp_path):
    cfg = PipelineConfig(
        first_pass_model="tiny",
        device="cpu",
        compute_type="int8",
        output_dir=str(tmp_path),
        vad=False,
    )
    run_pipeline(FIXTURE, cfg)
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1


def test_run_pipeline_writes_srt_when_requested(tmp_path):
    cfg = PipelineConfig(
        first_pass_model="tiny",
        device="cpu",
        compute_type="int8",
        output_dir=str(tmp_path),
        export_srt=True,
        vad=False,
    )
    run_pipeline(FIXTURE, cfg)
    assert list(tmp_path.glob("*.srt"))


def test_run_pipeline_writes_vtt_when_requested(tmp_path):
    cfg = PipelineConfig(
        first_pass_model="tiny",
        device="cpu",
        compute_type="int8",
        output_dir=str(tmp_path),
        export_vtt=True,
        vad=False,
    )
    run_pipeline(FIXTURE, cfg)
    assert list(tmp_path.glob("*.vtt"))
