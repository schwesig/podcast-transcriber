from src.pipeline.config import PipelineConfig

def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.first_pass_model == "distil-large-v3"
    assert cfg.yellow_pass_model == "turbo"
    assert cfg.red_pass_model == "large-v3"
    assert cfg.yellow_logprob == -0.6
    assert cfg.red_logprob == -1.0
    assert cfg.yellow_no_speech == 0.3
    assert cfg.red_no_speech == 0.6
    assert cfg.enable_large_pass is False
    assert cfg.vad is True
    assert cfg.beam_size == 5
    assert cfg.export_srt is False
    assert cfg.export_vtt is False
    assert cfg.dry_run is False
    assert cfg.verbose is False
    assert cfg.device == "auto"
    assert cfg.compute_type == "int8"
    assert cfg.model_cache_dir == ".models"
    assert cfg.language is None
    assert cfg.word_timestamps is False
