# tests/test_config.py
from src.config import TranscribeConfig

def test_default_config():
    cfg = TranscribeConfig()
    assert cfg.model == "base"
    assert cfg.device == "auto"
    assert cfg.compute_type == "int8"
    assert cfg.output_formats == ["txt"]
    assert cfg.language is None

def test_custom_config():
    cfg = TranscribeConfig(model="large-v3", device="cuda", output_formats=["txt", "json"])
    assert cfg.model == "large-v3"
    assert cfg.device == "cuda"
    assert cfg.output_formats == ["txt", "json"]
