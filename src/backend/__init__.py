import platform
from typing import Callable

from src.backend.local import Transcriber, LocalWhisperTranscriber
from src.config import TranscribeConfig

# Models without an mlx variant — fall back to faster-whisper
_NO_MLX_VARIANT = {"turbo", "distil-large-v3"}


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mlx_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _make_faster_whisper(config: TranscribeConfig) -> Transcriber:
    return LocalWhisperTranscriber(config)


def _make_mlx(config: TranscribeConfig) -> Transcriber:
    if config.model in _NO_MLX_VARIANT:
        print(f"  [backend] model '{config.model}' has no mlx variant, using faster-whisper")
        return LocalWhisperTranscriber(config)
    if not _mlx_available():
        print("  [backend] mlx-whisper not installed, using faster-whisper")
        return LocalWhisperTranscriber(config)
    from src.backend.mlx_whisper import MlxWhisperTranscriber
    return MlxWhisperTranscriber(config)


_REGISTRY: dict[str, Callable[[TranscribeConfig], Transcriber]] = {
    "faster-whisper": _make_faster_whisper,
    "mlx":            _make_mlx,
}


def get_transcriber(config: TranscribeConfig) -> Transcriber:
    """Factory. Add new backends to _REGISTRY above."""
    backend = config.backend
    if backend == "auto":
        backend = "mlx" if (_is_apple_silicon() and _mlx_available()) else "faster-whisper"
    factory = _REGISTRY.get(backend)
    if factory is None:
        raise ValueError(f"Unknown backend '{backend}'. Valid: {list(_REGISTRY)}")
    return factory(config)


__all__ = ["Transcriber", "get_transcriber"]
