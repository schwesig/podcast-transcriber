import platform
from typing import Callable

from src.backend.local import Transcriber, LocalWhisperTranscriber
from src.config import TranscribeConfig

def _mlx_supports(model: str) -> bool:
    """Whether mlx-whisper has a variant for this model.

    Asks _MODEL_MAP rather than keeping a blocklist: anything absent from it
    (large-v2, the .en models, anything added to faster-whisper later) would
    otherwise pass the check and fail in MlxWhisperTranscriber's constructor.
    """
    from src.backend.mlx_whisper import _MODEL_MAP
    return model in _MODEL_MAP


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
    if not _mlx_available():
        print("  [backend] mlx-whisper not installed, using faster-whisper")
        return LocalWhisperTranscriber(config)
    if not _mlx_supports(config.model):
        print(f"  [backend] model '{config.model}' has no mlx variant, using faster-whisper")
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
        use_mlx = (
            _is_apple_silicon()
            and _mlx_available()
            and _mlx_supports(config.model)
        )
        backend = "mlx" if use_mlx else "faster-whisper"
    factory = _REGISTRY.get(backend)
    if factory is None:
        raise ValueError(f"Unknown backend '{backend}'. Valid: {list(_REGISTRY)}")
    return factory(config)


__all__ = ["Transcriber", "get_transcriber"]
