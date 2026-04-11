from src.backend.local import Transcriber, LocalWhisperTranscriber
from src.config import TranscribeConfig

def get_transcriber(config: TranscribeConfig) -> Transcriber:
    """Factory. Extend here to support OpenAI API backend."""
    return LocalWhisperTranscriber(config)

__all__ = ["Transcriber", "get_transcriber"]
