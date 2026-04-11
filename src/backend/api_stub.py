"""
Placeholder for future OpenAI Whisper API backend.

To activate:
1. pip install openai
2. Set OPENAI_API_KEY env var
3. Implement OpenAIWhisperTranscriber.transcribe() using openai.Audio.transcriptions.create()
4. Register it in get_transcriber() in __init__.py
"""
from pathlib import Path
from src.output import Segment

class OpenAIWhisperTranscriber:
    def transcribe(self, audio_path: Path) -> list[Segment]:
        raise NotImplementedError("OpenAI API backend not yet implemented")
