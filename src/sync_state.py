from pathlib import Path


def is_processed(episode_dir: Path) -> bool:
    """Return True if transcript.txt already exists for this episode."""
    return (episode_dir / "transcript.txt").is_file()


def mark_processed(episode_dir: Path) -> None:
    """Create the episode directory. Transcript file is written by the caller."""
    episode_dir.mkdir(parents=True, exist_ok=True)
