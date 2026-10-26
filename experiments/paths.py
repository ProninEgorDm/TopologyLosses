"""Repository locations that do not depend on the current working directory."""

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dataset_dir() -> Path:
    return repo_root() / "src" / "VessMAP"


def resolve_path(path: str | Path | None, default: Path | None = None) -> Path:
    if path is None or path == "":
        if default is None:
            raise ValueError("A path is required.")
        return default
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    return candidate
