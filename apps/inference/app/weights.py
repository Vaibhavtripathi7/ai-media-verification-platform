from pathlib import Path

from huggingface_hub import snapshot_download

from .config import Settings


def download(settings: Settings) -> Path:
    """Fetch the pinned model files into the Hugging Face cache, or find them there."""
    return Path(
        snapshot_download(
            settings.model_repo,
            revision=settings.model_revision,
            allow_patterns=["config.json", "preprocessor_config.json", settings.model_file],
        )
    )


if __name__ == "__main__":
    print(download(Settings.from_env()))
