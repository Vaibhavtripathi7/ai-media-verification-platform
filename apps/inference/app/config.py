import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_repo: str = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
    # Pinned: this repo has had its weights and config replaced in place before.
    model_revision: str = "ac6ee457bea904a373065754107451793b56db00"
    model_file: str = "onnx/model.onnx"
    # Physical cores. Running on hyperthreads as well made this model slower.
    threads: int = 4
    max_body_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 40_000_000
    # Requests handled at once, counting reads, decodes and the model queue.
    # Past this, new requests get a 503 straight away instead of waiting.
    max_pending: int = 8

    @property
    def model_id(self) -> str:
        return f"{self.model_repo}@{self.model_revision[:7]}"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            model_repo=env.get("MODEL_REPO", cls.model_repo),
            model_revision=env.get("MODEL_REVISION", cls.model_revision),
            model_file=env.get("MODEL_FILE", cls.model_file),
            threads=int(env.get("INFERENCE_THREADS", cls.threads)),
            max_body_bytes=int(env.get("MAX_BODY_BYTES", cls.max_body_bytes)),
            max_pixels=int(env.get("MAX_IMAGE_PIXELS", cls.max_pixels)),
            max_pending=int(env.get("MAX_PENDING", cls.max_pending)),
        )
