import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .imaging import Preprocessor


class Detector:
    """Single-logit AI-image classifier exported to ONNX. A higher logit means more
    likely generated. Calibrating it into a probability is left to the caller."""

    def __init__(self, model_dir: Path, onnx_file: str, threads: int, max_pixels: int):
        config = json.loads((model_dir / "config.json").read_text())
        if config.get("num_labels") != 1:
            raise ValueError(
                f"expected a single-logit model, got num_labels={config.get('num_labels')}"
            )

        self.preprocess = Preprocessor.from_config(
            json.loads((model_dir / "preprocessor_config.json").read_text()), max_pixels
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_dir / onnx_file), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, batch: np.ndarray) -> np.ndarray:
        (logits,) = self.session.run(None, {self.input_name: batch})
        return logits[:, 0]
