import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path


class Embedder_2:
    def __init__(self, path="models/Xenova/all-MiniLM-L6-v2"):
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        
        # Apple Silicon Mac M2 hardware acceleration configurations
        providers = [
            ('CoreMLExecutionProvider', {
                'MLComputeUnits': 'ALL',           # Uses M2 CPU, GPU, and Neural Engine
                'ModelFormat': 'MLProgram',        # Optimizes performance for Apple chips
                'RequireStaticInputShapes': '0',   # Dynamically accepts sentences of varying lengths
            }),
            'CPUExecutionProvider'                 # Fallback if a specific operator isn't supported by CoreML
        ]
        
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=providers
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text, normalize=True):
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts, normalize=True):
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled
