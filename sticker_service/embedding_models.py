from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from sticker_service.config import CFG

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_LIBROSA", "1")
os.environ.setdefault("TRANSFORMERS_NO_AUDIO", "1")
os.environ["HF_HUB_CACHE"] = str(CFG.MODEL_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CFG.MODEL_DIR)


def _disable_audio_backends() -> None:
    try:
        from transformers.utils import import_utils
        import_utils._librosa_available = False
        import_utils._soundfile_available = False
        import_utils._torchaudio_available = False
        import_utils._torchcodec_available = False
    except Exception:
        pass


class TextEmbedder:
    """
    Generic embedding wrapper for encoder models.
    - pooling: mean or cls
    - normalize: L2 normalization
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        precision: str = "fp32",
        pooling: str = "mean",
    ):
        self.model_name = model_name
        self.precision = precision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        dtype = torch.float16 if precision == "fp16" else torch.float32
        if self.device == "cpu" and dtype == torch.float16:
            dtype = torch.float32
        self.dtype = dtype

        _disable_audio_backends()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CFG.MODEL_DIR))
        try:
            self.model = AutoModel.from_pretrained(model_name, dtype=self.dtype, cache_dir=str(CFG.MODEL_DIR))
        except TypeError:
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype, cache_dir=str(CFG.MODEL_DIR))
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        outs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            last = out.last_hidden_state  # [B, T, H]

            if self.pooling == "cls":
                vec = last[:, 0]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                masked = last * mask
                summed = masked.sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1)
                vec = summed / denom  # [B, H]

            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            outs.append(vec.detach().cpu().float().numpy().astype(np.float32))

        return np.concatenate(outs, axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class TextReranker:
    """
    Cross-encoder reranker for query-doc pairs.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        precision: str = "fp32",
    ):
        self.model_name = model_name
        self.precision = precision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if precision == "fp16" else torch.float32
        if self.device == "cpu" and dtype == torch.float16:
            dtype = torch.float32
        self.dtype = dtype

        _disable_audio_backends()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CFG.MODEL_DIR))
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                dtype=self.dtype,
                cache_dir=str(CFG.MODEL_DIR),
            )
        except TypeError:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                cache_dir=str(CFG.MODEL_DIR),
            )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, docs: List[str], batch_size: int = 16) -> np.ndarray:
        if not docs:
            return np.zeros((0,), dtype=np.float32)
        scores = []
        for i in range(0, len(docs), batch_size):
            chunk = docs[i : i + batch_size]
            inputs = self.tokenizer(
                [query] * len(chunk),
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            logits = out.logits.squeeze(-1)
            scores.append(logits.detach().cpu().float().numpy().astype(np.float32))
        return np.concatenate(scores, axis=0)
