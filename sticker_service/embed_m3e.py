from __future__ import annotations
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from typing import List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from sticker_service.config import CFG

os.environ["HF_HUB_CACHE"] = str(CFG.MODEL_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CFG.MODEL_DIR)

class M3EEmbedder:
    """
    纯 torch 版本 m3e embedding。
    - mean pooling
    - L2 normalize
    """
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CFG.MODEL_DIR))
        self.model = AutoModel.from_pretrained(model_name, cache_dir=str(CFG.MODEL_DIR)).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            # m3e-small hidden size generally 512，但别写死，返回空即可
            return np.zeros((0, 0), dtype=np.float32)

        outs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            inputs = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            last = out.last_hidden_state  # [B, T, H]
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            masked = last * mask
            summed = masked.sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            mean = summed / denom  # [B, H]
            mean = torch.nn.functional.normalize(mean, p=2, dim=1)
            outs.append(mean.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(outs, axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        v = self.encode([text])
        return v[0]
