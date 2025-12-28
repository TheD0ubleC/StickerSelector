from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sticker_service.config import CFG
from sticker_service.utils import read_json, write_json


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    embed_model: str
    precision: str
    pooling: str
    mode: str = "embed"
    reranker_model: Optional[str] = None
    reranker_precision: Optional[str] = None


MODEL_SPECS = [
    ModelSpec(
        key="m3e-small-fp32",
        label="moka-ai/m3e-small fp32",
        embed_model="moka-ai/m3e-small",
        precision="fp32",
        pooling="mean",
    ),
    ModelSpec(
        key="m3e-small-fp16",
        label="moka-ai/m3e-small fp16",
        embed_model="moka-ai/m3e-small",
        precision="fp16",
        pooling="mean",
    ),
    ModelSpec(
        key="bge-small-zh-v1.5-fp32",
        label="BAAI/bge-small-zh-v1.5 fp32",
        embed_model="BAAI/bge-small-zh-v1.5",
        precision="fp32",
        pooling="cls",
    ),
    ModelSpec(
        key="bge-small-zh-v1.5-fp16",
        label="BAAI/bge-small-zh-v1.5 fp16",
        embed_model="BAAI/bge-small-zh-v1.5",
        precision="fp16",
        pooling="cls",
    ),
    ModelSpec(
        key="bge-large-zh-v1.5-fp32",
        label="BAAI/bge-large-zh-v1.5 fp32",
        embed_model="BAAI/bge-large-zh-v1.5",
        precision="fp32",
        pooling="cls",
    ),
    ModelSpec(
        key="bge-large-zh-v1.5-fp16",
        label="BAAI/bge-large-zh-v1.5 fp16",
        embed_model="BAAI/bge-large-zh-v1.5",
        precision="fp16",
        pooling="cls",
    ),
    ModelSpec(
        key="bge-large-zh-v1.5-rerank-fp32",
        label="BAAI/bge-large-zh-v1.5 fp32 + BAAI/bge-reranker-large fp32",
        embed_model="BAAI/bge-large-zh-v1.5",
        precision="fp32",
        pooling="cls",
        mode="rerank",
        reranker_model="BAAI/bge-reranker-large",
        reranker_precision="fp32",
    ),
]

MODEL_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}
DEFAULT_MODEL_KEY = "m3e-small-fp32"
DEFAULT_RECALL_TOPK = 20


@dataclass
class ModelState:
    model_key: str
    recall_topk: int
    pending_rebuild: bool

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "recall_topk": int(self.recall_topk),
            "pending_rebuild": bool(self.pending_rebuild),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelState":
        model_key = str(data.get("model_key") or DEFAULT_MODEL_KEY)
        if model_key not in MODEL_BY_KEY:
            model_key = DEFAULT_MODEL_KEY
        recall_topk = int(data.get("recall_topk") or DEFAULT_RECALL_TOPK)
        pending_rebuild = bool(data.get("pending_rebuild", False))
        return cls(model_key=model_key, recall_topk=recall_topk, pending_rebuild=pending_rebuild)


def load_model_state() -> ModelState:
    path = CFG.MODEL_STATE_PATH
    if path.exists():
        try:
            data = read_json(path)
            if isinstance(data, dict):
                return ModelState.from_dict(data)
        except Exception:
            pass
    state = ModelState(
        model_key=DEFAULT_MODEL_KEY,
        recall_topk=DEFAULT_RECALL_TOPK,
        pending_rebuild=False,
    )
    save_model_state(state)
    return state


def save_model_state(state: ModelState) -> None:
    write_json(CFG.MODEL_STATE_PATH, state.to_dict())


def get_model_spec(model_key: str) -> ModelSpec:
    return MODEL_BY_KEY.get(model_key, MODEL_BY_KEY[DEFAULT_MODEL_KEY])
