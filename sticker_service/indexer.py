from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import sqlite3

from sticker_service import db
from sticker_service.utils import safe_json_list


@dataclass
class StickerMeta:
    id: int
    series_id: int
    series_name: str
    series_enabled: bool
    filename: str
    ext: str
    enabled: bool
    needs_tag: bool
    tags: list[str]


class StickerIndex:
    def __init__(self):
        self._metas: List[StickerMeta] = []
        self._embs: Optional[np.ndarray] = None  # [N, D]
        self._dim: int = 0

    def refresh(self, con: sqlite3.Connection) -> None:
        # 索引里默认排除 needs_tag 的（否则会把“表情包”默认embedding也算进去，污染结果）
        pairs = db.list_stickers(con, include_disabled=True, include_needs_tag=False)
        metas: List[StickerMeta] = []
        embs: List[np.ndarray] = []
        dim = 0

        for st, sr in pairs:
            v = db.blob_to_vec(st.emb_blob, st.emb_dim)
            if dim == 0:
                dim = v.shape[0]
            if v.shape[0] != dim:
                continue

            metas.append(StickerMeta(
                id=st.id,
                series_id=st.series_id,
                series_name=sr.name,
                series_enabled=sr.enabled,
                filename=st.filename,
                ext=st.ext,
                enabled=st.enabled,
                needs_tag=st.needs_tag,
                tags=safe_json_list(st.tags_json),
            ))
            embs.append(v)

        self._metas = metas
        self._embs = np.stack(embs, axis=0).astype(np.float32) if embs else None
        self._dim = dim

    def select(
        self,
        query_vec: np.ndarray,
        topk: int = 3,
        only_enabled: bool = True,
        series_name: Optional[str] = None
    ) -> List[Tuple[StickerMeta, float]]:
        if self._embs is None or not self._metas:
            return []

        q = query_vec.astype(np.float32)
        if q.ndim != 1:
            q = q.reshape(-1)
        scores = self._embs @ q  # [N]

        cand = []
        for i, m in enumerate(self._metas):
            if only_enabled:
                if not m.series_enabled:
                    continue
                if not m.enabled:
                    continue
            if series_name and m.series_name != series_name:
                continue
            cand.append((m, float(scores[i])))

        cand.sort(key=lambda x: x[1], reverse=True)
        return cand[:max(1, int(topk))]

    @property
    def dim(self) -> int:
        return self._dim
