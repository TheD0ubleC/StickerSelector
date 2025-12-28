from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Iterable, List, Any


def safe_filename(ext: str) -> str:
    ext = (ext or "").lower()
    if not ext.startswith("."):
        ext = "." + ext
    return f"{uuid.uuid4().hex}{ext}"


def parse_tags(text: str) -> List[str]:
    """
    支持：空格 / 换行 / 逗号 / 中文逗号 / 分号 / 中文分号 / 竖线 / 顿号 分隔
    """
    if not text:
        return []
    parts = re.split(r"[\s,，;；|、]+", str(text).strip())
    out: List[str] = []
    seen = set()
    for p in parts:
        p = str(p).strip()
        if not p:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def tags_to_text(tags: Iterable[str]) -> str:
    return " ".join([str(t).strip() for t in tags if t and str(t).strip()])


def read_json(path: Path):
    return json.loads(path.read_text("utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")


def tail_lines(path: Path, n: int = 400) -> str:
    if not path.exists():
        return ""
    data = path.read_text("utf-8", errors="ignore").splitlines()
    return "\n".join(data[-n:])


def safe_json_list(s: Any) -> List[str]:
    """
    DB 里 tags_json 可能是 None / "" / 非法 JSON / 非 list。
    这里一律安全地转成 list[str]。
    """
    if s is None:
        return []
    if isinstance(s, list):
        return [str(x) for x in s if str(x).strip()]
    if not isinstance(s, str):
        s = str(s)

    s = s.strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
    except Exception:
        return []
    return []
