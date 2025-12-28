from __future__ import annotations

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_LIBROSA"] = "1"
os.environ["TRANSFORMERS_NO_AUDIO"] = "1"

import ctypes
import io
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import HfApi, hf_hub_download
try:
    from huggingface_hub.file_download import repo_folder_name  # type: ignore
except Exception:
    repo_folder_name = None
from sticker_service.config import CFG
from sticker_service.utils import (
    safe_filename,
    parse_tags,
    tags_to_text,
    tail_lines,
    safe_json_list,
    read_json,
    write_json,
)
from sticker_service.embedding_models import TextEmbedder, TextReranker
from sticker_service.indexer import StickerIndex
from sticker_service.model_state import (
    DEFAULT_RECALL_TOPK,
    MODEL_SPECS,
    ModelState,
    get_model_spec,
    load_model_state,
    save_model_state,
)
from sticker_service import db


# ---------- logging ----------
logger = logging.getLogger("sticker")
logger.setLevel(logging.INFO)
CFG.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(str(CFG.LOG_PATH), encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

os.environ["HF_HUB_CACHE"] = str(CFG.MODEL_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CFG.MODEL_DIR)


# ---------- app ----------
app = FastAPI(title="Sticker Selector Service")
templates = Jinja2Templates(directory=str(CFG.ROOT / "templates"))

# static
app.mount("/static", StaticFiles(directory=str(CFG.ROOT / "static")), name="static")
app.mount("/stickers", StaticFiles(directory=str(CFG.STICKER_DIR)), name="stickers")


# ---------- globals ----------
CON = db.connect(CFG.DB_PATH)
DB_LOCK = threading.Lock()
db.init_db(CON)

INDEX = StickerIndex()
EMBED: Optional[TextEmbedder] = None
RERANK: Optional[TextReranker] = None

MODEL_STATE = load_model_state()
MODEL_LOCK = threading.Lock()
MODEL_KEYS = {spec.key for spec in MODEL_SPECS}

DOWNLOAD_LOCK = threading.Lock()
DOWNLOAD_STATE: dict[str, Any] = {
    "status": "idle",
    "model_key": "",
    "progress": 0.0,
    "message": "",
    "bytes_total": 0,
    "bytes_done": 0,
    "speed_mbps": 0.0,
    "eta_seconds": 0,
    "done_files": 0,
    "total_files": 0,
    "current_file": "",
    "current_file_bytes": 0,
    "current_file_total": 0,
    "pid": 0,
    "apply_switch": False,
    "updated_at": 0.0,
}

EMBEDDING_LOCK = threading.Lock()
EMBEDDING_STATE: dict[str, Any] = {
    "status": "idle",
    "total": 0,
    "done": 0,
    "progress": 0.0,
    "message": "",
    "updated_at": 0.0,
}

DEFAULT_TAG_TEXT = "表情包"


# ---------- middleware ----------
@app.middleware("http")
async def embedding_gate(request: Request, call_next):
    if request.method.upper() == "GET":
        path = request.url.path
        if not _any_model_downloaded():
            if not (
                path.startswith("/api")
                or path.startswith("/static")
                or path.startswith("/stickers")
                or path == "/progress"
                or path == "/init"
            ):
                return RedirectResponse("/init", status_code=302)
        if not (
            path.startswith("/api")
            or path.startswith("/static")
            or path.startswith("/stickers")
            or path == "/progress"
        ):
            dl = _get_download_state()
            dl_status = str(dl.get("status") or "")
            if dl_status in {"downloading", "ready"}:
                return RedirectResponse("/progress", status_code=302)
            state = _get_embedding_state()
            pending = _current_model_state().pending_rebuild
            if pending or state.get("status") == "running":
                return RedirectResponse("/progress", status_code=302)
    return await call_next(request)


# ---------- DB compat helpers ----------
def db_list_stickers_compat(
    series_id: Optional[int],
    include_disabled: bool,
    include_needs_tag: bool,
):
    """
    兼容旧版 db.list_stickers() 签名（有些版本没有 include_needs_tag 参数）
    """
    try:
        return db.list_stickers(CON, series_id=series_id, include_disabled=include_disabled, include_needs_tag=include_needs_tag)
    except TypeError:
        # fallback: old signature
        return db.list_stickers(CON, series_id=series_id, include_disabled=include_disabled)


def db_count_pending_compat(batch_id: Optional[int] = None) -> int:
    """
    兼容旧版 count_pending()（有些版本不支持 batch_id）
    """
    try:
        return db.count_pending(CON, batch_id=batch_id)
    except TypeError:
        return db.count_pending(CON)


def _set_download_state(**kwargs: Any) -> None:
    state = dict(DOWNLOAD_STATE)
    state.update(kwargs)
    state["updated_at"] = time.time()
    path = CFG.DOWNLOAD_STATE_PATH
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(path)
    except Exception:
        logger.exception("Failed to write download state")
    with DOWNLOAD_LOCK:
        DOWNLOAD_STATE.update(state)


def _get_download_state() -> dict[str, Any]:
    path = CFG.DOWNLOAD_STATE_PATH
    if path.exists():
        for _ in range(2):
            try:
                data = read_json(path)
                if isinstance(data, dict):
                    return data
            except Exception:
                time.sleep(0.02)
    with DOWNLOAD_LOCK:
        return dict(DOWNLOAD_STATE)


def _reset_download_state_on_startup() -> None:
    dl_state = _get_download_state()
    status = str(dl_state.get("status") or "")
    model_key = str(dl_state.get("model_key") or "")
    if status == "downloading":
        pid = int(dl_state.get("pid") or 0)
        if pid and not _pid_alive(pid):
            _set_download_state(
                status="error",
                model_key=str(dl_state.get("model_key") or ""),
                progress=float(dl_state.get("progress") or 0),
                message="download worker stopped",
            )
    if status == "ready":
        _set_download_state(
            status="idle",
            model_key=str(dl_state.get("model_key") or ""),
            progress=100.0,
            message="downloaded",
        )
    if model_key and status in {"done", "ready"}:
        spec = get_model_spec(model_key)
        downloaded, _ = _model_download_status(spec)
        if not downloaded:
            _set_download_state(
                status="idle",
                model_key="",
                progress=0.0,
                message="cache missing",
                bytes_total=0,
                bytes_done=0,
                speed_mbps=0.0,
                eta_seconds=0,
                done_files=0,
                total_files=0,
                current_file="",
                current_file_bytes=0,
                current_file_total=0,
                pid=0,
                apply_switch=False,
            )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return psutil.pid_exists(pid)
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            handle = ctypes.windll.kernel32.OpenProcess(1, 0, int(pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
        except Exception:
            return False
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _terminate_pid(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        return True
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            subprocess.check_call(["taskkill", "/PID", str(pid), "/T", "/F"])
            return True
        except Exception:
            return False
    try:
        import signal

        os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def _set_embedding_state(**kwargs: Any) -> None:
    with EMBEDDING_LOCK:
        EMBEDDING_STATE.update(kwargs)
        EMBEDDING_STATE["updated_at"] = time.time()


def _get_embedding_state() -> dict[str, Any]:
    with EMBEDDING_LOCK:
        return dict(EMBEDDING_STATE)


def _embedding_path(sticker_id: int) -> Path:
    return CFG.EMBEDDING_DIR / f"{int(sticker_id)}.npy"


def save_embedding_local(sticker_id: int, emb: Optional[np.ndarray]) -> None:
    if emb is None:
        return
    path = _embedding_path(sticker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, np.asarray(emb, dtype=np.float32))
        tmp.replace(path)
    except Exception:
        logger.exception("Failed to save embedding: %s", path)


def _current_model_state() -> ModelState:
    with MODEL_LOCK:
        return ModelState(
            model_key=MODEL_STATE.model_key,
            recall_topk=MODEL_STATE.recall_topk,
            pending_rebuild=MODEL_STATE.pending_rebuild,
        )


def _load_persisted_model_state() -> ModelState:
    path = CFG.MODEL_STATE_PATH
    if path.exists():
        try:
            data = read_json(path)
            if isinstance(data, dict):
                return ModelState.from_dict(data)
        except Exception:
            pass
    return _current_model_state()


def _update_model_state(new_state: ModelState, persist: bool = True) -> None:
    with MODEL_LOCK:
        MODEL_STATE.model_key = new_state.model_key
        MODEL_STATE.recall_topk = new_state.recall_topk
        MODEL_STATE.pending_rebuild = new_state.pending_rebuild
    if persist:
        save_model_state(new_state)


def _any_model_downloaded() -> bool:
    try:
        for spec in MODEL_SPECS:
            try:
                downloaded, _ = _model_download_status(spec)
                if downloaded:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    try:
        for item in CFG.MODEL_DIR.glob("models--*"):
            if (item / "snapshots").exists():
                return True
    except Exception:
        pass
    return False


def ensure_models() -> bool:
    global EMBED
    global RERANK
    if not _any_model_downloaded():
        logger.info("No downloaded models found, skip loading")
        EMBED = None
        RERANK = None
        return False
    state = _current_model_state()
    spec = get_model_spec(state.model_key)

    if EMBED is None or EMBED.model_name != spec.embed_model or EMBED.precision != spec.precision or EMBED.pooling != spec.pooling:
        logger.info("Loading embed model: %s (%s)", spec.embed_model, spec.precision)
        EMBED = TextEmbedder(spec.embed_model, precision=spec.precision, pooling=spec.pooling)
        logger.info("Embed model loaded on device=%s dtype=%s", EMBED.device, EMBED.dtype)

    if spec.mode == "rerank" and spec.reranker_model:
        if RERANK is None or RERANK.model_name != spec.reranker_model or RERANK.precision != (spec.reranker_precision or "fp32"):
            logger.info("Loading reranker model: %s (%s)", spec.reranker_model, spec.reranker_precision or "fp32")
            RERANK = TextReranker(spec.reranker_model, precision=spec.reranker_precision or "fp32")
            logger.info("Reranker loaded on device=%s dtype=%s", RERANK.device, RERANK.dtype)
    else:
        RERANK = None
    return True


def refresh_index() -> None:
    if not ensure_models():
        return
    with DB_LOCK:
        INDEX.refresh(CON)
    logger.info("Index refreshed: N=%s dim=%s", len(getattr(INDEX, "_metas", [])), INDEX.dim)


def _run_embedding_rebuild() -> None:
    logger.info("Embedding rebuild started")
    _set_embedding_state(status="running", total=0, done=0, progress=0.0, message="embedding")
    if not ensure_models():
        _set_embedding_state(status="idle", total=0, done=0, progress=0.0, message="no model")
        return
    assert EMBED is not None

    with DB_LOCK:
        pairs = db.list_stickers(CON, include_disabled=True, include_needs_tag=True)

    total = len(pairs)
    _set_embedding_state(total=total, done=0, progress=0.0, message=f"embedding 0/{total}")

    done = 0
    for st, _ in pairs:
        tags = safe_json_list(getattr(st, "tags_json", "[]"))
        text = tags_to_text(tags) if tags else DEFAULT_TAG_TEXT
        emb = EMBED.encode_one(text)
        with DB_LOCK:
            db.update_sticker_emb(CON, int(st.id), emb)
        save_embedding_local(int(st.id), emb)

        done += 1
        progress = (done / total * 100.0) if total > 0 else 100.0
        _set_embedding_state(done=done, progress=progress, message=f"embedding {done}/{total}")

    refresh_index()
    new_state = _current_model_state()
    new_state.pending_rebuild = False
    _update_model_state(new_state, persist=True)
    _set_embedding_state(status="done", progress=100.0, message="embedding done")
    logger.info("Embedding rebuild finished")


def _start_embedding_rebuild() -> None:
    state = _get_embedding_state()
    if state.get("status") == "running":
        return
    t = threading.Thread(target=_run_embedding_rebuild, daemon=True)
    t.start()


def _normalize_endpoint(endpoint: str) -> str:
    cleaned = str(endpoint or "").strip()
    if not cleaned:
        return "https://huggingface.co"
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    return f"https://{cleaned}"


def _load_hf_endpoint() -> str:
    path = CFG.HF_ENDPOINT_PATH
    if path.exists():
        try:
            data = read_json(path)
            if isinstance(data, dict):
                return _normalize_endpoint(str(data.get("endpoint") or ""))
        except Exception:
            pass
    return "https://huggingface.co"


def _save_hf_endpoint(endpoint: str) -> str:
    value = _normalize_endpoint(endpoint)
    write_json(CFG.HF_ENDPOINT_PATH, {"endpoint": value})
    os.environ["HF_ENDPOINT"] = value
    return value


def _get_hf_endpoint() -> str:
    value = _load_hf_endpoint()
    os.environ["HF_ENDPOINT"] = value
    return value


_FILE_LIST_CACHE: dict[str, tuple[float, list[tuple[str, str, int]]]] = {}


def _should_skip_repo_file(name: str) -> bool:
    lname = name.lower()
    return lname.endswith((".onnx", ".onnx_data", ".pb", ".h5"))


def _get_repo_files(repo_id: str) -> list[tuple[str, str, int]]:
    now = time.time()
    cached = _FILE_LIST_CACHE.get(repo_id)
    if cached and (now - cached[0]) < 60:
        return cached[1]
    api = HfApi(endpoint=_get_hf_endpoint())
    info = api.model_info(repo_id=repo_id)
    files: list[tuple[str, str, int]] = []
    for sibling in getattr(info, "siblings", []) or []:
        name = getattr(sibling, "rfilename", None)
        if not name:
            continue
        if _should_skip_repo_file(str(name)):
            continue
        size = int(getattr(sibling, "size", 0) or 0)
        files.append((repo_id, name, size))
    _FILE_LIST_CACHE[repo_id] = (now, files)
    return files


def _build_download_list(repo_id: str) -> list[tuple[str, str, int]]:
    return _get_repo_files(repo_id)


def _download_models_for_spec(spec, recall_topk: int) -> None:
    repos = [spec.embed_model]
    if spec.mode == "rerank" and spec.reranker_model:
        repos.append(spec.reranker_model)

    files: list[tuple[str, str, int]] = []
    for repo_id in repos:
        files.extend(_build_download_list(repo_id))

    total_bytes = sum(size for _, _, size in files if size > 0)
    total_files = len(files)
    done_files = 0
    done_bytes = 0

    _set_download_state(
        status="downloading",
        model_key=spec.key,
        progress=0.0,
        message="downloading",
        bytes_total=0,
        bytes_done=0,
        speed_mbps=0.0,
        eta_seconds=0,
        done_files=0,
        total_files=0,
        current_file="",
        current_file_bytes=0,
        current_file_total=0,
    )

    for repo_id, filename, size in files:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(CFG.MODEL_DIR),
            endpoint=_get_hf_endpoint(),
        )
        done_files += 1
        done_bytes += size
        if total_bytes > 0:
            progress = min(100.0, done_bytes / total_bytes * 100.0)
        else:
            progress = min(100.0, done_files / max(1, total_files) * 100.0)
        _set_download_state(
            status="downloading",
            model_key=spec.key,
            progress=progress,
            message=f"downloading {done_files}/{total_files}",
            bytes_total=total_bytes,
            bytes_done=done_bytes,
            speed_mbps=0.0,
            eta_seconds=0,
            done_files=done_files,
            total_files=total_files,
            current_file=filename,
            current_file_bytes=size,
            current_file_total=size,
        )

    next_state = ModelState(
        model_key=spec.key,
        recall_topk=recall_topk,
        pending_rebuild=True,
    )
    save_model_state(next_state)
    _set_download_state(
        status="ready",
        model_key=spec.key,
        progress=100.0,
        message="ready",
        bytes_total=total_bytes,
        bytes_done=done_bytes,
        speed_mbps=0.0,
        eta_seconds=0,
        done_files=done_files,
        total_files=total_files,
        current_file="",
        current_file_bytes=0,
        current_file_total=0,
    )


def _hf_cache_root() -> Path:
    return CFG.MODEL_DIR


def _repo_cache_path(repo_id: str) -> Path:
    root = _hf_cache_root()
    if repo_folder_name is not None:
        try:
            return root / repo_folder_name(repo_id=repo_id, repo_type="model")
        except TypeError:
            try:
                return root / repo_folder_name(repo_id, repo_type="model")
            except Exception:
                pass
        except Exception:
            pass
    safe = repo_id.replace("/", "--")
    return root / f"models--{safe}"


def _resolve_cache_path(path: Path) -> Path:
    try:
        return path.resolve()
    except Exception:
        return path


def _has_incomplete_marker(path: Path) -> bool:
    for candidate in (path, _resolve_cache_path(path)):
        try:
            incomplete = Path(str(candidate) + ".incomplete")
            if incomplete.exists():
                return True
        except Exception:
            continue
    return False


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(200)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def _looks_like_zero_placeholder(path: Path, expected_size: int) -> bool:
    if expected_size <= 0:
        return False
    try:
        size = path.stat().st_size
    except Exception:
        return False
    if size <= 0:
        return True
    if size < 1024 * 1024:
        return False
    try:
        with path.open("rb") as f:
            head = f.read(4096)
            if head and any(b != 0 for b in head):
                return False
            f.seek(max(0, size - 4096))
            tail = f.read(4096)
            if tail and any(b != 0 for b in tail):
                return False
        return True
    except Exception:
        return False


def _is_valid_cached_file(path: Path, expected_size: int) -> bool:
    try:
        if not path.exists() or not path.is_file():
            return False
    except Exception:
        return False
    if _has_incomplete_marker(path):
        return False
    try:
        size = path.stat().st_size
    except Exception:
        return False
    if size <= 0:
        return False
    if expected_size > 0 and size != expected_size:
        return False
    if _looks_like_lfs_pointer(path):
        return False
    if _looks_like_zero_placeholder(path, expected_size):
        return False
    return True


def _is_repo_downloaded(repo_id: str) -> bool:
    files = _get_repo_files(repo_id)
    if not files:
        return False
    for _, filename, size in files:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(CFG.MODEL_DIR),
                local_files_only=True,
                endpoint=_get_hf_endpoint(),
            )
        except Exception:
            return False
        if not _is_valid_cached_file(Path(path), int(size or 0)):
            return False
    return True


def _model_download_status(spec) -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not _is_repo_downloaded(spec.embed_model):
        missing.append(spec.embed_model)
    if spec.mode == "rerank" and spec.reranker_model:
        if not _is_repo_downloaded(spec.reranker_model):
            missing.append(spec.reranker_model)
    return (len(missing) == 0), missing


def _clear_model_cache(spec) -> None:
    repos = [spec.embed_model]
    if spec.mode == "rerank" and spec.reranker_model:
        repos.append(spec.reranker_model)
    for repo_id in repos:
        path = _repo_cache_path(repo_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def _start_download_process(model_key: str, recall_topk: int, apply_switch: bool) -> None:
    _set_download_state(
        status="downloading",
        model_key=model_key,
        progress=0.0,
        message="starting",
        bytes_total=0,
        bytes_done=0,
        speed_mbps=0.0,
        eta_seconds=0,
        done_files=0,
        total_files=0,
        current_file="",
        current_file_bytes=0,
        current_file_total=0,
        pid=0,
        apply_switch=bool(apply_switch),
    )
    args = [
        sys.executable,
        "-m",
        "sticker_service.download_worker",
        "--model-key",
        model_key,
        "--recall-topk",
        str(int(recall_topk)),
        "--apply" if apply_switch else "--download-only",
        "--state-path",
        str(CFG.DOWNLOAD_STATE_PATH),
    ]
    try:
        proc = subprocess.Popen(args, cwd=str(CFG.ROOT.parent), env=os.environ.copy())
        _set_download_state(pid=int(proc.pid or 0))
    except Exception:
        _set_download_state(
            status="error",
            message="failed to start download worker",
            bytes_total=0,
            bytes_done=0,
            speed_mbps=0.0,
            eta_seconds=0,
            done_files=0,
            total_files=0,
            current_file="",
            current_file_bytes=0,
            current_file_total=0,
        )

@app.on_event("startup")
def on_startup() -> None:
    _get_hf_endpoint()
    _reset_download_state_on_startup()
    has_models = _any_model_downloaded()
    if has_models:
        ensure_models()
    with DB_LOCK:
        if not db.list_series(CON):
            db.create_series(CON, "default")
            logger.info("Created default series")
    state = _current_model_state()
    if state.pending_rebuild and has_models:
        _start_embedding_rebuild()
    elif has_models:
        refresh_index()
        _set_embedding_state(status="done", progress=100.0, message="ready")
    else:
        _set_embedding_state(status="idle", progress=0.0, message="no model")


# ---------- match helpers ----------
def softmax_fit(scores: np.ndarray, scale: float) -> np.ndarray:
    if scores.size == 0:
        return scores
    z = scores.astype(np.float32) * float(scale)
    z = z - np.max(z)
    e = np.exp(z)
    s = np.sum(e)
    if s <= 0:
        return np.zeros_like(scores, dtype=np.float32)
    return (e / s).astype(np.float32)


def score_to_match_rate(raw_score: float) -> float:
    """
    将语义相似度（cosine, [-1, 1]）映射为 0~100。
    """
    try:
        val = float(raw_score)
    except Exception:
        return 0.0
    val = max(0.0, min(1.0, val))
    return val * 100.0


def safe_export_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "series"


# ---------- core selection ----------
def select_stickers(
    tags_list: List[str],
    k: int,
    series: Optional[str],
    order: str,
) -> List[Dict[str, Any]]:
    if not ensure_models():
        return []
    assert EMBED is not None
    state = _current_model_state()
    spec = get_model_spec(state.model_key)

    k = max(1, min(int(k), CFG.MAX_TOPK))
    q_text = tags_to_text(tags_list)
    q = EMBED.encode_one(q_text)  # normalized
    recall_k = k
    if spec.mode == "rerank":
        recall_k = max(k, int(state.recall_topk or DEFAULT_RECALL_TOPK))
    with DB_LOCK:
        top = INDEX.select(q, topk=max(recall_k, 1), only_enabled=True, series_name=series)

    if not top:
        return []

    if spec.mode == "rerank" and RERANK is not None:
        docs = [tags_to_text(m.tags) if m.tags else DEFAULT_TAG_TEXT for (m, _) in top]
        rerank_scores = RERANK.score(q_text, docs)
        combined = []
        for (m, emb_score), rr_score in zip(top, rerank_scores.tolist()):
            combined.append((m, float(rr_score), float(emb_score)))
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:k]

        raw_scores = np.array([s for (_, s, _) in combined], dtype=np.float32)
        fit_probs = softmax_fit(raw_scores, scale=CFG.FIT_SOFTMAX_SCALE)
        out = []
        for (m, rr, emb_score), p in zip(combined, fit_probs.tolist()):
            fit_rate = float(p * 100.0)
            match_rate = score_to_match_rate(emb_score)
            out.append({
                "id": m.id,
                "series": m.series_name,
                "url": f"/stickers/{m.series_name}/{m.filename}",
                "raw": float(rr),
                "embed_raw": float(emb_score),
                "fit_rate": fit_rate,
                "match_rate": match_rate,
                "tags": m.tags,
            })
    else:
        raw_scores = np.array([s for (_, s) in top], dtype=np.float32)
        fit_probs = softmax_fit(raw_scores, scale=CFG.FIT_SOFTMAX_SCALE)
        out = []
        for (m, raw), p in zip(top, fit_probs.tolist()):
            fit_rate = float(p * 100.0)
            match_rate = score_to_match_rate(raw)
            out.append({
                "id": m.id,
                "series": m.series_name,
                "url": f"/stickers/{m.series_name}/{m.filename}",
                "raw": float(raw),
                "fit_rate": fit_rate,
                "match_rate": match_rate,
                "tags": m.tags,
            })

    order = (order or "none").lower()
    if order == "desc":
        out.sort(key=lambda x: (x.get("match_rate", 0.0), x.get("raw", 0.0)), reverse=True)
    elif order == "asc":
        out.sort(key=lambda x: (x.get("match_rate", 0.0), x.get("raw", 0.0)))
    return out


# ---------- API ----------
@app.post("/api/select")
def api_select(payload: Dict[str, Any] = Body(...)):
    tags = payload.get("tags", "")
    k = payload.get("k", CFG.DEFAULT_TOPK)
    series = payload.get("series")
    order = payload.get("order", "desc")

    if isinstance(tags, list):
        tags_list = []
        seen = set()
        for t in tags:
            for part in parse_tags(str(t)):
                if part not in seen:
                    seen.add(part)
                    tags_list.append(part)
    else:
        tags_list = parse_tags(str(tags or ""))

    if not tags_list:
        return {"items": []}

    items = select_stickers(tags_list, k, series, order)
    logger.info("api_select tags=%s k=%s series=%s order=%s -> %s", tags_list, k, series, order, len(items))
    state = _current_model_state()
    spec = get_model_spec(state.model_key)
    match_mode = "embed_cosine"
    meta: Dict[str, Any] = {
        "k": int(k),
        "series": series or "",
        "order": order,
        "match_mode": match_mode,
        "fit_mode": "softmax_topk",
        "fit_scale": float(CFG.FIT_SOFTMAX_SCALE),
        "model_key": spec.key,
    }
    if spec.mode == "rerank":
        meta["match_mode"] = "embed_cosine_rerank"
        meta["recall_topk"] = int(state.recall_topk or DEFAULT_RECALL_TOPK)
        meta["reranker_model"] = spec.reranker_model or ""
    return {
        "items": items,
        "meta": meta,
    }


@app.get("/api/series")
def api_series():
    with DB_LOCK:
        rows = db.list_series(CON)
    return {"items": [{"id": r.id, "name": r.name, "enabled": r.enabled} for r in rows]}


@app.post("/api/series/exists")
def api_series_exists(payload: Dict[str, Any] = Body(...)):
    name = str(payload.get("name", "")).strip()
    if not name:
        return {"exists": False, "name": ""}
    with DB_LOCK:
        sr = db.get_series_by_name(CON, name)
    return {"exists": bool(sr), "id": sr.id if sr else None, "name": name}


@app.post("/api/series/count")
def api_series_count(payload: Dict[str, Any] = Body(...)):
    series_id = payload.get("series_id")
    name = payload.get("name")

    with DB_LOCK:
        rows = db.list_series(CON)

    if series_id is not None or name:
        target = None
        if series_id is not None:
            try:
                sid = int(series_id)
            except Exception:
                sid = 0
            target = next((r for r in rows if r.id == sid), None)
        elif name:
            target = next((r for r in rows if r.name == str(name).strip()), None)

        if not target:
            raise HTTPException(404, "series not found")

        with DB_LOCK:
            count = db.count_stickers_in_series(CON, target.id)
        return {"id": target.id, "name": target.name, "count": int(count)}

    with DB_LOCK:
        counts = db.list_series_counts(CON)
    items = [{"id": r.id, "name": r.name, "count": int(counts.get(r.id, 0))} for r in rows]
    return {"items": items, "meta": {"total_series": len(items)}}


@app.get("/api/stats")
def api_stats():
    pending = db_count_pending_compat(None)
    return {"pending": pending}


def _get_ram_info() -> tuple[int, int]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass

    if sys.platform.startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys), int(stat.ullAvailPhys)

    if sys.platform.startswith("linux"):
        total = 0
        available = 0
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) * 1024
        except Exception:
            pass
        return total, available

    return 0, 0


def _get_cpu_max_mhz() -> int:
    try:
        import psutil  # type: ignore

        freq = psutil.cpu_freq()
        if freq and freq.max:
            return int(freq.max)
    except Exception:
        pass

    if sys.platform.startswith("win"):
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "MaxClockSpeed"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            nums = [int(x) for x in re.findall(r"\d+", out)]
            if nums:
                return max(nums)
        except Exception:
            pass
    return 0


def _device_info() -> dict[str, Any]:
    cuda_version = torch.version.cuda or ""
    cuda_build = bool(cuda_version)
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else ""
    device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_vram_gb = 0.0
    gpu_clock_mhz = 0
    gpu_mem_clock_mhz = 0
    if cuda_available:
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_vram_gb = round(float(props.total_memory) / (1024**3), 2)
            gpu_clock_mhz = int(getattr(props, "clock_rate", 0) / 1000)
            gpu_mem_clock_mhz = int(getattr(props, "memory_clock_rate", 0) / 1000)
        except Exception:
            pass

    total_ram, avail_ram = _get_ram_info()
    cpu_cores = os.cpu_count() or 0
    cpu_max_mhz = _get_cpu_max_mhz()
    return {
        "torch_version": torch.__version__,
        "cuda_build": "cuda" if cuda_build else "cpu",
        "cuda_version": cuda_version,
        "cuda_available": bool(cuda_available),
        "cuda_device": device_name,
        "cuda_device_count": int(device_count),
        "cuda_vram_gb": gpu_vram_gb,
        "cuda_clock_mhz": int(gpu_clock_mhz),
        "cuda_mem_clock_mhz": int(gpu_mem_clock_mhz),
        "ram_total_gb": round(total_ram / (1024**3), 2) if total_ram else 0.0,
        "ram_available_gb": round(avail_ram / (1024**3), 2) if avail_ram else 0.0,
        "cpu_cores": int(cpu_cores),
        "cpu_max_mhz": int(cpu_max_mhz),
        "platform": platform.system(),
    }


def _recommend_model(device: dict[str, Any]) -> dict[str, Any]:
    cores = int(device.get("cpu_cores") or 0)
    ram_gb = float(device.get("ram_total_gb") or 0.0)
    cpu_mhz = int(device.get("cpu_max_mhz") or 0)
    cuda_available = bool(device.get("cuda_available"))
    vram_gb = float(device.get("cuda_vram_gb") or 0.0)

    cpu_fast = cpu_mhz == 0 or cpu_mhz >= 2200

    # ===== 默认兜底 =====
    level = "L0"
    confidence = "high"
    key = "m3e-small-fp16"
    reason = "极限低配环境（1GB 内存级别），仅保证模型可运行。"
    recall_topk = DEFAULT_RECALL_TOPK

    # ===== L4：高配 + GPU（专家模式）=====
    if (
        ram_gb >= 12
        and cores >= 8
        and cuda_available
        and vram_gb >= 6
    ):
        level = "L4"
        confidence = "low"
        key = "bge-large-zh-v1.5-fp32+rerank-fp32"
        reason = (
            f"检测到高配设备（{cores} 核 / {ram_gb:.1f}GB，GPU {vram_gb:.1f}GB），"
            "可启用 Rerank 获得极限准确率，仅推荐给开发者或高端玩家。"
        )

    # ===== L2：高精度主力（无 GPU 也可）=====
    elif ram_gb >= 6 and cores >= 6 and cpu_fast:
        level = "L2"
        confidence = "medium"
        key = (
            "bge-large-zh-v1.5-fp32"
            if ram_gb >= 8
            else "bge-large-zh-v1.5-fp16"
        )
        reason = (
            f"{cores} 核 / {ram_gb:.1f}GB，主流 PC 或开发者电脑，"
            "偏向准确率，CPU 推理延迟较高。"
        )

    # ===== L1：低成本云 / 性价比 =====
    elif ram_gb >= 2:
        level = "L1"
        confidence = "high"
        key = "bge-small-zh-v1.5-fp32"
        reason = (
            f"{cores} 核 / {ram_gb:.1f}GB，低成本云服务器或轻量环境，"
            "性价比极高，准确率明显优于 m3e-small。"
        )

    spec = get_model_spec(key)
    return {
        "level": level,
        "confidence": confidence,
        "model_key": spec.key,
        "label": spec.label,
        "reason": reason,
        "recall_topk": int(recall_topk),
    }




@app.get("/api/model/status")
def api_model_status():
    state = _current_model_state()
    spec = get_model_spec(state.model_key)
    device = _device_info()
    models = []
    for s in MODEL_SPECS:
        try:
            downloaded, missing = _model_download_status(s)
        except Exception:
            logger.exception("Failed to check model cache: %s", s.key)
            downloaded, missing = False, [s.embed_model]
        models.append(
            {
                "key": s.key,
                "label": s.label,
                "mode": s.mode,
                "embed_model": s.embed_model,
                "precision": s.precision,
                "pooling": s.pooling,
                "reranker_model": s.reranker_model or "",
                "downloaded": downloaded,
                "missing": missing,
            }
        )
    return {
        "current": {
            "model_key": spec.key,
            "label": spec.label,
            "mode": spec.mode,
            "embed_model": spec.embed_model,
            "precision": spec.precision,
            "reranker_model": spec.reranker_model or "",
            "recall_topk": int(state.recall_topk or DEFAULT_RECALL_TOPK),
            "pending_rebuild": bool(state.pending_rebuild),
        },
        "download": _get_download_state(),
        "embedding": _get_embedding_state(),
        "device": device,
        "hf_endpoint": _get_hf_endpoint(),
        "recommendation": _recommend_model(device),
        "models": models,
    }


@app.get("/api/model/hf-endpoint")
def api_model_hf_endpoint():
    return {"endpoint": _get_hf_endpoint()}


@app.post("/api/model/hf-endpoint")
def api_model_hf_endpoint_update(payload: Dict[str, Any] = Body(...)):
    endpoint = str(payload.get("endpoint") or "").strip()
    if not endpoint:
        raise HTTPException(400, "endpoint required")
    value = _save_hf_endpoint(endpoint)
    return {"ok": True, "endpoint": value}


@app.get("/api/progress/status")
def api_progress_status():
    state = _current_model_state()
    return {
        "download": _get_download_state(),
        "embedding": _get_embedding_state(),
        "pending_rebuild": bool(state.pending_rebuild),
        "current_model_key": state.model_key,
    }


@app.post("/api/model/switch")
def api_model_switch(payload: Dict[str, Any] = Body(...)):
    model_key = str(payload.get("model_key") or "").strip()
    recall_topk = payload.get("recall_topk")
    if model_key not in MODEL_KEYS:
        raise HTTPException(400, "model_key invalid")

    try:
        recall_topk_val = int(recall_topk) if recall_topk is not None else DEFAULT_RECALL_TOPK
    except Exception:
        recall_topk_val = DEFAULT_RECALL_TOPK
    recall_topk_val = max(1, recall_topk_val)

    state = _current_model_state()
    if model_key == state.model_key:
        if state.recall_topk != recall_topk_val:
            state.recall_topk = recall_topk_val
            _update_model_state(state, persist=True)
        return {"ok": True, "status": "ready", "restart": False, "model_key": model_key}

    dl_state = _get_download_state()
    if str(dl_state.get("status") or "") in {"downloading", "ready"}:
        return {"ok": False, "status": str(dl_state.get("status")), "message": "download busy"}

    spec = get_model_spec(model_key)
    downloaded, _ = _model_download_status(spec)
    if downloaded:
        new_state = ModelState(
            model_key=spec.key,
            recall_topk=recall_topk_val,
            pending_rebuild=True,
        )
        save_model_state(new_state)
        _set_download_state(
            status="ready",
            model_key=spec.key,
            progress=100.0,
            message="ready",
            apply_switch=True,
            bytes_total=0,
            bytes_done=0,
            speed_mbps=0.0,
            eta_seconds=0,
            done_files=0,
            total_files=0,
            current_file="",
            current_file_bytes=0,
            current_file_total=0,
            pid=0,
        )
        return {"ok": True, "status": "ready", "model_key": model_key, "restart": False}

    _start_download_process(spec.key, recall_topk_val, apply_switch=True)
    return {"ok": True, "status": "downloading", "model_key": model_key}


@app.post("/api/model/switch/apply")
def api_model_switch_apply():
    dl_state = _get_download_state()
    status = str(dl_state.get("status") or "")
    if status != "ready":
        return {"ok": False, "status": status, "message": "switch not ready"}
    if not bool(dl_state.get("apply_switch", False)):
        return {"ok": False, "status": status, "message": "no switch pending"}

    model_key = str(dl_state.get("model_key") or "")
    if model_key not in MODEL_KEYS:
        raise HTTPException(400, "model_key invalid")

    spec = get_model_spec(model_key)
    downloaded, missing = _model_download_status(spec)
    if not downloaded:
        return {"ok": False, "status": "missing", "model_key": model_key, "missing": missing}

    target_state = _load_persisted_model_state()
    recall_topk_val = max(1, int(target_state.recall_topk or DEFAULT_RECALL_TOPK))
    if target_state.model_key != spec.key:
        target_state = ModelState(
            model_key=spec.key,
            recall_topk=recall_topk_val,
            pending_rebuild=True,
        )
    else:
        target_state.recall_topk = recall_topk_val
        target_state.pending_rebuild = True
    _update_model_state(target_state, persist=True)

    _set_download_state(
        status="idle",
        model_key=model_key,
        progress=100.0,
        message="switching",
        bytes_total=0,
        bytes_done=0,
        speed_mbps=0.0,
        eta_seconds=0,
        done_files=0,
        total_files=0,
        current_file="",
        current_file_bytes=0,
        current_file_total=0,
        pid=0,
        apply_switch=False,
    )
    _start_embedding_rebuild()
    return {"ok": True, "status": "switching", "model_key": model_key}


@app.post("/api/model/switch/defer")
def api_model_switch_defer():
    dl_state = _get_download_state()
    status = str(dl_state.get("status") or "")
    if status != "ready":
        return {"ok": False, "status": status, "message": "switch not ready"}
    if not bool(dl_state.get("apply_switch", False)):
        return {"ok": False, "status": status, "message": "no switch pending"}

    model_key = str(dl_state.get("model_key") or "")
    _set_download_state(
        status="idle",
        model_key=model_key,
        progress=100.0,
        message="switch deferred",
        bytes_total=0,
        bytes_done=0,
        speed_mbps=0.0,
        eta_seconds=0,
        done_files=0,
        total_files=0,
        current_file="",
        current_file_bytes=0,
        current_file_total=0,
        pid=0,
        apply_switch=False,
    )
    return {"ok": True, "status": "deferred", "model_key": model_key}


@app.post("/api/model/download")
def api_model_download(payload: Dict[str, Any] = Body(...)):
    model_key = str(payload.get("model_key") or "").strip()
    if model_key not in MODEL_KEYS:
        raise HTTPException(400, "model_key invalid")

    dl_state = _get_download_state()
    if str(dl_state.get("status") or "") in {"downloading", "ready"}:
        return {"ok": False, "status": str(dl_state.get("status")), "message": "download busy"}

    spec = get_model_spec(model_key)
    downloaded, _ = _model_download_status(spec)
    if downloaded:
        return {"ok": True, "status": "downloaded", "model_key": model_key}

    _start_download_process(spec.key, DEFAULT_RECALL_TOPK, apply_switch=False)
    return {"ok": True, "status": "downloading", "model_key": model_key}


@app.post("/api/model/delete")
def api_model_delete(payload: Dict[str, Any] = Body(...)):
    model_key = str(payload.get("model_key") or "").strip()
    if model_key not in MODEL_KEYS:
        raise HTTPException(400, "model_key invalid")

    dl_state = _get_download_state()
    if str(dl_state.get("status") or "") in {"downloading", "ready"}:
        return {"ok": False, "status": str(dl_state.get("status")), "message": "download busy"}

    state = _current_model_state()
    if model_key == state.model_key:
        return {"ok": False, "status": "in_use", "message": "current model in use"}

    spec = get_model_spec(model_key)
    _clear_model_cache(spec)
    return {"ok": True, "status": "deleted", "model_key": model_key}


@app.post("/api/model/download/cancel")
def api_model_download_cancel(payload: Dict[str, Any] = Body(...)):
    clear = bool(payload.get("clear", True))
    dl_state = _get_download_state()
    status = str(dl_state.get("status") or "")
    model_key = str(dl_state.get("model_key") or "")
    pid = int(dl_state.get("pid") or 0)
    if status not in {"downloading", "ready"}:
        return {"ok": True, "status": "idle"}

    if pid:
        _terminate_pid(pid)
    if clear and model_key in MODEL_KEYS:
        _clear_model_cache(get_model_spec(model_key))

    _set_download_state(
        status="idle",
        model_key=model_key,
        progress=0.0,
        message="cancelled",
        bytes_total=0,
        bytes_done=0,
        speed_mbps=0.0,
        eta_seconds=0,
        done_files=0,
        total_files=0,
        current_file="",
        current_file_bytes=0,
        current_file_total=0,
        pid=0,
        apply_switch=False,
    )
    return {"ok": True, "status": "cancelled", "model_key": model_key}


@app.get("/api/embedding/status")
def api_embedding_status():
    return _get_embedding_state()


@app.get("/api/health")
def api_health():
    return {"ok": True}


def _restart_process() -> None:
    time.sleep(0.6)
    args = [sys.executable, "-m", "uvicorn"] + sys.argv[1:]
    try:
        logger.info("Restarting process: %s", " ".join(args))
        os.execv(sys.executable, args)
    except Exception:
        logger.exception("Restart failed, fallback to spawn")
        try:
            subprocess.Popen(args, cwd=str(CFG.ROOT.parent), env=os.environ.copy())
        finally:
            os._exit(0)


@app.post("/api/restart")
def api_restart():
    t = threading.Thread(target=_restart_process, daemon=True)
    t.start()
    return {"ok": True}



@app.get("/api/pending")
def api_pending(batch_id: Optional[int] = None, limit: int = 2000):
    with DB_LOCK:
        pairs = db.list_pending_stickers(CON, batch_id=int(batch_id) if batch_id is not None else None, limit=limit)
    items = []
    for st, sr in pairs:
        items.append({
            "id": st.id,
            "series": sr.name,
            "batch_id": st.batch_id,
            "filename": st.filename,
            "url": f"/stickers/{sr.name}/{st.filename}",
            "enabled": st.enabled,
            "tags": safe_json_list(st.tags_json),
        })
    return {"items": items, "meta": {"batch_id": batch_id, "count": len(items)}}


@app.post("/api/stickers/bulk_update")
async def api_bulk_update(payload: Dict[str, Any]):
    if not ensure_models():
        raise HTTPException(400, "model not ready")
    assert EMBED is not None

    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        return {"ok": True, "updated": 0}

    cleaned: list[tuple[int, list[str], bool]] = []
    for it in items:
        try:
            sid = int(it.get("id"))
            enabled = bool(it.get("enabled", True))
            tags = it.get("tags", [])
            if isinstance(tags, str):
                tags = parse_tags(tags)
            if not isinstance(tags, list):
                tags = []
            tags = [str(x).strip() for x in tags if str(x).strip()]
            cleaned.append((sid, tags, enabled))
        except Exception:
            continue

    def emb_provider(tags_list: list[str]) -> np.ndarray:
        text = tags_to_text(tags_list) if tags_list else "表情包"
        return EMBED.encode_one(text)

    batch_ids: set[int] = set()
    with DB_LOCK:
        batch_ids.update(db.get_batch_ids_for_stickers(CON, [sid for sid, _, _ in cleaned]))
        n = db.bulk_update_tags_and_enabled(CON, cleaned, emb_provider, on_emb=save_embedding_local)

        cleaned_batches: list[dict[str, Any]] = []
        for bid in batch_ids:
            if db.count_pending(CON, batch_id=bid) == 0:
                detached, deleted = db.detach_batch_keep_stickers(CON, bid)
                cleaned_batches.append({
                    "batch_id": bid,
                    "detached": int(detached),
                    "deleted": bool(deleted),
                })
    refresh_index()
    return {"ok": True, "updated": n, "cleaned_batches": cleaned_batches}


@app.post("/api/stickers/bulk_action")
async def api_bulk_action(payload: Dict[str, Any] = Body(...)):
    action = str(payload.get("action", "")).strip().lower()
    ids = payload.get("ids", [])
    if not isinstance(ids, list):
        raise HTTPException(400, "ids must be list")
    ids = [int(x) for x in ids if str(x).strip().isdigit()]
    ids = list(dict.fromkeys([x for x in ids if x > 0]))
    if not ids:
        return {"ok": True, "action": action, "updated": 0}

    if action not in {"enable", "disable", "move", "delete"}:
        raise HTTPException(400, "invalid action")

    updated = 0
    moved = 0
    deleted = 0
    skipped: list[int] = []

    if action in {"enable", "disable"}:
        enabled = action == "enable"
        with DB_LOCK:
            for sid in ids:
                try:
                    db.set_sticker_enabled(CON, sid, enabled)
                    updated += 1
                except Exception:
                    skipped.append(sid)
        refresh_index()
        return {"ok": True, "action": action, "updated": updated, "skipped": skipped}

    if action == "move":
        series_id = payload.get("series_id")
        if series_id is None:
            raise HTTPException(400, "series_id required")
        try:
            target_id = int(series_id)
        except Exception:
            raise HTTPException(400, "invalid series_id")

        with DB_LOCK:
            target_sr = db.get_series(CON, target_id)
        if not target_sr:
            raise HTTPException(404, "series not found")

        for sid in ids:
            with DB_LOCK:
                got = db.get_sticker(CON, sid)
            if not got:
                skipped.append(sid)
                continue
            st, sr = got
            if st.series_id == target_sr.id:
                skipped.append(sid)
                continue

            src_path = CFG.STICKER_DIR / sr.name / st.filename
            dst_dir = CFG.STICKER_DIR / target_sr.name
            dst_dir.mkdir(parents=True, exist_ok=True)

            new_filename = st.filename
            dst_path = dst_dir / new_filename
            if dst_path.exists():
                new_filename = safe_filename(Path(st.filename).suffix)
                dst_path = dst_dir / new_filename

            try:
                if src_path.exists():
                    src_path.replace(dst_path)
                else:
                    logger.warning("Sticker file missing, only updating DB: %s", src_path)
            except Exception:
                logger.exception("Failed to move sticker file: %s -> %s", src_path, dst_path)

            with DB_LOCK:
                db.update_sticker_series(
                    CON,
                    sid,
                    target_sr.id,
                    filename=new_filename if new_filename != st.filename else None,
                )
            moved += 1

        refresh_index()
        return {"ok": True, "action": action, "moved": moved, "skipped": skipped}

    if action == "delete":
        for sid in ids:
            with DB_LOCK:
                got = db.get_sticker(CON, sid)
            if not got:
                skipped.append(sid)
                continue
            st, sr = got
            img_path = CFG.STICKER_DIR / sr.name / st.filename
            try:
                if img_path.exists():
                    img_path.unlink()
            except Exception:
                logger.exception("Failed to delete file: %s", img_path)

            with DB_LOCK:
                ok = db.delete_sticker(CON, sid)
            if ok:
                deleted += 1
            else:
                skipped.append(sid)

        refresh_index()
        return {"ok": True, "action": action, "deleted": deleted, "skipped": skipped}

    return {"ok": True, "action": action, "updated": updated}


@app.post("/api/sticker/delete")
async def api_delete_sticker(payload: Dict[str, Any]):
    sid = int(payload.get("id", 0))
    if sid <= 0:
        raise HTTPException(400, "invalid id")

    with DB_LOCK:
        got = db.get_sticker(CON, sid)
    if not got:
        raise HTTPException(404, "sticker not found")
    st, sr = got

    img_path = CFG.STICKER_DIR / sr.name / st.filename
    try:
        if img_path.exists():
            img_path.unlink()
    except Exception:
        logger.exception("Failed to delete file: %s", img_path)

    with DB_LOCK:
        ok = db.delete_sticker(CON, sid)

    refresh_index()
    return {"ok": bool(ok), "id": sid}


@app.post("/api/batch/delete")
async def api_delete_batch(payload: Dict[str, Any]):
    bid = int(payload.get("batch_id", 0))
    if bid <= 0:
        raise HTTPException(400, "invalid batch_id")

    with DB_LOCK:
        batch = db.get_batch(CON, bid)
        pairs = db.list_stickers_by_batch(CON, bid)

    if not batch:
        raise HTTPException(404, "batch not found")

    for st, sr in pairs:
        img_path = CFG.STICKER_DIR / sr.name / st.filename
        try:
            if img_path.exists():
                img_path.unlink()
        except Exception:
            logger.exception("Failed to delete file: %s", img_path)

    with DB_LOCK:
        deleted_stickers = db.delete_batch(CON, bid)

    refresh_index()
    return {"ok": True, "batch_id": bid, "deleted_stickers": int(deleted_stickers)}


@app.post("/api/batch/cleanup_if_done")
def api_cleanup_batch(payload: Dict[str, Any]):
    bid = int(payload.get("batch_id", 0))
    if bid <= 0:
        raise HTTPException(400, "invalid batch_id")

    with DB_LOCK:
        batch = db.get_batch(CON, bid)

    if not batch:
        return {"ok": True, "batch_id": bid, "deleted": False, "reason": "batch not found"}

    pending = db_count_pending_compat(bid)
    if pending > 0:
        return {"ok": True, "batch_id": bid, "pending": int(pending), "deleted": False}

    with DB_LOCK:
        detached, deleted = db.detach_batch_keep_stickers(CON, bid)
    logger.info("Auto-clean batch_id=%s detached=%s", bid, detached)
    return {
        "ok": True,
        "batch_id": bid,
        "pending": 0,
        "deleted": bool(deleted),
        "detached": int(detached),
    }


# ---------- Web pages ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse("/try", status_code=302)


@app.get("/admin", response_class=HTMLResponse)
def admin_home():
    return RedirectResponse("/admin/pending", status_code=302)


@app.get("/try", response_class=HTMLResponse)
def page_try(request: Request):
    return templates.TemplateResponse("try.html", {"request": request, "title": "试用", "__PAGE__": "try"})


@app.get("/lab", response_class=HTMLResponse)
def page_lab(request: Request):
    return templates.TemplateResponse("lab.html", {"request": request, "title": "API 调试 / 教学", "__PAGE__": "lab"})


@app.get("/benchmark", response_class=HTMLResponse)
def page_benchmark(request: Request):
    state = _current_model_state()
    return templates.TemplateResponse(
        "benchmark.html",
        {
            "request": request,
            "title": "模型",
            "__PAGE__": "benchmark",
            "models": MODEL_SPECS,
            "current_model_key": state.model_key,
            "recall_topk": int(state.recall_topk or DEFAULT_RECALL_TOPK),
        },
    )


@app.get("/init", response_class=HTMLResponse)
def page_init(request: Request):
    if _any_model_downloaded():
        return RedirectResponse("/benchmark", status_code=302)
    return templates.TemplateResponse(
        "init.html",
        {
            "request": request,
            "title": "初始化模型",
            "__PAGE__": "init",
        },
    )


@app.get("/progress", response_class=HTMLResponse)
def page_progress(request: Request):
    return templates.TemplateResponse(
        "progress.html",
        {
            "request": request,
            "title": "处理中",
            "__PAGE__": "progress",
        },
    )

@app.get("/logs", response_class=HTMLResponse)
def page_logs(request: Request, n: int = 600):
    n = max(50, min(int(n), 5000))
    content = tail_lines(CFG.LOG_PATH, n=n)
    return templates.TemplateResponse("logs.html", {"request": request, "title": "日志", "content": content, "n": n})


# ---------- Info ----------
@app.get("/about", response_class=HTMLResponse)
def page_about(request: Request):
    return templates.TemplateResponse(
        "about.html",
        {
            "request": request,
            "title": "关于",
            "__PAGE__": "about",
            "app_version": CFG.APP_VERSION,
        },
    )


# ---------- Admin: series ----------
@app.get("/admin/series", response_class=HTMLResponse)
def admin_series(request: Request):
    with DB_LOCK:
        rows = db.list_series(CON)
        counts = db.list_series_counts(CON)
    return templates.TemplateResponse(
        "admin_series.html",
        {
            "request": request,
            "title": "系列管理",
            "series": rows,
            "series_counts": counts,
            "__PAGE__": "series",
        },
    )


@app.post("/admin/series/create")
def admin_series_create(name: str = Form(...)):
    name = (name or "").strip()
    if not name:
        raise HTTPException(400, "name empty")
    with DB_LOCK:
        if db.get_series_by_name(CON, name):
            return RedirectResponse("/admin/series", status_code=302)
        db.create_series(CON, name)
    refresh_index()
    return RedirectResponse("/admin/series", status_code=302)


@app.post("/admin/series/toggle")
def admin_series_toggle(id: int = Form(...), enabled: int = Form(...)):
    with DB_LOCK:
        db.set_series_enabled(CON, int(id), bool(int(enabled)))
    refresh_index()
    return RedirectResponse("/admin/series", status_code=302)


def build_series_package(sr) -> bytes:
    with DB_LOCK:
        pairs = db.list_stickers(CON, series_id=sr.id, include_disabled=True, include_needs_tag=True)

    items = []
    missing = []
    for st, _ in pairs:
        tags = safe_json_list(getattr(st, "tags_json", "[]"))
        rel_path = f"stickers/{st.filename}"
        items.append({
            "filename": st.filename,
            "path": rel_path,
            "ext": st.ext,
            "enabled": bool(st.enabled),
            "needs_tag": bool(getattr(st, "needs_tag", False)),
            "tags": tags,
        })
        file_path = CFG.STICKER_DIR / sr.name / st.filename
        if not file_path.exists():
            missing.append(st.filename)

    manifest = {
        "version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "series": {"name": sr.name, "enabled": bool(sr.enabled)},
        "stickers": items,
        "stats": {"total": len(items), "missing": len(missing)},
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for it in items:
            file_path = CFG.STICKER_DIR / sr.name / it["filename"]
            if file_path.exists():
                zf.write(file_path, it["path"])
            else:
                logger.warning("Export missing file: %s", file_path)

    return buf.getvalue()


@app.get("/admin/series/export")
def admin_series_export(series_id: int):
    with DB_LOCK:
        sr = db.get_series(CON, int(series_id))
    if not sr:
        raise HTTPException(404, "series not found")
    buf = io.BytesIO(build_series_package(sr))
    buf.seek(0)
    safe_name = safe_export_name(sr.name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{ts}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.post("/admin/series/export_bulk")
async def admin_series_export_bulk(payload: Dict[str, Any] = Body(...)):
    series_ids = payload.get("series_ids", [])
    if not isinstance(series_ids, list):
        raise HTTPException(400, "series_ids must be list")
    series_ids = [int(x) for x in series_ids if str(x).strip().isdigit()]
    series_ids = list(dict.fromkeys([x for x in series_ids if x > 0]))
    if not series_ids:
        raise HTTPException(400, "series_ids empty")

    with DB_LOCK:
        rows = {s.id: s for s in db.list_series(CON)}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sid in series_ids:
            sr = rows.get(sid)
            if not sr:
                continue
            data = build_series_package(sr)
            safe_name = safe_export_name(sr.name)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            zf.writestr(f"{safe_name}_{ts}.zip", data)

    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    headers = {"Content-Disposition": f'attachment; filename="series_export_{ts}.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.post("/admin/series/delete_bulk")
async def admin_series_delete_bulk(payload: Dict[str, Any] = Body(...)):
    series_ids = payload.get("series_ids", [])
    if not isinstance(series_ids, list):
        raise HTTPException(400, "series_ids must be list")
    series_ids = [int(x) for x in series_ids if str(x).strip().isdigit()]
    series_ids = list(dict.fromkeys([x for x in series_ids if x > 0]))
    if not series_ids:
        return {"ok": True, "deleted": 0}

    deleted = 0
    skipped: list[int] = []

    with DB_LOCK:
        rows = {s.id: s for s in db.list_series(CON)}

    for sid in series_ids:
        sr = rows.get(sid)
        if not sr:
            skipped.append(sid)
            continue

        with DB_LOCK:
            pairs = db.list_stickers(CON, series_id=sr.id, include_disabled=True, include_needs_tag=True)

        for st, _ in pairs:
            img_path = CFG.STICKER_DIR / sr.name / st.filename
            try:
                if img_path.exists():
                    img_path.unlink()
            except Exception:
                logger.exception("Failed to delete file: %s", img_path)

        with DB_LOCK:
            ok = db.delete_series(CON, sr.id)
        if ok:
            deleted += 1
        else:
            skipped.append(sid)

        series_dir = CFG.STICKER_DIR / sr.name
        try:
            if series_dir.exists() and not any(series_dir.iterdir()):
                series_dir.rmdir()
        except Exception:
            logger.exception("Failed to remove series dir: %s", series_dir)

    refresh_index()
    return {"ok": True, "deleted": deleted, "skipped": skipped}


@app.post("/admin/series/import")
async def admin_series_import(
    file: UploadFile = File(...),
    mode: str = Form("ask"),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "invalid zip")

    if "manifest.json" not in zf.namelist():
        raise HTTPException(400, "manifest.json not found")

    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    series_info = manifest.get("series") or {}
    series_name = (series_info.get("name") or manifest.get("series_name") or "").strip()
    if not series_name:
        raise HTTPException(400, "series name missing")

    with DB_LOCK:
        existing = db.get_series_by_name(CON, series_name)

    if existing and mode != "merge":
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "conflict": True,
                "series_name": series_name,
                "series_id": existing.id,
                "message": "series exists",
            },
        )

    with DB_LOCK:
        if existing:
            sr = existing
        else:
            sr = db.create_series(CON, series_name)
            if "enabled" in series_info:
                db.set_series_enabled(CON, sr.id, bool(series_info.get("enabled")))

    if not ensure_models():
        raise HTTPException(400, "model not ready")
    assert EMBED is not None

    stickers = manifest.get("stickers") or manifest.get("items") or []
    dest_dir = CFG.STICKER_DIR / sr.name
    dest_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped: list[dict[str, str]] = []

    for it in stickers:
        filename = str(it.get("filename") or it.get("name") or "").strip()
        zip_path = str(it.get("path") or "").strip()
        if not filename and zip_path:
            filename = Path(zip_path).name
        if not filename:
            skipped.append({"file": zip_path or "", "reason": "missing filename"})
            continue

        filename = Path(filename).name
        ext = str(it.get("ext") or Path(filename).suffix).lower()
        if ext not in CFG.ALLOWED_EXTS:
            skipped.append({"file": filename, "reason": f"ext not allowed: {ext}"})
            continue

        if not zip_path:
            zip_path = f"stickers/{filename}"
        if zip_path not in zf.namelist():
            skipped.append({"file": filename, "reason": "file not found in package"})
            continue

        data = zf.read(zip_path)
        target_name = filename
        target_path = dest_dir / target_name
        if target_path.exists():
            target_name = safe_filename(ext or Path(filename).suffix)
            target_path = dest_dir / target_name

        target_path.write_bytes(data)

        tags = it.get("tags", [])
        if isinstance(tags, str):
            tags = parse_tags(tags)
        if not isinstance(tags, list):
            tags = []
        tags = [str(x).strip() for x in tags if str(x).strip()]

        emb = EMBED.encode_one(tags_to_text(tags) if tags else "表情包")
        with DB_LOCK:
            sid = db.create_sticker(
                CON,
                sr.id,
                target_name,
                ext or Path(target_name).suffix,
                tags,
                emb,
                batch_id=None,
                needs_tag=(len(tags) == 0),
                enabled=bool(it.get("enabled", True)),
            )
        save_embedding_local(sid, emb)
        imported += 1

    refresh_index()
    return {
        "ok": True,
        "series_name": sr.name,
        "series_id": sr.id,
        "imported": imported,
        "skipped": skipped,
        "merged": bool(existing),
    }


# ---------- Admin: stickers ----------
@app.get("/admin/stickers", response_class=HTMLResponse)
def admin_stickers(request: Request, series_id: Optional[int] = None, show_disabled: int = 1):

    with DB_LOCK:
        rows = db.list_series(CON)
        counts = db.list_series_counts(CON)
        sel = int(series_id) if series_id is not None and str(series_id).strip() else None
        show = bool(int(show_disabled))

        pairs = []
        if sel is not None:
            # 兼容：即使 db.list_stickers 旧版不支持 include_needs_tag 也不会炸
            pairs = db_list_stickers_compat(series_id=sel, include_disabled=show, include_needs_tag=True)

    items = []
    if sel is not None:
        for st, sr in pairs:
            items.append({
                "id": st.id,
                "series_id": st.series_id,
                "series_name": sr.name,
                "series_enabled": getattr(sr, "enabled", True),
                "filename": st.filename,
                "enabled": st.enabled,
                "needs_tag": getattr(st, "needs_tag", False),
                "tags": safe_json_list(getattr(st, "tags_json", "[]")),
                "url": f"/stickers/{sr.name}/{st.filename}",
            })

    return_to = request.url.path
    if request.url.query:
        return_to += f"?{request.url.query}"

    series_cards = [
        {
            "id": s.id,
            "name": s.name,
            "enabled": s.enabled,
            "count": int(counts.get(s.id, 0)),
        }
        for s in rows
    ]

    return templates.TemplateResponse("admin_stickers.html", {
        "request": request,
        "title": "表情包管理",
        "series": rows,
        "stickers": items,
        "series_cards": series_cards,
        "selected_series_id": sel,
        "show_disabled": show,
        "return_to": return_to,
        "mode": "stickers" if sel is not None else "series",
        "__PAGE__": "stickers"
    })


# ---------- Admin: pending / batch ----------
@app.get("/admin/pending", response_class=HTMLResponse)
def admin_pending(request: Request):
    with DB_LOCK:
        pending_count = db_count_pending_compat(None)
        batches = db.list_batches(CON, limit=60)
        series = {s.id: s for s in db.list_series(CON)}
    return templates.TemplateResponse("pending.html", {
        "request": request,
        "title": "待打 Tag",
        "pending_count": pending_count,
        "batches": batches,
        "series_map": series,
        "__PAGE__": "pending"
    })


@app.get("/admin/upload", response_class=HTMLResponse)
def admin_upload(request: Request):
    with DB_LOCK:
        rows = db.list_series(CON)
    return templates.TemplateResponse(
        "upload_batch.html",
        {"request": request, "title": "批量上传（创建批次）", "series": rows, "max_upload_mb": int(CFG.MAX_UPLOAD_MB), "__PAGE__": "upload"},
    )


@app.post("/admin/upload")
async def admin_upload_post(
    request: Request,
    files: List[UploadFile] = File(...),
    series_id: int = Form(...),
    note: str = Form(""),
):
    with DB_LOCK:
        sr = db.get_series(CON, int(series_id))
    if not sr:
        raise HTTPException(400, "series not found")

    with DB_LOCK:
        batch = db.create_batch(CON, sr.id, note=note or "")

    batch_dir = CFG.STICKER_DIR / sr.name
    batch_dir.mkdir(parents=True, exist_ok=True)

    if not ensure_models():
        raise HTTPException(400, "model not ready")
    assert EMBED is not None

    created_ids: list[int] = []
    skipped: list[dict[str, str]] = []

    max_bytes = int(CFG.MAX_UPLOAD_MB) * 1024 * 1024

    for f in files:
        if not f.filename:
            skipped.append({"file": "", "reason": "empty filename"})
            continue

        ext = Path(f.filename).suffix.lower()
        if ext not in CFG.ALLOWED_EXTS:
            skipped.append({"file": f.filename, "reason": f"ext not allowed: {ext}"})
            continue

        raw = await f.read()
        if len(raw) > max_bytes:
            skipped.append({"file": f.filename, "reason": f"too large: {len(raw)} bytes"})
            continue

        fname = safe_filename(ext)
        path = batch_dir / fname
        path.write_bytes(raw)

        emb = EMBED.encode_one("表情包")
        with DB_LOCK:
            sid = db.create_sticker(
                CON, sr.id, fname, ext,
                tags=[], emb=emb, batch_id=batch.id,
                needs_tag=True, enabled=True
            )
        save_embedding_local(sid, emb)
        created_ids.append(sid)

    logger.info("batch upload batch_id=%s series=%s created=%s skipped=%s", batch.id, sr.name, len(created_ids), len(skipped))

    redirect_url = f"/admin/batch/{batch.id}"
    wants_json = (
        request.headers.get("x-requested-with", "").lower() == "xmlhttprequest"
        or "application/json" in request.headers.get("accept", "").lower()
    )
    if wants_json:
        return JSONResponse({
            "ok": True,
            "batch_id": batch.id,
            "series": sr.name,
            "created": len(created_ids),
            "skipped": skipped,
            "redirect_url": redirect_url,
        })

    return RedirectResponse(redirect_url, status_code=302)


@app.get("/admin/batch/{batch_id}", response_class=HTMLResponse)
def admin_batch_tagging(request: Request, batch_id: int):
    with DB_LOCK:
        batch = db.get_batch(CON, int(batch_id))
        rows = db.list_series(CON)
        pending_count = db_count_pending_compat(int(batch_id))
    if not batch:
        raise HTTPException(404, "batch not found")
    return templates.TemplateResponse("batch_tagging.html", {
        "request": request,
        "title": f"批量打 Tag / Batch #{batch.id}",
        "batch_id": batch.id,
        "batch_note": getattr(batch, "note", ""),
        "batch_series_id": getattr(batch, "series_id", 0),
        "pending_count": pending_count,
        "series": rows,
        "__PAGE__": "batch"
    })


# ---------- Admin: sticker edit (single) ----------
@app.get("/admin/sticker/{sticker_id}", response_class=HTMLResponse)
def admin_sticker_edit(request: Request, sticker_id: int):
    with DB_LOCK:
        got = db.get_sticker(CON, int(sticker_id))
        rows = db.list_series(CON)
    if not got:
        raise HTTPException(404, "not found")
    st, sr = got

    data = {
        "id": st.id,
        "series_id": st.series_id,
        "series_name": sr.name,
        "filename": st.filename,
        "url": f"/stickers/{sr.name}/{st.filename}",
        "enabled": st.enabled,
        "needs_tag": getattr(st, "needs_tag", False),
        "tags": safe_json_list(getattr(st, "tags_json", "[]")),
    }
    return templates.TemplateResponse("sticker_edit.html", {
        "request": request,
        "title": f"编辑 #{st.id}",
        "st": data,
        "series": rows,
        "__PAGE__": "edit"
    })


@app.post("/admin/sticker/{sticker_id}/update")
def admin_sticker_update(
    sticker_id: int,
    tags: str = Form(""),
    enabled: int = Form(1),
    series_id: Optional[int] = Form(None),
    return_to: str = Form(""),
):
    if not ensure_models():
        raise HTTPException(400, "model not ready")
    assert EMBED is not None
    tags_list = parse_tags(tags)
    emb = EMBED.encode_one(tags_to_text(tags_list) if tags_list else "表情包")

    with DB_LOCK:
        got = db.get_sticker(CON, int(sticker_id))
    if not got:
        raise HTTPException(404, "not found")
    st, sr = got

    target_series_id: Optional[int] = None
    if series_id is not None:
        try:
            target_series_id = int(series_id)
        except Exception:
            target_series_id = None

    if target_series_id and target_series_id != st.series_id:
        with DB_LOCK:
            target_sr = db.get_series(CON, target_series_id)
        if not target_sr:
            raise HTTPException(400, "series not found")

        src_path = CFG.STICKER_DIR / sr.name / st.filename
        dst_dir = CFG.STICKER_DIR / target_sr.name
        dst_dir.mkdir(parents=True, exist_ok=True)

        new_filename = st.filename
        dst_path = dst_dir / new_filename
        if dst_path.exists():
            new_filename = safe_filename(Path(st.filename).suffix)
            dst_path = dst_dir / new_filename

        try:
            if src_path.exists():
                src_path.replace(dst_path)
            else:
                logger.warning("Sticker file missing, only updating DB: %s", src_path)
        except Exception:
            logger.exception("Failed to move sticker file: %s -> %s", src_path, dst_path)

        with DB_LOCK:
            db.update_sticker_series(
                CON,
                int(sticker_id),
                target_sr.id,
                filename=new_filename if new_filename != st.filename else None,
            )

    with DB_LOCK:
        db.update_sticker_tags_and_emb(CON, int(sticker_id), tags_list, emb, needs_tag=(len(tags_list) == 0))
        db.set_sticker_enabled(CON, int(sticker_id), bool(int(enabled)))
    save_embedding_local(int(sticker_id), emb)

    refresh_index()
    logger.info("update sticker id=%s tags=%s enabled=%s series_id=%s", sticker_id, tags_list, enabled, target_series_id)

    return_to = (return_to or "").strip()
    if not return_to.startswith("/"):
        return_to = f"/admin/sticker/{sticker_id}"
    return RedirectResponse(return_to, status_code=302)
