from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi, hf_hub_download
try:
    hf_tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")
except Exception:
    hf_tqdm_mod = None

from sticker_service.config import CFG
from sticker_service.model_state import (
    DEFAULT_RECALL_TOPK,
    ModelState,
    get_model_spec,
    save_model_state,
)


os.environ["HF_HUB_CACHE"] = str(CFG.MODEL_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CFG.MODEL_DIR)

STATE_PATH: Path = CFG.DOWNLOAD_STATE_PATH
_ACTIVE_PROGRESS: Optional["_DownloadProgress"] = None


def _normalize_endpoint(endpoint: str) -> str:
    cleaned = str(endpoint or "").strip()
    if not cleaned:
        return "https://huggingface.co"
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    return f"https://{cleaned}"


def _get_hf_endpoint() -> str:
    path = CFG.HF_ENDPOINT_PATH
    if path.exists():
        try:
            data = path.read_text("utf-8")
            import json

            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return _normalize_endpoint(str(parsed.get("endpoint") or ""))
        except Exception:
            pass
    return "https://huggingface.co"


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = write_json_str(state)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(data, "utf-8")
        tmp.replace(path)
        return
    except Exception:
        pass
    try:
        path.write_text(data, "utf-8")
    except Exception:
        pass


def _set_download_state(state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    state["pid"] = os.getpid()
    _write_state(STATE_PATH, state)


def write_json_str(obj: dict[str, Any]) -> str:
    return write_json_bytes(obj).decode("utf-8")


def write_json_bytes(obj: dict[str, Any]) -> bytes:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def _build_download_list(repo_id: str) -> list[tuple[str, str, int]]:
    api = HfApi(endpoint=_get_hf_endpoint())
    info = api.model_info(repo_id=repo_id)
    files: list[tuple[str, str, int]] = []
   


    for sibling in getattr(info, "siblings", []) or []:
        name = getattr(sibling, "rfilename", None)
        lname = name.lower()

        if lname.endswith((".onnx", ".onnx_data", ".pb", ".h5")):
            continue
        if not name:
            continue
        size = int(getattr(sibling, "size", 0) or 0)
        files.append((repo_id, name, size))
    return files


def _progress_callback(current_bytes: Optional[float], total_bytes: Optional[float]) -> None:
    tracker = _ACTIVE_PROGRESS
    if tracker is None:
        return
    try:
        tracker.on_chunk(current_bytes, total_bytes)
    except Exception:
        pass


def _install_progress_hook() -> None:
    if hf_tqdm_mod is None:
        return
    if getattr(hf_tqdm_mod, "_sticker_progress_patched", False):
        return
    orig_tqdm = hf_tqdm_mod.tqdm

    class _StateTqdm(orig_tqdm):
        def __init__(self, *args, **kwargs):
            self._sticker_total = kwargs.get("total")
            self._sticker_n = float(kwargs.get("initial") or 0)
            super().__init__(*args, **kwargs)
            _progress_callback(self._sticker_n, self._sticker_total)

        def update(self, n=1):
            try:
                inc = float(n)
            except Exception:
                inc = 0.0
            self._sticker_n += inc
            result = super().update(n)
            _progress_callback(self._sticker_n, self._sticker_total)
            return result

    hf_tqdm_mod.tqdm = _StateTqdm
    hf_tqdm_mod._sticker_progress_patched = True


class _DownloadProgress:
    def __init__(
        self,
        *,
        model_key: str,
        total_files: int,
        total_bytes: int,
        apply_switch: bool,
    ) -> None:
        self.model_key = model_key
        self.total_files = max(1, int(total_files))
        self.total_bytes = max(0, int(total_bytes))
        self.use_bytes = self.total_bytes > 0
        self.apply_switch = bool(apply_switch)
        self.done_files = 0
        self.done_bytes = 0
        self.current_file_bytes = 0.0
        self.current_file_size = 0
        self.current_file_name = ""
        self.start_time = time.perf_counter()
        self.last_emit = 0.0

    def set_current_file(self, name: str, size: int) -> None:
        self.current_file_bytes = 0.0
        self.current_file_size = int(size or 0)
        self.current_file_name = str(name or "")

    def on_chunk(self, current_bytes: Optional[float], total_bytes: Optional[float]) -> None:
        if current_bytes is None:
            return
        try:
            current_val = float(current_bytes)
        except Exception:
            return
        if current_val < 0:
            return
        self.current_file_bytes = max(self.current_file_bytes, current_val)
        if self.current_file_size <= 0 and total_bytes:
            try:
                size = int(total_bytes)
                self.current_file_size = size
                self.total_bytes += max(0, size)
            except Exception:
                pass
        self._emit_state(force=False)

    def finish_file(self) -> None:
        final_bytes = max(int(self.current_file_bytes), int(self.current_file_size))
        self.done_bytes += final_bytes
        self.done_files += 1
        self.current_file_bytes = 0.0
        self.current_file_size = 0
        self.current_file_name = ""
        self._emit_state(force=True)

    def _calc_progress(self, bytes_done: int) -> float:
        if self.use_bytes and self.total_bytes > 0:
            return min(100.0, bytes_done / self.total_bytes * 100.0)
        file_fraction = 0.0
        if self.current_file_size > 0:
            file_fraction = min(1.0, self.current_file_bytes / self.current_file_size)
        return min(100.0, (self.done_files + file_fraction) / max(1, self.total_files) * 100.0)

    def _emit_state(self, force: bool) -> None:
        now = time.perf_counter()
        if not force and (now - self.last_emit) < 0.25:
            return
        self.last_emit = now
        bytes_done = int(self.done_bytes + self.current_file_bytes)
        progress = self._calc_progress(bytes_done)
        elapsed = max(0.001, now - self.start_time)
        speed_mbps = (bytes_done / elapsed) / (1024 * 1024) if bytes_done > 0 else 0.0
        remaining = max(0, int(self.total_bytes) - int(bytes_done))
        eta_seconds = int(remaining / (speed_mbps * 1024 * 1024)) if speed_mbps > 0 and self.total_bytes > 0 else 0
        _set_download_state(
            {
                "status": "downloading",
                "model_key": self.model_key,
                "progress": progress,
                "message": f"downloading {self.done_files}/{self.total_files}",
                "bytes_total": int(self.total_bytes),
                "bytes_done": int(bytes_done),
                "speed_mbps": round(speed_mbps, 2),
                "eta_seconds": int(eta_seconds),
                "done_files": int(self.done_files),
                "total_files": int(self.total_files),
                "current_file": self.current_file_name,
                "current_file_bytes": int(self.current_file_bytes),
                "current_file_total": int(self.current_file_size),
                "apply_switch": self.apply_switch,
            }
        )


def run_download(model_key: str, recall_topk: int, apply_switch: bool) -> None:
    spec = get_model_spec(model_key)
    repos = [spec.embed_model]
    if spec.mode == "rerank" and spec.reranker_model:
        repos.append(spec.reranker_model)

    _set_download_state(
        {
            "status": "downloading",
            "model_key": spec.key,
            "progress": 0.0,
            "message": "resolving",
            "bytes_total": 0,
            "bytes_done": 0,
            "speed_mbps": 0.0,
            "eta_seconds": 0,
            "done_files": 0,
            "total_files": 0,
            "current_file": "",
            "current_file_bytes": 0,
            "current_file_total": 0,
            "apply_switch": bool(apply_switch),
        }
    )

    files: list[tuple[str, str, int]] = []
    for repo_id in repos:
        files.extend(_build_download_list(repo_id))

    total_bytes = sum(size for _, _, size in files if size > 0)
    total_files = len(files)

    _install_progress_hook()
    tracker = _DownloadProgress(
        model_key=spec.key,
        total_files=total_files,
        total_bytes=total_bytes,
        apply_switch=apply_switch,
    )
    _set_download_state(
        {
            "status": "downloading",
            "model_key": spec.key,
            "progress": 0.0,
            "message": f"downloading 0/{max(1, total_files)}",
            "bytes_total": int(total_bytes),
            "bytes_done": 0,
            "speed_mbps": 0.0,
            "eta_seconds": 0,
            "done_files": 0,
            "total_files": int(total_files),
            "current_file": "",
            "current_file_bytes": 0,
            "current_file_total": 0,
            "apply_switch": bool(apply_switch),
        }
    )

    for repo_id, filename, size in files:
        tracker.set_current_file(filename, size)
        try:
            global _ACTIVE_PROGRESS
            _ACTIVE_PROGRESS = tracker
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(CFG.MODEL_DIR),
                endpoint=_get_hf_endpoint(),
            )
        finally:
            _ACTIVE_PROGRESS = None
        tracker.finish_file()

    elapsed = max(0.001, time.perf_counter() - tracker.start_time)
    final_speed_mbps = (tracker.done_bytes / elapsed) / (1024 * 1024) if tracker.done_bytes > 0 else 0.0

    if apply_switch:
        next_state = ModelState(
            model_key=spec.key,
            recall_topk=recall_topk,
            pending_rebuild=True,
        )
        save_model_state(next_state)
        _set_download_state(
            {
                "status": "ready",
                "model_key": spec.key,
                "progress": 100.0,
                "message": "ready",
                "bytes_total": int(tracker.total_bytes),
                "bytes_done": int(tracker.done_bytes),
                "speed_mbps": round(final_speed_mbps, 2),
                "eta_seconds": 0,
                "done_files": int(tracker.done_files),
                "total_files": int(tracker.total_files),
                "current_file": "",
                "current_file_bytes": 0,
                "current_file_total": 0,
                "apply_switch": True,
            }
        )
    else:
        _set_download_state(
            {
                "status": "done",
                "model_key": spec.key,
                "progress": 100.0,
                "message": "downloaded",
                "bytes_total": int(tracker.total_bytes),
                "bytes_done": int(tracker.done_bytes),
                "speed_mbps": round(final_speed_mbps, 2),
                "eta_seconds": 0,
                "done_files": int(tracker.done_files),
                "total_files": int(tracker.total_files),
                "current_file": "",
                "current_file_bytes": 0,
                "current_file_total": 0,
                "apply_switch": False,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--recall-topk", type=int, default=DEFAULT_RECALL_TOPK)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--state-path")
    args = parser.parse_args()
    recall_topk = max(1, int(args.recall_topk or DEFAULT_RECALL_TOPK))
    apply_switch = bool(args.apply) and not bool(args.download_only)
    if args.state_path:
        try:
            global STATE_PATH
            STATE_PATH = Path(str(args.state_path)).expanduser()
        except Exception:
            pass
    try:
        run_download(str(args.model_key), recall_topk, apply_switch)
    except Exception:
        _set_download_state(
            {
                "status": "error",
                "model_key": str(args.model_key),
                "progress": 0.0,
                "message": "download failed",
                "eta_seconds": 0,
                "done_files": 0,
                "total_files": 0,
                "current_file": "",
                "current_file_bytes": 0,
                "current_file_total": 0,
                "apply_switch": apply_switch,
            }
        )
        raise


if __name__ == "__main__":
    main()
