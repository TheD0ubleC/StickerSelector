from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    # Models
    M3E_MODEL: str = "moka-ai/m3e-small"

    # App
    APP_VERSION: str = "Release-1.0.0"

    # Runtime dirs
    ROOT: Path = Path(__file__).resolve().parent
    RUNTIME_DIR: Path = ROOT / "data_runtime"
    DB_PATH: Path = RUNTIME_DIR / "stickers.db"
    LOG_PATH: Path = RUNTIME_DIR / "app.log"
    STICKER_DIR: Path = RUNTIME_DIR / "stickers"
    EMBEDDING_DIR: Path = RUNTIME_DIR / "embeddings"
    MODEL_STATE_PATH: Path = RUNTIME_DIR / "model_state.json"
    DOWNLOAD_STATE_PATH: Path = RUNTIME_DIR / "download_state.json"
    MODEL_DIR: Path = RUNTIME_DIR / "models"
    HF_ENDPOINT_PATH: Path = RUNTIME_DIR / "hf_endpoint.json"

    # UI / API defaults
    DEFAULT_TOPK: int = 6
    MAX_TOPK: int = 50
    # Similarity -> "fit rate" (softmax scale)
    FIT_SOFTMAX_SCALE: float = 55.0

    # File limits
    MAX_UPLOAD_MB: int = 25
    ALLOWED_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".gif", ".webp")


CFG = AppConfig()
CFG.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
CFG.STICKER_DIR.mkdir(parents=True, exist_ok=True)
CFG.EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
CFG.MODEL_DIR.mkdir(parents=True, exist_ok=True)
