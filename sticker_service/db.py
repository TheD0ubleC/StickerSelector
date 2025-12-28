from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np


# =========================
# Models
# =========================

@dataclass
class SeriesRow:
    id: int
    name: str
    enabled: bool
    created_at: float


@dataclass
class BatchRow:
    id: int
    series_id: int
    note: str
    created_at: float


@dataclass
class StickerRow:
    id: int
    series_id: int
    filename: str
    ext: str
    enabled: bool
    needs_tag: bool
    batch_id: Optional[int]
    tags_json: str
    emb_blob: Optional[bytes]
    emb_dim: int
    created_at: float


# =========================
# Connection / Migration
# =========================

def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA foreign_keys=ON;")
    return con


def _has_column(con: sqlite3.Connection, table: str, col: str) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return any(r["name"] == col for r in cur.fetchall())


def _ensure_column(con: sqlite3.Connection, table: str, col: str, ddl: str) -> None:
    if not _has_column(con, table, col):
        cur = con.cursor()
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {ddl};")
        con.commit()


def init_db(con: sqlite3.Connection) -> None:
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS series (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 1,
          created_at REAL NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS batch (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          series_id INTEGER NOT NULL,
          note TEXT NOT NULL DEFAULT '',
          created_at REAL NOT NULL,
          FOREIGN KEY(series_id) REFERENCES series(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sticker (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          series_id INTEGER NOT NULL,
          filename TEXT NOT NULL,
          ext TEXT NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 1,
          needs_tag INTEGER NOT NULL DEFAULT 0,
          batch_id INTEGER,
          tags_json TEXT NOT NULL DEFAULT '[]',
          emb_blob BLOB,
          emb_dim INTEGER NOT NULL DEFAULT 0,
          created_at REAL NOT NULL,
          FOREIGN KEY(series_id) REFERENCES series(id) ON DELETE CASCADE,
          FOREIGN KEY(batch_id) REFERENCES batch(id) ON DELETE SET NULL
        );
        """
    )

    # Helpful indices
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sticker_series_id ON sticker(series_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sticker_enabled ON sticker(enabled);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sticker_needs_tag ON sticker(needs_tag);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sticker_batch_id ON sticker(batch_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_batch_series_id ON batch(series_id);")

    con.commit()

    # In case this DB was created by an old version (migrations)
    _ensure_column(con, "sticker", "needs_tag", "needs_tag INTEGER NOT NULL DEFAULT 0")
    _ensure_column(con, "sticker", "batch_id", "batch_id INTEGER")
    _ensure_column(con, "sticker", "tags_json", "tags_json TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(con, "sticker", "emb_dim", "emb_dim INTEGER NOT NULL DEFAULT 0")

    con.commit()


# =========================
# Utils
# =========================

def _now() -> float:
    return float(time.time())


def _tags_to_json(tags: Iterable[str]) -> str:
    return json.dumps(list(tags), ensure_ascii=False)


def _row_to_series(r: sqlite3.Row) -> SeriesRow:
    return SeriesRow(id=int(r["id"]), name=str(r["name"]), enabled=bool(r["enabled"]), created_at=float(r["created_at"]))


def _row_to_batch(r: sqlite3.Row) -> BatchRow:
    return BatchRow(
        id=int(r["id"]),
        series_id=int(r["series_id"]),
        note=str(r["note"] or ""),
        created_at=float(r["created_at"]),
    )


def _row_to_sticker(r: sqlite3.Row) -> StickerRow:
    return StickerRow(
        id=int(r["id"]),
        series_id=int(r["series_id"]),
        filename=str(r["filename"]),
        ext=str(r["ext"]),
        enabled=bool(r["enabled"]),
        needs_tag=bool(r["needs_tag"]),
        batch_id=int(r["batch_id"]) if r["batch_id"] is not None else None,
        tags_json=str(r["tags_json"] or "[]"),
        emb_blob=r["emb_blob"],
        emb_dim=int(r["emb_dim"] or 0),
        created_at=float(r["created_at"]),
    )


# =========================
# Series
# =========================

def list_series(con: sqlite3.Connection) -> List[SeriesRow]:
    cur = con.cursor()
    cur.execute("SELECT * FROM series ORDER BY id ASC;")
    return [_row_to_series(r) for r in cur.fetchall()]


def get_series(con: sqlite3.Connection, series_id: int) -> Optional[SeriesRow]:
    cur = con.cursor()
    cur.execute("SELECT * FROM series WHERE id=?;", (int(series_id),))
    r = cur.fetchone()
    return _row_to_series(r) if r else None


def get_series_by_name(con: sqlite3.Connection, name: str) -> Optional[SeriesRow]:
    cur = con.cursor()
    cur.execute("SELECT * FROM series WHERE name=?;", (name,))
    r = cur.fetchone()
    return _row_to_series(r) if r else None


def create_series(con: sqlite3.Connection, name: str) -> SeriesRow:
    name = (name or "").strip()
    if not name:
        raise ValueError("series name empty")
    cur = con.cursor()
    ts = _now()
    cur.execute("INSERT INTO series(name, enabled, created_at) VALUES(?, 1, ?);", (name, ts))
    con.commit()
    return get_series(con, int(cur.lastrowid))  # type: ignore[return-value]


def set_series_enabled(con: sqlite3.Connection, series_id: int, enabled: bool) -> None:
    cur = con.cursor()
    cur.execute("UPDATE series SET enabled=? WHERE id=?;", (1 if enabled else 0, int(series_id)))
    con.commit()


def delete_series(con: sqlite3.Connection, series_id: int) -> bool:
    cur = con.cursor()
    cur.execute("DELETE FROM series WHERE id=?;", (int(series_id),))
    changed = cur.rowcount > 0
    con.commit()
    return bool(changed)


# =========================
# Batch
# =========================

def create_batch(con: sqlite3.Connection, series_id: int, note: str = "") -> BatchRow:
    cur = con.cursor()
    ts = _now()
    cur.execute("INSERT INTO batch(series_id, note, created_at) VALUES(?, ?, ?);", (int(series_id), note or "", ts))
    con.commit()
    return get_batch(con, int(cur.lastrowid))  # type: ignore[return-value]


def get_batch(con: sqlite3.Connection, batch_id: int) -> Optional[BatchRow]:
    cur = con.cursor()
    cur.execute("SELECT * FROM batch WHERE id=?;", (int(batch_id),))
    r = cur.fetchone()
    return _row_to_batch(r) if r else None


def list_batches(con: sqlite3.Connection, limit: int = 60) -> List[BatchRow]:
    limit = max(1, min(int(limit), 500))
    cur = con.cursor()
    cur.execute("SELECT * FROM batch ORDER BY id DESC LIMIT ?;", (limit,))
    return [_row_to_batch(r) for r in cur.fetchall()]


def delete_batch(con: sqlite3.Connection, batch_id: int) -> int:
    """
    Delete batch row and detach/delete its stickers.
    Return number of deleted stickers.
    """
    bid = int(batch_id)
    cur = con.cursor()
    cur.execute("SELECT COUNT(1) as c FROM sticker WHERE batch_id=?;", (bid,))
    cnt = int(cur.fetchone()["c"])  # type: ignore[index]
    # Delete stickers first
    cur.execute("DELETE FROM sticker WHERE batch_id=?;", (bid,))
    cur.execute("DELETE FROM batch WHERE id=?;", (bid,))
    con.commit()
    return cnt


def detach_batch_keep_stickers(con: sqlite3.Connection, batch_id: int) -> tuple[int, int]:
    """
    Delete batch row while keeping stickers by clearing their batch_id.
    Return (detached_stickers, deleted_batch_rows).
    """
    bid = int(batch_id)
    cur = con.cursor()
    cur.execute("UPDATE sticker SET batch_id=NULL WHERE batch_id=?;", (bid,))
    detached = int(cur.rowcount)
    cur.execute("DELETE FROM batch WHERE id=?;", (bid,))
    deleted = int(cur.rowcount)
    con.commit()
    return detached, deleted


# =========================
# Sticker
# =========================

def list_stickers(
    con: sqlite3.Connection,
    series_id: Optional[int] = None,
    include_disabled: bool = True,
    include_needs_tag: bool = False,
) -> List[Tuple[StickerRow, SeriesRow]]:
    cur = con.cursor()

    sql = """
    SELECT
      sticker.*,
      series.name as series_name,
      series.enabled as series_enabled,
      series.created_at as series_created
    FROM sticker
    JOIN series ON series.id = sticker.series_id
    """
    params: list[Any] = []
    conds: list[str] = []

    if series_id is not None:
        conds.append("sticker.series_id=?")
        params.append(int(series_id))

    if not include_disabled:
        conds.append("sticker.enabled=1")
        conds.append("series.enabled=1")

    if not include_needs_tag:
        conds.append("sticker.needs_tag=0")

    if conds:
        sql += " WHERE " + " AND ".join(conds)

    sql += " ORDER BY sticker.id DESC;"

    cur.execute(sql, params)

    out: list[Tuple[StickerRow, SeriesRow]] = []
    for r in cur.fetchall():
        srow = SeriesRow(
            id=int(r["series_id"]),
            name=str(r["series_name"]),
            enabled=bool(r["series_enabled"]),
            created_at=float(r["series_created"]),
        )
        st = _row_to_sticker(r)
        out.append((st, srow))
    return out


def list_stickers_by_batch(con: sqlite3.Connection, batch_id: int) -> List[Tuple[StickerRow, SeriesRow]]:
    cur = con.cursor()
    cur.execute(
        """
        SELECT
          sticker.*,
          series.name as series_name,
          series.enabled as series_enabled,
          series.created_at as series_created
        FROM sticker
        JOIN series ON series.id = sticker.series_id
        WHERE sticker.batch_id=?
        ORDER BY sticker.id DESC;
        """,
        (int(batch_id),),
    )
    out: list[Tuple[StickerRow, SeriesRow]] = []
    for r in cur.fetchall():
        srow = SeriesRow(
            id=int(r["series_id"]),
            name=str(r["series_name"]),
            enabled=bool(r["series_enabled"]),
            created_at=float(r["series_created"]),
        )
        out.append((_row_to_sticker(r), srow))
    return out


def count_stickers_in_series(con: sqlite3.Connection, series_id: int) -> int:
    cur = con.cursor()
    cur.execute("SELECT COUNT(1) as c FROM sticker WHERE series_id=?;", (int(series_id),))
    return int(cur.fetchone()["c"])  # type: ignore[index]


def list_series_counts(con: sqlite3.Connection) -> dict[int, int]:
    cur = con.cursor()
    cur.execute("SELECT series_id, COUNT(1) as c FROM sticker GROUP BY series_id;")
    return {int(r["series_id"]): int(r["c"]) for r in cur.fetchall()}


def get_sticker(con: sqlite3.Connection, sticker_id: int) -> Optional[Tuple[StickerRow, SeriesRow]]:
    cur = con.cursor()
    cur.execute(
        """
        SELECT
          sticker.*,
          series.name as series_name,
          series.enabled as series_enabled,
          series.created_at as series_created
        FROM sticker
        JOIN series ON series.id = sticker.series_id
        WHERE sticker.id=?;
        """,
        (int(sticker_id),),
    )
    r = cur.fetchone()
    if not r:
        return None
    srow = SeriesRow(
        id=int(r["series_id"]),
        name=str(r["series_name"]),
        enabled=bool(r["series_enabled"]),
        created_at=float(r["series_created"]),
    )
    return (_row_to_sticker(r), srow)


def create_sticker(
    con: sqlite3.Connection,
    series_id: int,
    filename: str,
    ext: str,
    tags: List[str],
    emb: Optional[np.ndarray],
    batch_id: Optional[int] = None,
    needs_tag: bool = False,
    enabled: bool = True,
) -> int:
    cur = con.cursor()
    ts = _now()

    emb_blob: Optional[bytes] = None
    emb_dim = 0
    if emb is not None:
        v = np.asarray(emb, dtype=np.float32)
        emb_dim = int(v.size)
        emb_blob = v.tobytes()

    cur.execute(
        """
        INSERT INTO sticker(series_id, filename, ext, enabled, needs_tag, batch_id, tags_json, emb_blob, emb_dim, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            int(series_id),
            str(filename),
            str(ext),
            1 if enabled else 0,
            1 if needs_tag else 0,
            int(batch_id) if batch_id is not None else None,
            _tags_to_json(tags),
            emb_blob,
            emb_dim,
            ts,
        ),
    )
    con.commit()
    return int(cur.lastrowid)


def set_sticker_enabled(con: sqlite3.Connection, sticker_id: int, enabled: bool) -> None:
    cur = con.cursor()
    cur.execute("UPDATE sticker SET enabled=? WHERE id=?;", (1 if enabled else 0, int(sticker_id)))
    con.commit()


def update_sticker_tags_and_emb(
    con: sqlite3.Connection,
    sticker_id: int,
    tags: List[str],
    emb: Optional[np.ndarray],
    needs_tag: bool,
) -> None:
    emb_blob: Optional[bytes] = None
    emb_dim = 0
    if emb is not None:
        v = np.asarray(emb, dtype=np.float32)
        emb_dim = int(v.size)
        emb_blob = v.tobytes()

    cur = con.cursor()
    cur.execute(
        """
        UPDATE sticker
        SET tags_json=?, emb_blob=?, emb_dim=?, needs_tag=?
        WHERE id=?;
        """,
        (_tags_to_json(tags), emb_blob, emb_dim, 1 if needs_tag else 0, int(sticker_id)),
    )
    con.commit()


def update_sticker_emb(
    con: sqlite3.Connection,
    sticker_id: int,
    emb: Optional[np.ndarray],
) -> None:
    emb_blob: Optional[bytes] = None
    emb_dim = 0
    if emb is not None:
        v = np.asarray(emb, dtype=np.float32)
        emb_dim = int(v.size)
        emb_blob = v.tobytes()

    cur = con.cursor()
    cur.execute(
        """
        UPDATE sticker
        SET emb_blob=?, emb_dim=?
        WHERE id=?;
        """,
        (emb_blob, emb_dim, int(sticker_id)),
    )
    con.commit()


def update_sticker_series(
    con: sqlite3.Connection,
    sticker_id: int,
    series_id: int,
    filename: Optional[str] = None,
) -> None:
    cur = con.cursor()
    if filename:
        cur.execute(
            "UPDATE sticker SET series_id=?, filename=?, batch_id=NULL WHERE id=?;",
            (int(series_id), str(filename), int(sticker_id)),
        )
    else:
        cur.execute(
            "UPDATE sticker SET series_id=?, batch_id=NULL WHERE id=?;",
            (int(series_id), int(sticker_id)),
        )
    con.commit()


def delete_sticker(con: sqlite3.Connection, sticker_id: int) -> bool:
    cur = con.cursor()
    cur.execute("DELETE FROM sticker WHERE id=?;", (int(sticker_id),))
    changed = cur.rowcount > 0
    con.commit()
    return bool(changed)


def bulk_update_tags_and_enabled(
    con: sqlite3.Connection,
    items: List[Tuple[int, List[str], bool]],
    emb_provider: Callable[[List[str]], np.ndarray],
    on_emb: Optional[Callable[[int, np.ndarray], None]] = None,
) -> int:
    """
    items: [(sticker_id, tags, enabled), ...]
    Return updated rows count.
    """
    cur = con.cursor()
    updated = 0
    for sid, tags, enabled in items:
        tags_clean = [str(t).strip() for t in tags if str(t).strip()]
        needs_tag = len(tags_clean) == 0
        emb = emb_provider(tags_clean)
        v = np.asarray(emb, dtype=np.float32)
        if on_emb is not None:
            on_emb(int(sid), v)
        cur.execute(
            """
            UPDATE sticker
            SET tags_json=?, emb_blob=?, emb_dim=?, enabled=?, needs_tag=?
            WHERE id=?;
            """,
            (
                _tags_to_json(tags_clean),
                v.tobytes(),
                int(v.size),
                1 if enabled else 0,
                1 if needs_tag else 0,
                int(sid),
            ),
        )
        updated += int(cur.rowcount)
    con.commit()
    return updated


def get_batch_ids_for_stickers(con: sqlite3.Connection, sticker_ids: Iterable[int]) -> List[int]:
    ids = [int(x) for x in sticker_ids if int(x) > 0]
    if not ids:
        return []
    cur = con.cursor()
    placeholders = ",".join(["?"] * len(ids))
    cur.execute(
        f"SELECT DISTINCT batch_id FROM sticker WHERE id IN ({placeholders}) AND batch_id IS NOT NULL;",
        ids,
    )
    return [int(r["batch_id"]) for r in cur.fetchall() if r["batch_id"] is not None]


def disable_sticker(con: sqlite3.Connection, sticker_id: int) -> None:
    set_sticker_enabled(con, int(sticker_id), False)


# =========================
# Pending / helpers
# =========================

def count_pending(con: sqlite3.Connection, batch_id: Optional[int] = None) -> int:
    cur = con.cursor()
    if batch_id is None:
        cur.execute("SELECT COUNT(1) as c FROM sticker WHERE needs_tag=1;")
        return int(cur.fetchone()["c"])  # type: ignore[index]
    cur.execute("SELECT COUNT(1) as c FROM sticker WHERE needs_tag=1 AND batch_id=?;", (int(batch_id),))
    return int(cur.fetchone()["c"])  # type: ignore[index]


def list_pending_stickers(
    con: sqlite3.Connection,
    batch_id: Optional[int] = None,
    limit: int = 2000,
) -> List[Tuple[StickerRow, SeriesRow]]:
    limit = max(1, min(int(limit), 5000))
    cur = con.cursor()
    if batch_id is None:
        cur.execute(
            """
            SELECT
              sticker.*,
              series.name as series_name,
              series.enabled as series_enabled,
              series.created_at as series_created
            FROM sticker
            JOIN series ON series.id = sticker.series_id
            WHERE sticker.needs_tag=1
            ORDER BY sticker.id DESC
            LIMIT ?;
            """,
            (limit,),
        )
    else:
        cur.execute(
            """
            SELECT
              sticker.*,
              series.name as series_name,
              series.enabled as series_enabled,
              series.created_at as series_created
            FROM sticker
            JOIN series ON series.id = sticker.series_id
            WHERE sticker.needs_tag=1 AND sticker.batch_id=?
            ORDER BY sticker.id DESC
            LIMIT ?;
            """,
            (int(batch_id), limit),
        )

    out: list[Tuple[StickerRow, SeriesRow]] = []
    for r in cur.fetchall():
        srow = SeriesRow(
            id=int(r["series_id"]),
            name=str(r["series_name"]),
            enabled=bool(r["series_enabled"]),
            created_at=float(r["series_created"]),
        )
        out.append((_row_to_sticker(r), srow))
    return out


# =========================
# Embeddings
# =========================

def blob_to_vec(blob: Optional[bytes], dim: int) -> Optional[np.ndarray]:
    if blob is None:
        return None
    if dim <= 0:
        dim = len(blob) // 4
    return np.frombuffer(blob, dtype=np.float32, count=int(dim)).copy()
