"""
Azure API response cache for DocVision.

Caches Document Intelligence and GPT Vision responses to avoid
re-processing the same document / page.  Uses a content-hash key
so identical images always hit the cache regardless of filename.

Storage: JSON files under ``.cache/azure/`` (configurable).
Thread-safe via a simple lock.

Usage::

    cache = ResponseCache()                         # uses .cache/azure/
    cache = ResponseCache(cache_dir="my_cache")

    key = cache.make_key(image_bytes, service="di", model="prebuilt-layout")
    hit = cache.get(key)                            # dict | None
    if hit is None:
        result = provider.analyze(...)
        cache.put(key, result)
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class ResponseCache:
    """
    Content-addressed file-based cache for Azure API responses.

    Each cached entry is a JSON file named ``<hash>.json`` stored under
    the configured cache directory.  A lightweight in-memory index
    avoids repeated disk reads during a single session.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/azure",
        enabled: bool = True,
        max_entries: int = 500,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.max_entries = max_entries
        self._index: Dict[str, Path] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    # ── key generation ───────────────────────────────────────────────────

    @staticmethod
    def make_key(
        content: bytes,
        service: str = "di",
        model: str = "",
        extra: str = "",
    ) -> str:
        """
        Compute a deterministic cache key from content + service params.

        Args:
            content:  Raw bytes (image PNG or PDF bytes).
            service:  ``"di"`` or ``"gpt"``.
            model:    Model / deployment name.
            extra:    Any extra differentiator (e.g. document_type for GPT).

        Returns:
            Hex digest string used as cache key / filename.
        """
        h = hashlib.sha256()
        h.update(content)
        h.update(service.encode())
        h.update(model.encode())
        if extra:
            h.update(extra.encode())
        return h.hexdigest()

    # ── core operations ──────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Look up a cached response.  Returns ``None`` on miss."""
        if not self.enabled:
            return None

        with self._lock:
            path = self._index.get(key)

        if path is None or not path.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            logger.debug(f"[Cache HIT] key={key[:12]}…")
            return data.get("response")
        except Exception as exc:
            logger.warning(f"[Cache] Failed to read {path}: {exc}")
            self._misses += 1
            return None

    def put(self, key: str, response: Any, metadata: Optional[Dict] = None) -> None:
        """Store a response in the cache."""
        if not self.enabled:
            return

        path = self.cache_dir / f"{key}.json"
        entry = {
            "key": key,
            "cached_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "metadata": metadata or {},
            "response": self._serialise(response),
        }

        try:
            path.write_text(
                json.dumps(entry, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            with self._lock:
                self._index[key] = path
            logger.debug(f"[Cache PUT] key={key[:12]}…")
        except Exception as exc:
            logger.warning(f"[Cache] Failed to write {path}: {exc}")

        # Evict oldest entries if over limit
        self._maybe_evict()

    def has(self, key: str) -> bool:
        """Check if key exists without deserialising."""
        if not self.enabled:
            return False
        with self._lock:
            path = self._index.get(key)
        return path is not None and path.exists()

    # ── stats ────────────────────────────────────────────────────────────

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._index)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total) if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "enabled": self.enabled,
            "entries": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
            "cache_dir": str(self.cache_dir),
        }

    # ── management ───────────────────────────────────────────────────────

    def clear(self) -> int:
        """Delete all cached entries.  Returns count of entries removed."""
        count = 0
        with self._lock:
            for path in self._index.values():
                try:
                    path.unlink(missing_ok=True)
                    count += 1
                except Exception:
                    pass
            self._index.clear()
        self._hits = 0
        self._misses = 0
        logger.info(f"[Cache] Cleared {count} entries")
        return count

    # ── internals ────────────────────────────────────────────────────────

    def _load_index(self) -> None:
        """Scan cache_dir and populate the in-memory index."""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            key = p.stem  # filename without .json
            self._index[key] = p
            count += 1
        if count:
            logger.info(f"[Cache] Loaded {count} cached entries from {self.cache_dir}")

    def _maybe_evict(self) -> None:
        """Remove oldest entries if cache exceeds max_entries."""
        with self._lock:
            if len(self._index) <= self.max_entries:
                return

            # Sort by file modification time and remove oldest
            entries = sorted(
                self._index.items(),
                key=lambda kv: kv[1].stat().st_mtime if kv[1].exists() else 0,
            )
            to_remove = len(entries) - self.max_entries
            for key, path in entries[:to_remove]:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
                del self._index[key]

            if to_remove > 0:
                logger.info(f"[Cache] Evicted {to_remove} old entries")

    @staticmethod
    def _serialise(obj: Any) -> Any:
        """
        Make an object JSON-serialisable.

        Handles Pydantic models, dataclasses, numpy arrays, and plain dicts.
        Lists of such objects are processed recursively.
        """
        if obj is None:
            return None

        # Pydantic model
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")

        # Dataclass
        if hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict

            return asdict(obj)

        # Dict — recurse values
        if isinstance(obj, dict):
            return {k: ResponseCache._serialise(v) for k, v in obj.items()}

        # List — recurse elements
        if isinstance(obj, list):
            return [ResponseCache._serialise(v) for v in obj]

        # numpy ndarray → skip (images shouldn't be cached as arrays)
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return None
        except ImportError:
            pass

        return obj
