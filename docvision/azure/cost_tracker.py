"""
Azure API cost tracking for DocVision.

Logs every Azure API call (Document Intelligence + GPT Vision) with:
  - Timestamp, service, model/deployment
  - Token usage (GPT) and page count (DI)
  - Estimated cost in USD
  - Cumulative session totals

Thread-safe — the orchestrator may run pages concurrently.

Usage::

    tracker = CostTracker()
    tracker.record_di_call(pages=3, model="prebuilt-layout", latency=2.1)
    tracker.record_gpt_call(prompt_tokens=800, completion_tokens=200,
                            deployment="gpt-4o-mini", latency=1.5)
    print(tracker.summary())
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

from loguru import logger

from docvision.azure.pricing_service import (
    get_pricing_service,
    AzurePricingService,
    PricingUnavailableError,
)


@dataclass
class APICallRecord:
    """A single recorded Azure API call."""

    timestamp: str
    service: str  # "doc_intelligence" or "gpt_vision"
    model: str  # e.g. "prebuilt-layout" or "gpt-4o-mini"
    pages: int = 0  # DI: number of pages analysed
    prompt_tokens: int = 0  # GPT: input tokens
    completion_tokens: int = 0  # GPT: output tokens
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    doc_id: str = ""
    cached: bool = False  # True if result was served from cache (cost=0)


@dataclass
class CostTracker:
    """
    Accumulates Azure API call records and computes ACTUAL costs.

    Completely in-memory; call :meth:`summary` or :meth:`to_dict` to
    serialise.  Thread-safe via a reentrant lock.
    
    Pricing is fetched LIVE from Azure Retail Prices API.
    NO FALLBACKS - raises PricingUnavailableError if API is unreachable.
    """

    records: List[APICallRecord] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    max_records: int = 5000  # FIFO eviction to prevent unbounded growth
    _pricing_service: Optional[AzurePricingService] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the pricing service."""
        if self._pricing_service is None:
            self._pricing_service = get_pricing_service()

    # ── recording ────────────────────────────────────────────────────────

    def _get_di_cost(self, model: str) -> float:
        """
        Get DI cost per page from Azure Retail Prices API.
        
        Raises:
            PricingUnavailableError: If pricing cannot be fetched
        """
        return self._pricing_service.get_di_cost_per_page_sync(model)

    def _get_gpt_costs(self, deployment: str) -> tuple[float, float]:
        """
        Get GPT costs per 1K tokens from Azure Retail Prices API.
        
        Raises:
            PricingUnavailableError: If pricing cannot be fetched
        """
        return self._pricing_service.get_gpt_costs_sync(deployment)

    def record_di_call(
        self,
        pages: int = 1,
        model: str = "prebuilt-layout",
        latency: float = 0.0,
        doc_id: str = "",
        cached: bool = False,
    ) -> APICallRecord:
        """
        Record an Azure Document Intelligence API call.
        
        Pricing is fetched LIVE from Azure Retail Prices API.
        
        Raises:
            PricingUnavailableError: If pricing cannot be fetched from API
        """
        if cached:
            cost = 0.0
        else:
            per_page = self._get_di_cost(model)
            cost = pages * per_page

        record = APICallRecord(
            timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(),
            service="doc_intelligence",
            model=model,
            pages=pages,
            latency_seconds=round(latency, 3),
            estimated_cost_usd=round(cost, 6),
            doc_id=doc_id,
            cached=cached,
        )

        with self._lock:
            self.records.append(record)
            # Evict oldest records if over cap
            if len(self.records) > self.max_records:
                self.records = self.records[-self.max_records:]

        if cached:
            logger.info(
                f"[CostTracker] DI call (CACHED) — {pages} page(s), $0.00"
            )
        else:
            logger.info(
                f"[CostTracker] DI call — {pages} page(s), "
                f"model={model}, cost=${cost:.4f}, latency={latency:.2f}s"
            )

        return record

    def record_gpt_call(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        deployment: str = "gpt-4o-mini",
        latency: float = 0.0,
        doc_id: str = "",
        cached: bool = False,
    ) -> APICallRecord:
        """
        Record an Azure OpenAI GPT Vision API call.
        
        Pricing is fetched LIVE from Azure Retail Prices API.
        
        Raises:
            PricingUnavailableError: If pricing cannot be fetched from API
        """
        if cached:
            cost = 0.0
        else:
            input_rate, output_rate = self._get_gpt_costs(deployment)
            cost = (prompt_tokens / 1000) * input_rate + (
                completion_tokens / 1000
            ) * output_rate

        record = APICallRecord(
            timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(),
            service="gpt_vision",
            model=deployment,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=round(latency, 3),
            estimated_cost_usd=round(cost, 6),
            doc_id=doc_id,
            cached=cached,
        )

        with self._lock:
            self.records.append(record)
            if len(self.records) > self.max_records:
                self.records = self.records[-self.max_records:]

        total_tokens = prompt_tokens + completion_tokens
        if cached:
            logger.info(
                f"[CostTracker] GPT call (CACHED) — {total_tokens} tokens, $0.00"
            )
        else:
            logger.info(
                f"[CostTracker] GPT call — {total_tokens} tokens "
                f"(in={prompt_tokens}, out={completion_tokens}), "
                f"deployment={deployment}, cost=${cost:.6f}, latency={latency:.2f}s"
            )

        return record

    # ── queries ──────────────────────────────────────────────────────────

    @property
    def total_calls(self) -> int:
        with self._lock:
            return len(self.records)

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return round(sum(r.estimated_cost_usd for r in self.records), 6)

    @property
    def total_di_calls(self) -> int:
        with self._lock:
            return sum(1 for r in self.records if r.service == "doc_intelligence")

    @property
    def total_gpt_calls(self) -> int:
        with self._lock:
            return sum(1 for r in self.records if r.service == "gpt_vision")

    @property
    def total_pages_analysed(self) -> int:
        with self._lock:
            return sum(r.pages for r in self.records if r.service == "doc_intelligence")

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(
                r.prompt_tokens + r.completion_tokens
                for r in self.records
                if r.service == "gpt_vision"
            )

    @property
    def cache_hit_count(self) -> int:
        with self._lock:
            return sum(1 for r in self.records if r.cached)

    @property
    def cost_saved_by_cache(self) -> float:
        """Hypothetical cost of cached calls if they had hit the API."""
        with self._lock:
            total = 0.0
            for r in self.records:
                if not r.cached:
                    continue
                if r.service == "doc_intelligence":
                    per_page = self._get_di_cost(r.model)
                    total += r.pages * per_page
                elif r.service == "gpt_vision":
                    input_rate, output_rate = self._get_gpt_costs(r.model)
                    total += (r.prompt_tokens / 1000) * input_rate + (
                        r.completion_tokens / 1000
                    ) * output_rate
            return round(total, 6)

    # ── summary / serialisation ──────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║          Azure API Cost Summary                 ║",
            "╠══════════════════════════════════════════════════╣",
            f"║  Total API calls:        {self.total_calls:>6}                 ║",
            f"║  · Doc Intelligence:     {self.total_di_calls:>6}                 ║",
            f"║  · GPT Vision:           {self.total_gpt_calls:>6}                 ║",
            f"║  Pages analysed (DI):    {self.total_pages_analysed:>6}                 ║",
            f"║  Tokens used (GPT):     {self.total_tokens:>7}                 ║",
            f"║  Cache hits:             {self.cache_hit_count:>6}                 ║",
            f"║  Estimated cost:      ${self.total_cost_usd:>8.4f}                ║",
            f"║  Saved by cache:      ${self.cost_saved_by_cache:>8.4f}                ║",
            "╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise tracker state for JSON / API responses."""
        # Grab snapshot of records first to avoid holding lock while computing
        with self._lock:
            records_copy = list(self.records)
        
        # Compute stats without lock (using the snapshot)
        total_calls = len(records_copy)
        di_calls = sum(1 for r in records_copy if r.service == "doc_intelligence")
        gpt_calls = sum(1 for r in records_copy if r.service == "gpt_vision")
        pages = sum(r.pages for r in records_copy if r.service == "doc_intelligence")
        tokens = sum(r.prompt_tokens + r.completion_tokens for r in records_copy if r.service == "gpt_vision")
        cache_hits = sum(1 for r in records_copy if r.cached)
        cost = round(sum(r.estimated_cost_usd for r in records_copy), 6)
        
        # Compute saved-by-cache
        saved = 0.0
        for r in records_copy:
            if not r.cached:
                continue
            if r.service == "doc_intelligence":
                per_page = self._get_di_cost(r.model)
                saved += r.pages * per_page
            elif r.service == "gpt_vision":
                input_rate, output_rate = self._get_gpt_costs(r.model)
                saved += (r.prompt_tokens / 1000) * input_rate + (r.completion_tokens / 1000) * output_rate
        
        return {
            "total_calls": total_calls,
            "total_di_calls": di_calls,
            "total_gpt_calls": gpt_calls,
            "total_pages_analysed": pages,
            "total_tokens": tokens,
            "cache_hits": cache_hits,
            "estimated_cost_usd": cost,
            "cost_saved_by_cache_usd": round(saved, 6),
            "records": [
                {
                    "timestamp": r.timestamp,
                    "service": r.service,
                    "model": r.model,
                    "pages": r.pages,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "latency_seconds": r.latency_seconds,
                    "estimated_cost_usd": r.estimated_cost_usd,
                    "doc_id": r.doc_id,
                    "cached": r.cached,
                }
                for r in records_copy
            ],
        }

    def reset(self) -> None:
        """Clear all recorded calls."""
        with self._lock:
            self.records.clear()
        logger.info("[CostTracker] Reset — all records cleared")
