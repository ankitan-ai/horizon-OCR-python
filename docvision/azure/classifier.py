"""
Smart Document Classifier using GPT-5-nano.

Performs a lightweight classification call before the main extraction
to determine document type and complexity.  The result drives:

* Which ``_DOC_TYPE_HINTS`` prompt is used for GPT Vision KIE.
* Which Azure DI model is selected (``prebuilt-invoice`` vs ``prebuilt-layout``).
* Which GPT deployment handles extraction (scaled by complexity).
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ── Classification result ───────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Result of the document classification step."""

    document_type: str = "auto"
    """Detected document type (bol, invoice, receipt, delivery_ticket, other)."""

    complexity: str = "medium"
    """Document complexity: simple, medium, complex."""

    recommended_gpt_deployment: Optional[str] = None
    """GPT deployment to use for extraction (set by routing table)."""

    recommended_di_model: Optional[str] = None
    """Azure DI model to use (set by routing table)."""

    confidence: float = 0.0
    """Classifier's confidence in the classification."""

    raw_response: str = ""
    """Raw GPT response for debugging."""


# ── Routing table ───────────────────────────────────────────────────────────

# Maps (doc_type, complexity) → recommended GPT deployment.
# Available deployments: gpt-5.2, gpt-5-nano, gpt-5-mini, gpt-4o-mini, gpt-4.1-mini
_GPT_ROUTING: Dict[str, Dict[str, str]] = {
    "simple": {
        "invoice": "gpt-4o-mini",
        "receipt": "gpt-4o-mini",
        "bol": "gpt-4o-mini",
        "delivery_ticket": "gpt-4o-mini",
        "other": "gpt-4o-mini",
    },
    "medium": {
        "invoice": "gpt-4o-mini",
        "receipt": "gpt-4o-mini",
        "bol": "gpt-4.1-mini",       # BOLs are typically denser
        "delivery_ticket": "gpt-4o-mini",
        "other": "gpt-4o-mini",
    },
    "complex": {
        "invoice": "gpt-4.1-mini",
        "receipt": "gpt-4o-mini",     # Receipts are rarely truly complex
        "bol": "gpt-5-mini",          # Dense multi-section BOLs - use gpt-5-mini
        "delivery_ticket": "gpt-4.1-mini",
        "other": "gpt-4.1-mini",
    },
}

# Maps doc_type → recommended Azure DI model
_DI_ROUTING: Dict[str, str] = {
    "invoice": "prebuilt-invoice",
    "receipt": "prebuilt-layout",
    "bol": "prebuilt-layout",
    "delivery_ticket": "prebuilt-layout",
    "other": "prebuilt-layout",
}


# ── Classifier prompt ──────────────────────────────────────────────────────

_CLASSIFIER_SYSTEM = """\
You are a document classifier. Given a document image, determine:
1. The document type: invoice, bol, receipt, delivery_ticket, or other
2. The complexity: simple, medium, or complex

Complexity guidelines:
- simple: clean, typed, single-column, few fields, standard layout
- medium: multiple sections, some tables, standard formatting
- complex: dense multi-column layout, many tables, mixed handwritten/printed, \
poor scan quality, multiple pages of data, non-standard format

Return ONLY a JSON object with exactly two keys:
{"type": "<document_type>", "complexity": "<complexity>"}

Do NOT include any other text.
"""


# ── Classifier class ───────────────────────────────────────────────────────

class DocumentClassifier:
    """
    Lightweight document classifier using GPT-nano.

    Usage::

        classifier = DocumentClassifier(azure_config)
        result = classifier.classify(page_image)
        # result.document_type  → "invoice"
        # result.complexity     → "complex"
        # result.recommended_gpt_deployment → "gpt-4.1-mini"
        # result.recommended_di_model       → "prebuilt-invoice"
    """

    # The nano deployment used only for classification (cheap + fast)
    CLASSIFIER_DEPLOYMENT = "gpt-5-nano"

    def __init__(self, azure_config, cost_tracker=None) -> None:
        from docvision.config import AzureConfig

        self._config: AzureConfig = azure_config
        self._client = None
        self.cost_tracker = cost_tracker
        self._default_deployment = azure_config.openai_deployment

    # ── lazy client ──────────────────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            from openai import AzureOpenAI
            import httpx
            import certifi

            self._client = AzureOpenAI(
                azure_endpoint=self._config.openai_endpoint,
                api_key=self._config.openai_key,
                api_version=self._config.openai_api_version,
                http_client=httpx.Client(verify=certifi.where()),
            )
        return self._client

    # ── public API ───────────────────────────────────────────────────

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """
        Classify a document image and return routing recommendations.

        Args:
            image: First page of the document (BGR numpy array).

        Returns:
            ClassificationResult with type, complexity, and model recommendations.
        """
        if not self._config.is_openai_ready:
            logger.warning("OpenAI not configured — skipping classification")
            return ClassificationResult()

        b64 = self._encode_image_b64(image)

        messages = [
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this document."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "low",  # low-res is enough for classification
                        },
                    },
                ],
            },
        ]

        t0 = time.perf_counter()
        response = self._call_classifier(messages)
        if response is None:
            return ClassificationResult()

        elapsed = time.perf_counter() - t0
        choice = response.choices[0] if response.choices else None
        raw = self._extract_content(choice)

        # If image-based call returned empty, retry with text-only prompt
        if not raw:
            logger.warning(
                f"Classifier returned empty content (tokens used: "
                f"{response.usage.completion_tokens if response.usage else '?'}) "
                f"— retrying text-only"
            )
            text_messages = [
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Classify this as a standard business document. "
                        'Respond with exactly: {"type": "other", "complexity": "medium"} '
                        "or pick a more specific type if you can infer one."
                    ),
                },
            ]
            retry_resp = self._call_classifier(text_messages)
            if retry_resp is not None:
                retry_choice = (
                    retry_resp.choices[0] if retry_resp.choices else None
                )
                raw = self._extract_content(retry_choice)
                # Merge usage for cost tracking
                if retry_resp.usage and response.usage:
                    response.usage.prompt_tokens += retry_resp.usage.prompt_tokens
                    response.usage.completion_tokens += (
                        retry_resp.usage.completion_tokens
                    )

        usage = response.usage

        logger.info(
            f"Document classified in {elapsed:.2f}s  "
            f"(tokens: {usage.prompt_tokens if usage else '?'}+"
            f"{usage.completion_tokens if usage else '?'})"
        )

        # Record cost
        if self.cost_tracker and usage:
            self.cost_tracker.record_gpt_call(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                deployment=self.CLASSIFIER_DEPLOYMENT,
                latency=elapsed,
            )

        # Parse response
        result = self._parse_response(raw)
        result.raw_response = raw

        # Apply routing tables
        result.recommended_gpt_deployment = self._route_gpt(
            result.document_type, result.complexity
        )
        result.recommended_di_model = _DI_ROUTING.get(
            result.document_type, "prebuilt-layout"
        )

        logger.info(
            f"Classification: type={result.document_type}, "
            f"complexity={result.complexity} → "
            f"GPT={result.recommended_gpt_deployment}, "
            f"DI={result.recommended_di_model}"
        )

        return result

    # ── internals ────────────────────────────────────────────────────

    def _call_classifier(self, messages: list) -> object | None:
        """Send a chat completion request with model-quirk fallbacks.

        GPT-5-nano uses internal reasoning tokens that count against
        ``max_completion_tokens``.  We set a generous budget (1024) so
        the model has room for both reasoning and the visible JSON
        output.  The actual visible output is tiny (~30 tokens).

        Returns the response object or *None* if every attempt fails.
        """
        try:
            try:
                return self.client.chat.completions.create(
                    model=self.CLASSIFIER_DEPLOYMENT,
                    messages=messages,
                    max_completion_tokens=1024,
                    temperature=0.0,
                )
            except Exception as inner:
                inner_msg = str(inner)
                if "max_completion_tokens" in inner_msg:
                    return self.client.chat.completions.create(
                        model=self.CLASSIFIER_DEPLOYMENT,
                        messages=messages,
                        max_tokens=1024,
                        temperature=0.0,
                    )
                elif "temperature" in inner_msg:
                    return self.client.chat.completions.create(
                        model=self.CLASSIFIER_DEPLOYMENT,
                        messages=messages,
                        max_completion_tokens=1024,
                    )
                else:
                    raise
        except Exception as exc:
            logger.warning(f"Classification call failed ({exc}) — using defaults")
            return None

    @staticmethod
    def _extract_content(choice) -> str:
        """Pull text content from a chat completion choice.

        Checks ``message.content``, ``tool_calls``, and ``refusal``.
        Logs diagnostics when content is unexpectedly empty.
        """
        if choice is None or choice.message is None:
            return ""
        raw = (choice.message.content or "").strip()
        if not raw:
            # Some models put output in tool_calls instead of content
            tool_calls = getattr(choice.message, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    if fn and getattr(fn, "arguments", None):
                        logger.info("Classifier output found in tool_calls")
                        return fn.arguments.strip()

            refusal = getattr(choice.message, "refusal", None)
            if refusal:
                logger.warning(f"Classifier refused: {refusal}")
            finish = getattr(choice, "finish_reason", None)
            logger.warning(
                f"Classifier empty content — finish_reason={finish}, "
                f"message keys: {list(vars(choice.message).keys())}"
            )
        return raw

    @staticmethod
    def _encode_image_b64(image: np.ndarray) -> str:
        success, buf = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image to PNG")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _parse_response(raw: str) -> ClassificationResult:
        """Parse the classifier's JSON response."""
        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse classifier response: {raw}")
                    return ClassificationResult()
            else:
                logger.warning(f"Could not parse classifier response: {raw}")
                return ClassificationResult()

        doc_type = str(data.get("type", "other")).lower().strip()
        complexity = str(data.get("complexity", "medium")).lower().strip()

        # Validate
        valid_types = {"invoice", "bol", "receipt", "delivery_ticket", "other"}
        valid_complexities = {"simple", "medium", "complex"}

        if doc_type not in valid_types:
            doc_type = "other"
        if complexity not in valid_complexities:
            complexity = "medium"

        return ClassificationResult(
            document_type=doc_type,
            complexity=complexity,
            confidence=0.90,
        )

    def _route_gpt(self, doc_type: str, complexity: str) -> str:
        """Look up the recommended GPT deployment from the routing table."""
        complexity_map = _GPT_ROUTING.get(complexity, _GPT_ROUTING["medium"])
        return complexity_map.get(doc_type, self._default_deployment)
