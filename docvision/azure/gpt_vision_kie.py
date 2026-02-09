"""
GPT Vision key-information extraction (KIE) provider for DocVision.

Sends a document image (plus optional OCR text) to Azure OpenAI GPT-4o / GPT-5
with a structured-output prompt and maps the response into ``List[Field]``,
ready for the rank-and-fuse pipeline.

This replaces two local models:  Donut  +  LayoutLMv3.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from docvision.config import AzureConfig
from docvision.types import (
    BoundingBox,
    Candidate,
    Field,
    FieldStatus,
    SourceEngine,
)


# ── Prompt templates per document type ──────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a document-understanding assistant.
Extract every key field from the document image (and optional OCR text) below.
Return a **single JSON object** where each key is the field name (snake_case)
and each value is the extracted value as a string.

Rules:
- Use snake_case for ALL field names (e.g. "bill_of_lading_number", "total_amount").
- Dates must be in YYYY-MM-DD format.
- Currency amounts must include only digits, dots, and an optional leading symbol ($, €, £).
- If a field is not present, omit it entirely.
- Do NOT wrap the JSON in markdown fences.
- Do NOT include commentary — return ONLY the JSON object.
"""

_DOC_TYPE_HINTS: Dict[str, str] = {
    "bol": (
        "This is a Bill of Lading. Look for fields like: "
        "bol_number, shipper_name, shipper_address, consignee_name, "
        "consignee_address, carrier_name, pro_number, pickup_date, "
        "delivery_date, origin_city, origin_state, destination_city, "
        "destination_state, weight, pieces, freight_class, "
        "commodity_description, special_instructions, total_charges."
    ),
    "invoice": (
        "This is an Invoice. Look for fields like: "
        "invoice_number, invoice_date, due_date, vendor_name, "
        "vendor_address, bill_to_name, bill_to_address, po_number, "
        "subtotal, tax_amount, total_amount, currency, payment_terms, "
        "line_items (item, quantity, unit_price, amount)."
    ),
    "receipt": (
        "This is a Receipt. Look for fields like: "
        "store_name, store_address, receipt_date, receipt_number, "
        "line_items (item, quantity, price), subtotal, tax, total, "
        "payment_method, card_last_four."
    ),
    "delivery_ticket": (
        "This is a Delivery Ticket. Look for fields like: "
        "ticket_number, delivery_date, customer_name, customer_address, "
        "driver_name, truck_number, product_description, quantity, "
        "unit, gross_weight, tare_weight, net_weight, temperature."
    ),
}


class GPTVisionExtractor:
    """
    Cloud-based KIE via Azure OpenAI GPT Vision.

    Usage::

        extractor = GPTVisionExtractor(config.azure)
        fields = extractor.extract(page_image, page_num=1)
        # fields is List[Field] ready for RankAndFuse
    """

    def __init__(self, azure_config: AzureConfig) -> None:
        if not azure_config.is_openai_ready:
            raise ValueError(
                "Azure OpenAI is not configured. "
                "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, and "
                "AZURE_OPENAI_DEPLOYMENT in your .env file or config YAML."
            )

        self._config = azure_config
        self._client = None  # lazy
        logger.info(
            "GPTVisionExtractor initialised  "
            f"(deployment={azure_config.openai_deployment}, "
            f"endpoint=…{azure_config.openai_endpoint[-30:]})"
        )

    # ── lazy client ──────────────────────────────────────────────────────────

    @property
    def client(self):
        """Lazy-initialise the Azure OpenAI SDK client."""
        if self._client is None:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=self._config.openai_endpoint,
                api_key=self._config.openai_key,
                api_version=self._config.openai_api_version,
            )
            logger.debug("AzureOpenAI client created")
        return self._client

    # ── public API ───────────────────────────────────────────────────────────

    def extract(
        self,
        image: np.ndarray,
        page_num: int = 1,
        ocr_text: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> List[Field]:
        """
        Extract key fields from a document page image using GPT Vision.

        Args:
            image:          BGR or RGB numpy array (H×W×3).
            page_num:       1-based page number.
            ocr_text:       Optional pre-extracted OCR text to include in prompt.
            document_type:  Document type hint (``auto``, ``bol``, ``invoice``, …).
                            Overrides ``AzureConfig.document_type`` if given.

        Returns:
            ``List[Field]`` ready for the rank-and-fuse pipeline.
        """
        doc_type = document_type or self._config.document_type or "auto"
        messages = self._build_messages(image, ocr_text, doc_type)

        t0 = time.perf_counter()
        logger.info(
            f"Sending page {page_num} to GPT Vision KIE "
            f"(deployment={self._config.openai_deployment}, type={doc_type})"
        )

        response = self.client.chat.completions.create(
            model=self._config.openai_deployment,
            messages=messages,
            max_tokens=self._config.gpt_max_tokens,
            temperature=self._config.gpt_temperature,
        )

        elapsed = time.perf_counter() - t0
        raw_text = (response.choices[0].message.content or "").strip()
        usage = response.usage

        logger.info(
            f"GPT Vision response in {elapsed:.2f}s  "
            f"(tokens: prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens})"
            if usage
            else f"GPT Vision response in {elapsed:.2f}s"
        )

        # Parse the JSON and convert to Field objects
        extracted = self._parse_response(raw_text)
        fields = self._dict_to_fields(extracted, page_num)

        logger.info(f"GPT Vision extracted {len(fields)} fields from page {page_num}")
        return fields

    # ── message construction ─────────────────────────────────────────────────

    def _build_messages(
        self,
        image: np.ndarray,
        ocr_text: Optional[str],
        document_type: str,
    ) -> List[Dict[str, Any]]:
        """Build the chat messages array for the API call."""
        system = _SYSTEM_PROMPT

        # Add document-type hint if available
        doc_hint = _DOC_TYPE_HINTS.get(document_type.lower(), "")
        if doc_hint:
            system += f"\n\n{doc_hint}"

        # Build user message content parts
        user_parts: List[Dict[str, Any]] = []

        # Image part
        b64 = self._encode_image_b64(image)
        user_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            }
        )

        # Optional OCR text part
        text_instruction = "Extract all key-value fields from this document image."
        if ocr_text:
            text_instruction += (
                "\n\nHere is the OCR text already extracted from this page "
                "(use it to improve accuracy):\n\n"
                f"```\n{ocr_text[:6000]}\n```"
            )
        user_parts.insert(0, {"type": "text", "text": text_instruction})

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_parts},
        ]

    # ── image encoding ───────────────────────────────────────────────────────

    @staticmethod
    def _encode_image_b64(image: np.ndarray) -> str:
        """Encode numpy image to base64 PNG string for the API."""
        success, buf = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image to PNG")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ── response parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """
        Parse GPT response text into a dictionary.

        Handles:
        - Clean JSON
        - JSON wrapped in markdown fences
        - Partial / malformed JSON (best-effort)
        """
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        # Try direct parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON object within the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        logger.warning("GPT Vision: could not parse JSON from response")
        logger.debug(f"Raw response: {raw[:500]}")
        return {}

    # ── Field conversion ─────────────────────────────────────────────────────

    def _dict_to_fields(
        self,
        data: Dict[str, Any],
        page_num: int,
        prefix: str = "",
    ) -> List[Field]:
        """
        Recursively convert extracted dict → ``List[Field]``.

        Mirrors the pattern used by ``DonutRunner._dict_to_fields``
        so the fuser treats them identically.
        """
        fields: List[Field] = []

        for key, value in data.items():
            field_name = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                fields.extend(
                    self._dict_to_fields(value, page_num, f"{field_name}.")
                )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        fields.extend(
                            self._dict_to_fields(
                                item, page_num, f"{field_name}[{i}]."
                            )
                        )
                    else:
                        fields.append(
                            self._create_field(f"{field_name}[{i}]", item, page_num)
                        )
            else:
                fields.append(self._create_field(field_name, value, page_num))

        return fields

    @staticmethod
    def _create_field(
        name: str,
        value: Any,
        page_num: int,
        confidence: float = 0.90,
    ) -> Field:
        """
        Create a single ``Field`` from a GPT-extracted key-value pair.

        GPT doesn't provide per-field confidence so we use a default of
        0.90 (high, because GPT-4o vision is strong at KIE).  The fuser
        will re-rank against other sources.
        """
        # Infer data type
        str_val = str(value)
        data_type = "string"
        if isinstance(value, (int, float)):
            data_type = "number"
        elif _looks_like_date(str_val):
            data_type = "date"
        elif _looks_like_currency(str_val):
            data_type = "currency"

        candidate = Candidate(
            source=SourceEngine.GPT_VISION,
            value=value,
            confidence=confidence,
            page=page_num,
        )

        status = (
            FieldStatus.CONFIDENT
            if confidence >= 0.8
            else FieldStatus.SINGLE_SOURCE
            if confidence >= 0.5
            else FieldStatus.UNCERTAIN
        )

        return Field(
            name=name,
            value=value,
            data_type=data_type,
            confidence=confidence,
            status=status,
            page=page_num,
            chosen_source=SourceEngine.GPT_VISION,
            candidates=[candidate],
        )


# ── utility helpers (module-level) ──────────────────────────────────────────

def _looks_like_date(value: str) -> bool:
    patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}-\d{2}-\d{4}",
        r"\w+ \d{1,2}, \d{4}",
    ]
    return any(re.search(p, value) for p in patterns)


def _looks_like_currency(value: str) -> bool:
    patterns = [
        r"[$€£¥]\s*[\d,]+\.?\d*",
        r"[\d,]+\.?\d*\s*[$€£¥]",
        r"\d+[.,]\d{2}$",
    ]
    return any(re.search(p, value) for p in patterns)
