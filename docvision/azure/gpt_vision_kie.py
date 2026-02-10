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

Return ONLY a single JSON object. No markdown, no commentary.

GLOBAL RULES
- All field names use snake_case.
- Dates: YYYY-MM-DD. Timestamps: YYYY-MM-DD HH:MM:SS (24-hour).
- Currency values: digits + optional decimal + optional symbol ($, €, £).
- Omit missing fields; never invent values.
- Do not include OCR text or reasoning in the output.
- Values should be strings unless representing arrays of rows (line_items).

TOP-LEVEL JSON KEYS (when available)
- id, metadata, page_count, pages[], tables[], fields[], validation
- normalized: { document_type, header{}, line_items[], totals{}, line_items_secondary[] }

STRUCTURE RULES (VERY STRICT)
1) Never mix header metadata with line-item data.
   - Header metadata goes to normalized.header.
   - Row/table data goes to normalized.line_items[] (and optional normalized.line_items_secondary[]).

2) Table-like data MUST be arrays:
   - "normalized.line_items": [ { ... }, { ... } ]
   - Never emit flat, numbered keys for rows.

3) Each line_items entry must ONLY contain row-level fields (examples):
   - item_description, product_description, sku, quantity, unit, unit_price, amount,
     gross_gallons, net_gallons, temperature, temperature_unit, api_gravity,
     compartment_number, taxes, service_date, etc.

4) Header fields must NEVER appear inside line_items (examples):
   - document_type, invoice_number, bol_number, receipt_number, po_number,
     vendor_name, customer_name, addresses, dates/timestamps, totals, account_numbers.

5) If totals exist, return:
   normalized.totals = { subtotal, tax_amount, total_amount, gross_gallons, net_gallons }

6) If multiple distinct tables exist, add:
   normalized.line_items_secondary: [ ... ]

7) When possible, include provenance on each line item:
   _evidence: { table_id, cell_refs: [{row, col}], page }

DOCUMENT-TYPE HINTS (if provided; do NOT ignore unexpected fields)
- invoice: prefer invoice_number, invoice_date, due_date, vendor_name, vendor_address,
  bill_to_name, bill_to_address, po_number, payment_terms, currency. Rows in line_items[]:
  item_description, quantity, unit_price, amount. Totals under normalized.totals.
- receipt: store_name, store_address, receipt_number, receipt_date, payment_method,
  card_last_four. Rows in line_items[]. Totals under normalized.totals.
- bol: bol_number, order_number, pro_number, po_number, shipper_name, shipper_address,
  consignee_name, consignee_address, carrier_name, carrier_scac, driver_name,
  trailer_number, truck_number, origin_city, origin_state, destination_city,
  destination_state, tcn_number, epa_number, load_start_timestamp, load_end_timestamp.
  Rows in line_items[]: product_description, un_number, hazard_class, packing_group,
  gross_gallons, net_gallons, temperature, api_gravity, compartment_number.
  If provided on the doc, put totals under normalized.totals.

OUTPUT SHAPE (MINIMAL EXAMPLE)
{
  "fields": [ ... ],
  "tables": [ ... ],
  "normalized": {
    "document_type": "invoice|receipt|bol|delivery_ticket|auto",
    "header": { "...header fields..." },
    "line_items": [ { "...row fields only..." } ],
    "totals": { "...totals..." }
  }
}
"""

_DOC_TYPE_HINTS: Dict[str, str] = {
    "bol": (
        "This is a Bill of Lading (BOL). Populate normalized.header with bol_number, order_number, "
        "pro_number, po_number, shipper_name, shipper_address, consignee_name, consignee_address, "
        "carrier_name, carrier_scac, trailer_number, truck_number, driver_name, origin_city, origin_state, "
        "destination_city, destination_state, tcn_number, epa_number, load_start_timestamp, load_end_timestamp. "
        "All product/compartment rows go to normalized.line_items[]. Each row may include: product_description, "
        "un_number, hazard_class, packing_group, gross_gallons, net_gallons, temperature, api_gravity, "
        "compartment_number, and optional _evidence {table_id, cell_refs, page}. If totals exist, use normalized.totals."
    ),
    "invoice": (
        "This is an Invoice. Put invoice_number, invoice_date, due_date, vendor_name, vendor_address, "
        "bill_to_name, bill_to_address, po_number, payment_terms, currency in normalized.header. "
        "Return line items in normalized.line_items[] with item_description, quantity, unit_price, amount. "
        "If present, use normalized.totals {subtotal, tax_amount, total_amount}."
    ),
    "receipt": (
        "This is a Receipt. Put store_name, store_address, receipt_number, receipt_date, payment_method, "
        "card_last_four in normalized.header. Return purchased rows in normalized.line_items[]. "
        "If present, use normalized.totals {subtotal, tax, total}."
    ),
    "delivery_ticket": (
        "This is a Delivery Ticket. Put ticket_number, order_number, delivery_date, customer_name, "
        "customer_address, driver_name, truck_number, trailer_number in normalized.header. "
        "Return delivered products in normalized.line_items[] with product_description, quantity, unit, "
        "gross_weight, net_weight, temperature, compartment_number. Put totals in normalized.totals if present."
    ),
    "auto": (
        "Unknown document type. Identify likely header fields in normalized.header and any table-like "
        "data in normalized.line_items[]. Keep strict separation between header and row data. "
        "If totals are found, use normalized.totals."
    )
}


class GPTVisionExtractor:
    """
    Cloud-based KIE via Azure OpenAI GPT Vision.

    Usage::

        extractor = GPTVisionExtractor(config.azure)
        fields = extractor.extract(page_image, page_num=1)
        # fields is List[Field] ready for RankAndFuse
    """

    def __init__(
        self,
        azure_config: AzureConfig,
        cost_tracker=None,
        response_cache=None,
    ) -> None:
        if not azure_config.is_openai_ready:
            raise ValueError(
                "Azure OpenAI is not configured. "
                "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, and "
                "AZURE_OPENAI_DEPLOYMENT in your .env file or config YAML."
            )

        self._config = azure_config
        self._client = None  # lazy
        self.cost_tracker = cost_tracker
        self.cache = response_cache
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
            import httpx
            import certifi

            self._client = AzureOpenAI(
                azure_endpoint=self._config.openai_endpoint,
                api_key=self._config.openai_key,
                api_version=self._config.openai_api_version,
                http_client=httpx.Client(verify=certifi.where()),
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
        deployment_override: Optional[str] = None,
    ) -> List[Field]:
        """
        Extract key fields from a document page image using GPT Vision.

        Args:
            image:              BGR or RGB numpy array (H×W×3).
            page_num:           1-based page number.
            ocr_text:           Optional pre-extracted OCR text to include in prompt.
            document_type:      Document type hint (``auto``, ``bol``, ``invoice``, …).
                                Overrides ``AzureConfig.document_type`` if given.
            deployment_override: GPT deployment name to use for this call only.
                                 When set by the smart classifier, this overrides
                                 ``AzureConfig.openai_deployment``.

        Returns:
            ``List[Field]`` ready for the rank-and-fuse pipeline.
        """
        doc_type = document_type or self._config.document_type or "auto"
        deployment = deployment_override or self._config.openai_deployment

        # ── Check cache ─────────────────────────────────────────────
        cache_key = None
        if self.cache is not None:
            image_bytes = self._encode_image_b64(image).encode()
            cache_key = self.cache.make_key(
                image_bytes,
                service="gpt",
                model=deployment,
                extra=doc_type,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(
                    f"Cache hit for GPT Vision page {page_num} — skipping API call"
                )
                fields = self._dict_to_fields(cached, page_num)
                if self.cost_tracker:
                    self.cost_tracker.record_gpt_call(
                        deployment=deployment,
                        latency=0.0, cached=True,
                    )
                return fields

        messages = self._build_messages(image, ocr_text, doc_type)

        t0 = time.perf_counter()
        logger.info(
            f"Sending page {page_num} to GPT Vision KIE "
            f"(deployment={deployment}, type={doc_type})"
        )

        try:
            response = self.client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_completion_tokens=self._config.gpt_max_tokens,
                temperature=self._config.gpt_temperature,
            )
        except Exception as _tok_err:
            if "max_completion_tokens" in str(_tok_err):
                # Older model — fall back to legacy parameter
                response = self.client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_tokens=self._config.gpt_max_tokens,
                    temperature=self._config.gpt_temperature,
                )
            else:
                raise

        elapsed = time.perf_counter() - t0
        raw_text = (response.choices[0].message.content or "").strip()
        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        logger.info(
            f"GPT Vision response in {elapsed:.2f}s  "
            f"(tokens: prompt={prompt_tokens}, "
            f"completion={completion_tokens})"
        )

        # ── Record cost ─────────────────────────────────────────────
        if self.cost_tracker:
            self.cost_tracker.record_gpt_call(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                deployment=deployment,
                latency=elapsed,
            )

        # Parse the JSON and convert to Field objects
        extracted = self._parse_response(raw_text)

        # ── Normalized schema sanity check (non-fatal) ──────────────
        if isinstance(extracted, dict) and "normalized" in extracted and isinstance(extracted["normalized"], dict):
            norm = extracted["normalized"]
            if "header" in norm and not isinstance(norm.get("header"), dict):
                logger.warning("normalized.header is not a dict; got=%s", type(norm.get("header")))
            if "line_items" in norm and not isinstance(norm.get("line_items"), list):
                logger.warning("normalized.line_items is not a list; got=%s", type(norm.get("line_items")))
            if "totals" in norm and not isinstance(norm.get("totals"), dict):
                logger.warning("normalized.totals is not a dict; got=%s", type(norm.get("totals")))

        # ── Promote normalized into namespaced Field[] entries ───────
        if isinstance(extracted, dict) and "normalized" in extracted and isinstance(extracted["normalized"], dict):
            norm = extracted["normalized"]
            # Copy into namespaced keys so _dict_to_fields() emits Field[]
            # entries like: normalized.header.vendor_name,
            # normalized.line_items[0].item_description, etc.
            extracted.setdefault("normalized.header", norm.get("header", {}))
            extracted.setdefault("normalized.line_items", norm.get("line_items", []))
            if "line_items_secondary" in norm:
                extracted.setdefault("normalized.line_items_secondary", norm.get("line_items_secondary", []))
            if "totals" in norm:
                extracted.setdefault("normalized.totals", norm.get("totals", {}))

        # ── Store in cache ──────────────────────────────────────────
        if self.cache is not None and cache_key and extracted:
            self.cache.put(cache_key, extracted, metadata={
                "page_num": page_num, "doc_type": doc_type,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })

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
