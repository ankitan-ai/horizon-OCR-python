"""Tests for GPT Vision KIE provider."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import json
import numpy as np

from docvision.config import AzureConfig, ProcessingMode
from docvision.types import (
    Field,
    FieldStatus,
    SourceEngine,
)
from docvision.azure.gpt_vision_kie import (
    GPTVisionExtractor,
    _looks_like_date,
    _looks_like_currency,
    _SYSTEM_PROMPT,
    _DOC_TYPE_HINTS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def azure_config():
    """AzureConfig with fake credentials (never hits the network)."""
    return AzureConfig(
        processing_mode=ProcessingMode.AZURE,
        openai_endpoint="https://fake.openai.azure.com/",
        openai_key="fake-key-000",
        openai_deployment="gpt-4o-mini",
        openai_api_version="2024-12-01-preview",
        gpt_max_tokens=4096,
        gpt_temperature=0.0,
        document_type="auto",
    )


@pytest.fixture
def extractor(azure_config):
    """GPTVisionExtractor with fake config (no real HTTP calls)."""
    return GPTVisionExtractor(azure_config)


@pytest.fixture
def dummy_image():
    """A small 100×200 BGR numpy image."""
    return np.zeros((100, 200, 3), dtype=np.uint8)


# ── Helper to build a mock OpenAI response ──────────────────────────────────

def _mock_response(content: str, prompt_tokens: int = 500, completion_tokens: int = 200):
    """Build a mock ChatCompletion response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ── Tests: initialisation ───────────────────────────────────────────────────

class TestExtractorInit:
    """Tests for extractor construction and validation."""

    def test_init_success(self, azure_config):
        ext = GPTVisionExtractor(azure_config)
        assert ext._config is azure_config
        assert ext._client is None  # lazy

    def test_init_raises_without_credentials(self, monkeypatch):
        # Clear env vars so load_dotenv values are ignored
        monkeypatch.delenv("AZURE_DOC_INTELLIGENCE_KEY", raising=False)
        monkeypatch.delenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        # Prevent load_dotenv from re-populating env vars from .env file
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
        bad_config = AzureConfig()  # no endpoint / key / deployment
        with pytest.raises(ValueError, match="not configured"):
            GPTVisionExtractor(bad_config)

    def test_lazy_client_not_created_at_init(self, extractor):
        assert extractor._client is None


# ── Tests: image encoding ───────────────────────────────────────────────────

class TestImageEncoding:
    """Tests for _encode_image_b64."""

    def test_encode_returns_string(self, dummy_image):
        result = GPTVisionExtractor._encode_image_b64(dummy_image)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_encode_is_valid_base64(self, dummy_image):
        import base64
        result = GPTVisionExtractor._encode_image_b64(dummy_image)
        # Should decode without error
        decoded = base64.b64decode(result)
        # Should start with PNG magic bytes
        assert decoded[:4] == b"\x89PNG"


# ── Tests: response parsing ─────────────────────────────────────────────────

class TestResponseParsing:
    """Tests for _parse_response."""

    def test_clean_json(self):
        raw = '{"invoice_number": "INV-001", "total_amount": "$1,234.56"}'
        result = GPTVisionExtractor._parse_response(raw)
        assert result == {"invoice_number": "INV-001", "total_amount": "$1,234.56"}

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"name": "test"}\n```'
        result = GPTVisionExtractor._parse_response(raw)
        assert result == {"name": "test"}

    def test_json_with_plain_fences(self):
        raw = '```\n{"key": "value"}\n```'
        result = GPTVisionExtractor._parse_response(raw)
        assert result == {"key": "value"}

    def test_json_embedded_in_text(self):
        raw = 'Here is the result:\n{"field": "val"}\nDone.'
        result = GPTVisionExtractor._parse_response(raw)
        assert result == {"field": "val"}

    def test_empty_response(self):
        result = GPTVisionExtractor._parse_response("")
        assert result == {}

    def test_non_json_response(self):
        result = GPTVisionExtractor._parse_response("I cannot parse this document.")
        assert result == {}

    def test_nested_json(self):
        raw = json.dumps({
            "vendor": {"name": "Acme", "address": "123 Main St"},
            "total": "500.00",
        })
        result = GPTVisionExtractor._parse_response(raw)
        assert result["vendor"]["name"] == "Acme"

    def test_json_with_leading_whitespace(self):
        raw = '  \n  {"a": 1}  \n  '
        result = GPTVisionExtractor._parse_response(raw)
        assert result == {"a": 1}


# ── Tests: Field conversion ─────────────────────────────────────────────────

class TestFieldConversion:
    """Tests for _dict_to_fields and _create_field."""

    def test_flat_dict(self, extractor):
        data = {"invoice_number": "INV-001", "total": "$500.00"}
        fields = extractor._dict_to_fields(data, page_num=1)

        assert len(fields) == 2
        names = {f.name for f in fields}
        assert "invoice_number" in names
        assert "total" in names

    def test_nested_dict(self, extractor):
        data = {"vendor": {"name": "Acme", "city": "Dallas"}}
        fields = extractor._dict_to_fields(data, page_num=1)

        assert len(fields) == 2
        names = {f.name for f in fields}
        assert "vendor.name" in names
        assert "vendor.city" in names

    def test_list_of_dicts(self, extractor):
        data = {
            "line_items": [
                {"item": "Widget", "qty": 5},
                {"item": "Gadget", "qty": 3},
            ]
        }
        fields = extractor._dict_to_fields(data, page_num=2)

        assert len(fields) == 4
        names = {f.name for f in fields}
        assert "line_items[0].item" in names
        assert "line_items[1].qty" in names

    def test_list_of_scalars(self, extractor):
        data = {"tags": ["urgent", "express"]}
        fields = extractor._dict_to_fields(data, page_num=1)

        assert len(fields) == 2
        assert fields[0].name == "tags[0]"
        assert fields[0].value == "urgent"

    def test_field_source_is_gpt_vision(self, extractor):
        data = {"key": "value"}
        fields = extractor._dict_to_fields(data, page_num=1)

        assert fields[0].chosen_source == SourceEngine.GPT_VISION
        assert fields[0].candidates[0].source == SourceEngine.GPT_VISION

    def test_field_confidence_default(self, extractor):
        data = {"key": "value"}
        fields = extractor._dict_to_fields(data, page_num=1)

        assert fields[0].confidence == 0.90
        assert fields[0].status == FieldStatus.CONFIDENT

    def test_field_page_set(self, extractor):
        data = {"key": "value"}
        fields = extractor._dict_to_fields(data, page_num=3)

        assert fields[0].page == 3
        assert fields[0].candidates[0].page == 3

    def test_date_type_detection(self, extractor):
        data = {"invoice_date": "2025-01-15"}
        fields = extractor._dict_to_fields(data, page_num=1)
        assert fields[0].data_type == "date"

    def test_currency_type_detection(self, extractor):
        data = {"total": "$1,234.56"}
        fields = extractor._dict_to_fields(data, page_num=1)
        assert fields[0].data_type == "currency"

    def test_number_type_detection(self, extractor):
        data = {"quantity": 42}
        fields = extractor._dict_to_fields(data, page_num=1)
        assert fields[0].data_type == "number"


# ── Tests: message construction ──────────────────────────────────────────────

class TestMessageConstruction:
    """Tests for _build_messages."""

    def test_basic_messages(self, extractor, dummy_image):
        msgs = extractor._build_messages(dummy_image, None, "auto")

        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert _SYSTEM_PROMPT in msgs[0]["content"]

    def test_user_content_has_image(self, extractor, dummy_image):
        msgs = extractor._build_messages(dummy_image, None, "auto")
        user_parts = msgs[1]["content"]

        image_parts = [p for p in user_parts if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_ocr_text_included(self, extractor, dummy_image):
        msgs = extractor._build_messages(dummy_image, "Sample OCR text", "auto")
        user_parts = msgs[1]["content"]

        text_parts = [p for p in user_parts if p.get("type") == "text"]
        assert len(text_parts) == 1
        assert "Sample OCR text" in text_parts[0]["text"]

    def test_doc_type_hint_bol(self, extractor, dummy_image):
        msgs = extractor._build_messages(dummy_image, None, "bol")
        system_content = msgs[0]["content"]
        assert "Bill of Lading" in system_content

    def test_doc_type_hint_invoice(self, extractor, dummy_image):
        msgs = extractor._build_messages(dummy_image, None, "invoice")
        system_content = msgs[0]["content"]
        assert "Invoice" in system_content

    def test_doc_type_hint_receipt(self, extractor, dummy_image):
        msgs = extractor._build_messages(dummy_image, None, "receipt")
        system_content = msgs[0]["content"]
        assert "Receipt" in system_content

    def test_doc_type_hint_delivery_ticket(self, extractor, dummy_image):
        msgs = extractor._build_messages(dummy_image, None, "delivery_ticket")
        system_content = msgs[0]["content"]
        assert "Delivery Ticket" in system_content

    def test_unknown_doc_type_no_extra_hint(self, extractor, dummy_image):
        msgs = extractor._build_messages(dummy_image, None, "unknown_type")
        system_content = msgs[0]["content"]
        # Should just have the base system prompt, no extra hint
        assert system_content.strip() == _SYSTEM_PROMPT.strip()

    def test_image_detail_high(self, extractor, dummy_image):
        msgs = extractor._build_messages(dummy_image, None, "auto")
        user_parts = msgs[1]["content"]
        image_part = [p for p in user_parts if p.get("type") == "image_url"][0]
        assert image_part["image_url"]["detail"] == "high"


# ── Tests: utility helpers ──────────────────────────────────────────────────

class TestUtilityHelpers:
    """Tests for _looks_like_date and _looks_like_currency."""

    @pytest.mark.parametrize(
        "value",
        ["2025-01-15", "01/15/2025", "15-01-2025", "January 15, 2025"],
    )
    def test_date_positive(self, value):
        assert _looks_like_date(value) is True

    @pytest.mark.parametrize("value", ["hello", "12345", "abc-def-ghi"])
    def test_date_negative(self, value):
        assert _looks_like_date(value) is False

    @pytest.mark.parametrize("value", ["$1,234.56", "€500.00", "1234.56$", "99.99"])
    def test_currency_positive(self, value):
        assert _looks_like_currency(value) is True

    @pytest.mark.parametrize("value", ["hello", "123", "abc"])
    def test_currency_negative(self, value):
        assert _looks_like_currency(value) is False


# ── Tests: doc type hints completeness ──────────────────────────────────────

class TestDocTypeHints:
    """Ensure prompt templates exist for expected document types."""

    @pytest.mark.parametrize("doc_type", ["bol", "invoice", "receipt", "delivery_ticket"])
    def test_hint_exists(self, doc_type):
        assert doc_type in _DOC_TYPE_HINTS
        assert len(_DOC_TYPE_HINTS[doc_type]) > 20


# ── Tests: full extract (mocked API call) ───────────────────────────────────

class TestExtract:
    """End-to-end test of extract() with a mocked Azure OpenAI response."""

    def test_extract_returns_fields(self, extractor, dummy_image):
        gpt_response = json.dumps({
            "invoice_number": "INV-2025-001",
            "invoice_date": "2025-06-15",
            "total_amount": "$3,450.00",
            "vendor_name": "Acme Corp",
        })
        mock_resp = _mock_response(gpt_response)

        with patch.object(
            type(extractor), "client", new_callable=PropertyMock
        ) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_prop.return_value = mock_client

            fields = extractor.extract(dummy_image, page_num=1)

        assert len(fields) == 4
        names = {f.name for f in fields}
        assert "invoice_number" in names
        assert "total_amount" in names

        inv_field = next(f for f in fields if f.name == "invoice_number")
        assert inv_field.value == "INV-2025-001"
        assert inv_field.chosen_source == SourceEngine.GPT_VISION

    def test_extract_with_ocr_text(self, extractor, dummy_image):
        gpt_response = json.dumps({"bol_number": "BOL-123"})
        mock_resp = _mock_response(gpt_response)

        with patch.object(
            type(extractor), "client", new_callable=PropertyMock
        ) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_prop.return_value = mock_client

            fields = extractor.extract(
                dummy_image,
                page_num=1,
                ocr_text="BOL Number: BOL-123",
                document_type="bol",
            )

        assert len(fields) == 1
        assert fields[0].value == "BOL-123"

    def test_extract_handles_empty_response(self, extractor, dummy_image):
        mock_resp = _mock_response("I could not process this document.")

        with patch.object(
            type(extractor), "client", new_callable=PropertyMock
        ) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_prop.return_value = mock_client

            fields = extractor.extract(dummy_image, page_num=1)

        assert fields == []

    def test_extract_with_nested_response(self, extractor, dummy_image):
        gpt_response = json.dumps({
            "shipper": {"name": "Acme", "city": "Dallas"},
            "weight": 5000,
        })
        mock_resp = _mock_response(gpt_response)

        with patch.object(
            type(extractor), "client", new_callable=PropertyMock
        ) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_prop.return_value = mock_client

            fields = extractor.extract(dummy_image, page_num=1)

        assert len(fields) == 3
        names = {f.name for f in fields}
        assert "shipper.name" in names
        assert "weight" in names
