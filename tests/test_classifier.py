"""Tests for the smart document classifier and routing logic."""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from docvision.config import AzureConfig, ProcessingMode, SmartRoutingConfig, Config
from docvision.azure.classifier import (
    DocumentClassifier,
    ClassificationResult,
    _GPT_ROUTING,
    _DI_ROUTING,
    _CLASSIFIER_SYSTEM,
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
    )


@pytest.fixture
def classifier(azure_config):
    """DocumentClassifier with fake config (no real HTTP calls)."""
    return DocumentClassifier(azure_config)


@pytest.fixture
def dummy_image():
    """A small 100×200 BGR numpy image."""
    return np.zeros((100, 200, 3), dtype=np.uint8)


def _mock_response(content: str, prompt_tokens: int = 50, completion_tokens: int = 20):
    """Build a mock ChatCompletion response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ── ClassificationResult tests ──────────────────────────────────────────────

class TestClassificationResult:
    """Tests for the ClassificationResult dataclass."""

    def test_defaults(self):
        result = ClassificationResult()
        assert result.document_type == "auto"
        assert result.complexity == "medium"
        assert result.recommended_gpt_deployment is None
        assert result.recommended_di_model is None
        assert result.confidence == 0.0

    def test_custom_values(self):
        result = ClassificationResult(
            document_type="invoice",
            complexity="complex",
            confidence=0.95,
        )
        assert result.document_type == "invoice"
        assert result.complexity == "complex"
        assert result.confidence == 0.95


# ── Routing table tests ─────────────────────────────────────────────────────

class TestRoutingTables:
    """Tests for the GPT and DI routing tables."""

    def test_gpt_routing_simple_invoice(self):
        assert _GPT_ROUTING["simple"]["invoice"] == "gpt-4o-mini"

    def test_gpt_routing_complex_bol(self):
        assert _GPT_ROUTING["complex"]["bol"] == "gpt-5-mini"

    def test_gpt_routing_medium_bol(self):
        assert _GPT_ROUTING["medium"]["bol"] == "gpt-4.1-mini"

    def test_gpt_routing_complex_invoice(self):
        assert _GPT_ROUTING["complex"]["invoice"] == "gpt-4.1-mini"

    def test_gpt_routing_all_complexities_have_all_types(self):
        for complexity in ("simple", "medium", "complex"):
            for doc_type in ("invoice", "receipt", "bol", "delivery_ticket", "other"):
                assert doc_type in _GPT_ROUTING[complexity], (
                    f"Missing {doc_type} in {complexity}"
                )

    def test_di_routing_invoice(self):
        assert _DI_ROUTING["invoice"] == "prebuilt-invoice"

    def test_di_routing_bol(self):
        assert _DI_ROUTING["bol"] == "prebuilt-layout"

    def test_di_routing_receipt(self):
        assert _DI_ROUTING["receipt"] == "prebuilt-layout"

    def test_di_routing_all_types_present(self):
        for doc_type in ("invoice", "receipt", "bol", "delivery_ticket", "other"):
            assert doc_type in _DI_ROUTING


# ── Response parsing tests ───────────────────────────────────────────────────

class TestParseResponse:
    """Tests for DocumentClassifier._parse_response."""

    def test_valid_json(self):
        result = DocumentClassifier._parse_response(
            '{"type": "invoice", "complexity": "complex"}'
        )
        assert result.document_type == "invoice"
        assert result.complexity == "complex"

    def test_valid_json_with_markdown_fence(self):
        result = DocumentClassifier._parse_response(
            '```json\n{"type": "bol", "complexity": "simple"}\n```'
        )
        assert result.document_type == "bol"
        assert result.complexity == "simple"

    def test_json_embedded_in_text(self):
        result = DocumentClassifier._parse_response(
            'The document is {"type": "receipt", "complexity": "medium"} as shown.'
        )
        assert result.document_type == "receipt"
        assert result.complexity == "medium"

    def test_invalid_type_normalised_to_other(self):
        result = DocumentClassifier._parse_response(
            '{"type": "contract", "complexity": "simple"}'
        )
        assert result.document_type == "other"
        assert result.complexity == "simple"

    def test_invalid_complexity_normalised_to_medium(self):
        result = DocumentClassifier._parse_response(
            '{"type": "invoice", "complexity": "extreme"}'
        )
        assert result.document_type == "invoice"
        assert result.complexity == "medium"

    def test_garbage_returns_default(self):
        result = DocumentClassifier._parse_response("not valid at all")
        assert result.document_type == "auto"
        assert result.complexity == "medium"

    def test_empty_string_returns_default(self):
        result = DocumentClassifier._parse_response("")
        assert result.document_type == "auto"
        assert result.complexity == "medium"

    def test_delivery_ticket_type(self):
        result = DocumentClassifier._parse_response(
            '{"type": "delivery_ticket", "complexity": "complex"}'
        )
        assert result.document_type == "delivery_ticket"


# ── Route GPT lookup tests ──────────────────────────────────────────────────

class TestRouteGPT:
    """Tests for DocumentClassifier._route_gpt."""

    def test_simple_invoice(self, classifier):
        assert classifier._route_gpt("invoice", "simple") == "gpt-4o-mini"

    def test_complex_bol(self, classifier):
        assert classifier._route_gpt("bol", "complex") == "gpt-5-mini"

    def test_unknown_complexity_falls_back_to_medium(self, classifier):
        result = classifier._route_gpt("invoice", "ultra")
        assert result == "gpt-4o-mini"  # medium → invoice → gpt-4o-mini

    def test_unknown_type_falls_back_to_default(self, classifier):
        result = classifier._route_gpt("contract", "simple")
        # Falls back to self._default_deployment which is gpt-4o-mini
        assert result == "gpt-4o-mini"


# ── Full classify flow (mocked API call) ─────────────────────────────────────

class TestClassifyFlow:
    """Tests for DocumentClassifier.classify with mocked OpenAI calls."""

    def test_classify_invoice_simple(self, classifier, dummy_image):
        mock_resp = _mock_response('{"type": "invoice", "complexity": "simple"}')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        classifier._client = mock_client

        result = classifier.classify(dummy_image)

        assert result.document_type == "invoice"
        assert result.complexity == "simple"
        assert result.recommended_gpt_deployment == "gpt-4o-mini"
        assert result.recommended_di_model == "prebuilt-invoice"

    def test_classify_complex_bol(self, classifier, dummy_image):
        mock_resp = _mock_response('{"type": "bol", "complexity": "complex"}')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        classifier._client = mock_client

        result = classifier.classify(dummy_image)

        assert result.document_type == "bol"
        assert result.complexity == "complex"
        assert result.recommended_gpt_deployment == "gpt-5-mini"
        assert result.recommended_di_model == "prebuilt-layout"

    def test_classify_api_failure_returns_defaults(self, classifier, dummy_image):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        classifier._client = mock_client

        result = classifier.classify(dummy_image)

        assert result.document_type == "auto"
        assert result.complexity == "medium"
        assert result.recommended_gpt_deployment is None

    def test_classify_unconfigured_openai_returns_defaults(self, dummy_image, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
        config = AzureConfig()  # No openai keys
        classifier = DocumentClassifier(config)

        result = classifier.classify(dummy_image)

        assert result.document_type == "auto"
        assert result.complexity == "medium"

    def test_classify_uses_nano_deployment(self, classifier, dummy_image):
        mock_resp = _mock_response('{"type": "receipt", "complexity": "simple"}')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        classifier._client = mock_client

        classifier.classify(dummy_image)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-5-nano"
        # Accepts either new or legacy token-limit parameter
        token_limit = call_kwargs.kwargs.get("max_completion_tokens") or call_kwargs.kwargs.get("max_tokens")
        assert token_limit == 1024

    def test_classify_records_cost(self, azure_config, dummy_image):
        cost_tracker = MagicMock()
        classifier = DocumentClassifier(azure_config, cost_tracker=cost_tracker)

        mock_resp = _mock_response('{"type": "invoice", "complexity": "medium"}')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        classifier._client = mock_client

        classifier.classify(dummy_image)

        cost_tracker.record_gpt_call.assert_called_once()
        call_kwargs = cost_tracker.record_gpt_call.call_args.kwargs
        assert call_kwargs["deployment"] == "gpt-5-nano"
        assert call_kwargs["prompt_tokens"] == 50
        assert call_kwargs["completion_tokens"] == 20


# ── SmartRoutingConfig tests ─────────────────────────────────────────────────

class TestSmartRoutingConfig:
    """Tests for the SmartRoutingConfig dataclass."""

    def test_defaults(self):
        cfg = SmartRoutingConfig()
        assert cfg.enable is True
        assert cfg.classifier_deployment == "gpt-5-nano"
        assert cfg.default_gpt_deployment == "gpt-4o-mini"
        assert cfg.classify_on_auto_only is True

    def test_in_main_config(self):
        config = Config()
        assert hasattr(config, "smart_routing")
        assert config.smart_routing.enable is True

    def test_from_dict(self):
        config = Config.from_dict({
            "smart_routing": {
                "enable": False,
                "classifier_deployment": "gpt-4o-mini",
                "default_gpt_deployment": "gpt-5-mini",
            }
        })
        assert config.smart_routing.enable is False
        assert config.smart_routing.classifier_deployment == "gpt-4o-mini"
        assert config.smart_routing.default_gpt_deployment == "gpt-5-mini"


# ── Config value changes tests ───────────────────────────────────────────────

class TestConfigValueChanges:
    """Tests that config defaults were updated correctly."""

    def test_dpi_is_500(self):
        from docvision.config import PDFConfig
        assert PDFConfig().dpi == 600

    def test_reroute_threshold_lowered(self):
        from docvision.config import ThresholdsConfig
        assert ThresholdsConfig().reroute_to_tesseract_below == 0.60

    def test_gpt_max_tokens_increased(self):
        from docvision.config import AzureConfig
        assert AzureConfig().gpt_max_tokens == 16384

    def test_preprocess_all_enabled(self):
        from docvision.config import PreprocessConfig
        cfg = PreprocessConfig()
        assert cfg.denoise is True
        assert cfg.clahe is True
        assert cfg.sharpen is True
        assert cfg.deskew is True
        assert cfg.dewarp is True


# ── System prompt validation ─────────────────────────────────────────────────

class TestClassifierPrompt:
    """Tests for the classifier system prompt."""

    def test_prompt_mentions_json(self):
        assert "JSON" in _CLASSIFIER_SYSTEM

    def test_prompt_mentions_all_doc_types(self):
        for t in ("invoice", "bol", "receipt", "delivery_ticket", "other"):
            assert t in _CLASSIFIER_SYSTEM

    def test_prompt_mentions_all_complexities(self):
        for c in ("simple", "medium", "complex"):
            assert c in _CLASSIFIER_SYSTEM

    def test_temperature_unsupported_fallback(self, classifier, dummy_image):
        """When the model rejects temperature=0.0, retry without it."""
        mock_resp = _mock_response('{"type": "bol", "complexity": "medium"}')
        mock_client = MagicMock()
        # First call raises temperature error, second call succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception(
                "Unsupported value: 'temperature' does not support 0.0 "
                "with this model. Only the default (1) value is supported."
            ),
            mock_resp,
        ]
        classifier._client = mock_client

        result = classifier.classify(dummy_image)

        assert result.document_type == "bol"
        assert mock_client.chat.completions.create.call_count == 2
        # Second call should not have temperature kwarg
        retry_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
        assert "temperature" not in retry_kwargs

    def test_empty_content_retries_text_only(self, classifier, dummy_image):
        """When image-based call returns empty content, retry text-only."""
        empty_resp = _mock_response("", prompt_tokens=465, completion_tokens=100)
        # Make the empty response's message.content return empty string
        empty_resp.choices[0].message.content = ""
        text_resp = _mock_response(
            '{"type": "other", "complexity": "medium"}',
            prompt_tokens=60,
            completion_tokens=15,
        )
        mock_client = MagicMock()
        # First call (image) → empty, second call (text-only) → valid JSON
        mock_client.chat.completions.create.side_effect = [empty_resp, text_resp]
        classifier._client = mock_client

        result = classifier.classify(dummy_image)

        assert result.document_type == "other"
        assert result.complexity == "medium"
        # Two API calls: image-based + text-only retry
        assert mock_client.chat.completions.create.call_count == 2
        # Second call should be text-only (no image_url in content)
        retry_msgs = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
        user_content = retry_msgs[1]["content"]
        assert isinstance(user_content, str)  # text-only, not a list with image

    def test_empty_content_both_attempts_returns_default(self, classifier, dummy_image):
        """When both image and text-only return empty, fall back to defaults."""
        empty_resp1 = _mock_response("")
        empty_resp1.choices[0].message.content = ""
        empty_resp2 = _mock_response("")
        empty_resp2.choices[0].message.content = ""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [empty_resp1, empty_resp2]
        classifier._client = mock_client

        result = classifier.classify(dummy_image)

        assert result.document_type == "auto"
        assert result.complexity == "medium"
