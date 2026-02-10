"""Integration tests for the full pipeline."""

import pytest
import numpy as np


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for DocumentProcessor."""
    
    def test_processor_initialization(self, test_config):
        from docvision.pipeline import DocumentProcessor
        
        processor = DocumentProcessor(test_config)
        
        assert processor is not None
        assert processor.config is not None
    
    def test_process_image(self, test_config, sample_image_path, temp_dir):
        from docvision.pipeline import DocumentProcessor, ProcessingOptions
        
        processor = DocumentProcessor(test_config)
        
        options = ProcessingOptions(
            preprocess=True,
            detect_layout=True,
            detect_text=True,
            detect_tables=True,
            run_ocr=False,  # Skip OCR to avoid model loading
            run_donut=False,
            run_layoutlmv3=False,
            run_validators=False,
            save_artifacts=False,
            save_json=False
        )
        
        result = processor.process(str(sample_image_path), options)
        
        assert result.success is True
        assert result.document is not None
        assert result.document.page_count == 1
    
    def test_process_nonexistent_file(self, test_config):
        from docvision.pipeline import DocumentProcessor
        
        processor = DocumentProcessor(test_config)
        
        result = processor.process("/nonexistent/file.pdf")
        
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()
    
    def test_process_unsupported_file(self, test_config, temp_dir):
        from docvision.pipeline import DocumentProcessor
        
        # Create a file with unsupported extension
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("test content")
        
        processor = DocumentProcessor(test_config)
        result = processor.process(str(unsupported_file))
        
        assert result.success is False
        assert "unsupported" in result.error.lower()


@pytest.mark.integration
class TestPipelineOutput:
    """Tests for pipeline output format."""
    
    def test_output_json_structure(self, test_config, sample_image_path):
        from docvision.pipeline import DocumentProcessor, ProcessingOptions
        
        processor = DocumentProcessor(test_config)
        
        options = ProcessingOptions(
            run_ocr=False,
            run_donut=False,
            run_layoutlmv3=False,
            save_json=False
        )
        
        result = processor.process(str(sample_image_path), options)
        
        if result.success:
            doc = result.document
            
            # Check required fields
            assert doc.id is not None
            assert doc.metadata is not None
            assert doc.page_count >= 1
            assert doc.pages is not None
            
            # Check metadata
            assert doc.metadata.filename is not None
            assert doc.metadata.file_type in ["pdf", "image"]
            
            # Check page structure
            for page in doc.pages:
                assert page.number >= 1
                assert page.metadata is not None
                assert page.metadata.width > 0
                assert page.metadata.height > 0


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineWithModels:
    """Integration tests that load ML models."""
    
    @pytest.mark.skip(reason="Requires model download")
    def test_full_pipeline_with_ocr(self, test_config, sample_pdf_path):
        from docvision.pipeline import DocumentProcessor, ProcessingOptions
        
        processor = DocumentProcessor(test_config)
        
        options = ProcessingOptions(
            preprocess=True,
            detect_layout=True,
            detect_text=True,
            detect_tables=True,
            run_ocr=True,
            run_donut=True,
            run_layoutlmv3=True,
            run_validators=True,
            save_artifacts=False,
            save_json=False
        )
        
        result = processor.process(str(sample_pdf_path), options)
        
        assert result.success is True
        assert result.document is not None
        
        # Check that fields were extracted
        doc = result.document
        assert len(doc.fields) > 0
        
        # Check field quality
        for field in doc.fields:
            assert field.name is not None
            assert field.value is not None
            assert 0 <= field.confidence <= 1


class TestDonutParser:
    """Tests for Donut output parsing including invoice/receipt format."""

    def _make_runner(self):
        """Create a DonutRunner without loading the model."""
        from docvision.kie.donut_runner import DonutRunner
        runner = DonutRunner.__new__(DonutRunner)
        runner.model_name = "test"
        runner.device = "cpu"
        runner.max_length = 768
        runner.processor = None
        runner.model = None
        runner.model_loaded = False
        return runner

    def test_parse_xml_tags(self):
        runner = self._make_runner()
        output = "<s_invoice_no>12345</s_invoice_no><s_total>$100.00</s_total>"
        result = runner._parse_output(output)
        assert result["invoice_no"] == "12345"
        assert result["total"] == "$100.00"

    def test_parse_sep_delimited_items(self):
        """The new invoice model outputs line items separated by <sep/>."""
        runner = self._make_runner()
        output = (
            "<s_items>"
            "<s_item_desc>Widget A</s_item_desc><s_item_qty>2</s_item_qty>"
            "<sep/>"
            "<s_item_desc>Widget B</s_item_desc><s_item_qty>5</s_item_qty>"
            "</s_items>"
        )
        result = runner._parse_output(output)
        assert "items" in result
        assert isinstance(result["items"], list)
        assert len(result["items"]) == 2
        assert result["items"][0]["item_desc"] == "Widget A"
        assert result["items"][0]["item_qty"] == "2"
        assert result["items"][1]["item_desc"] == "Widget B"
        assert result["items"][1]["item_qty"] == "5"

    def test_parse_full_invoice_output(self):
        """Simulate full invoice model output with header + items + summary."""
        runner = self._make_runner()
        output = (
            "<s_header>"
            "<s_invoice_no>INV-001</s_invoice_no>"
            "<s_invoice_date>01/15/2025</s_invoice_date>"
            "<s_seller>Acme Corp</s_seller>"
            "<s_client>Widget Inc</s_client>"
            "</s_header>"
            "<s_items>"
            "<s_item_desc>Part X</s_item_desc><s_item_qty>3</s_item_qty><s_item_net_price>10.00</s_item_net_price>"
            "<sep/>"
            "<s_item_desc>Part Y</s_item_desc><s_item_qty>1</s_item_qty><s_item_net_price>25.00</s_item_net_price>"
            "</s_items>"
            "<s_summary>"
            "<s_total_net_worth>$55.00</s_total_net_worth>"
            "<s_total_vat>$5.50</s_total_vat>"
            "<s_total_gross_worth>$60.50</s_total_gross_worth>"
            "</s_summary>"
        )
        result = runner._parse_output(output)

        # Header
        assert result["header"]["invoice_no"] == "INV-001"
        assert result["header"]["seller"] == "Acme Corp"

        # Items list
        assert isinstance(result["items"], list)
        assert len(result["items"]) == 2
        assert result["items"][1]["item_net_price"] == "25.00"

        # Summary
        assert result["summary"]["total_gross_worth"] == "$60.50"

    def test_parse_json_output(self):
        runner = self._make_runner()
        output = '{"invoice_no": "999", "total": "50.00"}'
        result = runner._parse_output(output)
        assert result["invoice_no"] == "999"
        assert result["total"] == "50.00"

    def test_parse_empty_output(self):
        runner = self._make_runner()
        result = runner._parse_output("")
        assert result == {}

    def test_fields_from_invoice_dict(self):
        """Ensure _dict_to_fields handles the invoice model's nested structure."""
        runner = self._make_runner()
        data = {
            "header": {"invoice_no": "INV-100", "seller": "Test Co"},
            "items": [
                {"item_desc": "A", "item_qty": "1"},
                {"item_desc": "B", "item_qty": "2"},
            ],
            "summary": {"total_gross_worth": "$300.00"},
        }
        fields = runner._dict_to_fields(data, confidence=0.85, page_num=1)
        names = [f.name for f in fields]
        assert "header.invoice_no" in names
        assert "header.seller" in names
        assert "items[0].item_desc" in names
        assert "items[1].item_qty" in names
        assert "summary.total_gross_worth" in names
