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
