"""
Tests for targeted_reocr module.

Tests the targeted re-OCR functionality for improving low-confidence text regions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from docvision.ocr.targeted_reocr import (
    TargetedReOCR,
    ReOCRConfig,
    ReOCRStrategy,
    ReOCRResult,
)
from docvision.types import TextLine, BoundingBox, ContentType, SourceEngine


class TestReOCRConfig:
    """Tests for ReOCRConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ReOCRConfig()
        
        assert config.confidence_threshold == 0.70
        assert config.improvement_threshold == 0.05
        assert config.strategy == ReOCRStrategy.ENSEMBLE
        assert config.max_reocr_lines == 50
        assert config.scale_factor == 2.0
        assert config.azure_retry_enabled is True
        assert config.azure_retry_threshold == 0.50
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ReOCRConfig(
            confidence_threshold=0.60,
            strategy=ReOCRStrategy.TROCR_ONLY,
            max_reocr_lines=20,
            azure_retry_enabled=False,
        )
        
        assert config.confidence_threshold == 0.60
        assert config.strategy == ReOCRStrategy.TROCR_ONLY
        assert config.max_reocr_lines == 20
        assert config.azure_retry_enabled is False


class TestReOCRResult:
    """Tests for ReOCRResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a ReOCRResult."""
        result = ReOCRResult(
            original_text="hello",
            original_confidence=0.5,
            new_text="Hello",
            new_confidence=0.85,
            improved=True,
            source=SourceEngine.TROCR,
        )
        
        assert result.original_text == "hello"
        assert result.new_text == "Hello"
        assert result.improved is True
        assert result.source == SourceEngine.TROCR


class TestTargetedReOCR:
    """Tests for TargetedReOCR class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for tests."""
        return ReOCRConfig(
            confidence_threshold=0.70,
            improvement_threshold=0.05,
            strategy=ReOCRStrategy.ENSEMBLE,
            max_reocr_lines=10,
        )
    
    @pytest.fixture
    def sample_text_lines(self):
        """Create sample text lines with various confidence levels."""
        return [
            TextLine(
                text="High confidence line",
                bbox=BoundingBox(x1=10, y1=10, x2=200, y2=30),
                confidence=0.95,
                source=SourceEngine.TROCR,
            ),
            TextLine(
                text="Medium confidence line",
                bbox=BoundingBox(x1=10, y1=40, x2=200, y2=60),
                confidence=0.65,
                source=SourceEngine.TROCR,
            ),
            TextLine(
                text="Low confidence line",
                bbox=BoundingBox(x1=10, y1=70, x2=200, y2=90),
                confidence=0.45,
                source=SourceEngine.TROCR,
            ),
            TextLine(
                text="Very low confidence",
                bbox=BoundingBox(x1=10, y1=100, x2=200, y2=120),
                confidence=0.30,
                source=SourceEngine.TROCR,
            ),
        ]
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    
    def test_init_with_defaults(self):
        """Test initialization with default config."""
        reocr = TargetedReOCR()
        
        assert reocr.config.confidence_threshold == 0.70
        assert reocr.device == "cpu"
        assert reocr.stats["total_processed"] == 0
    
    def test_init_with_custom_config(self, sample_config):
        """Test initialization with custom config."""
        reocr = TargetedReOCR(config=sample_config, device="cuda")
        
        assert reocr.config.confidence_threshold == 0.70
        assert reocr.device == "cuda"
    
    def test_identify_low_confidence_lines(self, sample_config, sample_text_lines):
        """Test identification of low-confidence lines."""
        reocr = TargetedReOCR(config=sample_config)
        
        low_conf = reocr.identify_low_confidence_lines(sample_text_lines)
        
        # Should find 3 lines below 0.70 threshold
        assert len(low_conf) == 3
        assert all(line.confidence < 0.70 for line in low_conf)
    
    def test_identify_low_confidence_with_custom_threshold(self, sample_config, sample_text_lines):
        """Test identification with custom threshold."""
        reocr = TargetedReOCR(config=sample_config)
        
        low_conf = reocr.identify_low_confidence_lines(sample_text_lines, threshold=0.50)
        
        # Should find 2 lines below 0.50 threshold
        assert len(low_conf) == 2
        assert all(line.confidence < 0.50 for line in low_conf)
    
    def test_identify_respects_max_lines(self, sample_config, sample_text_lines):
        """Test that max_lines limit is respected."""
        config = ReOCRConfig(confidence_threshold=0.70, max_reocr_lines=2)
        reocr = TargetedReOCR(config=config)
        
        low_conf = reocr.identify_low_confidence_lines(sample_text_lines)
        
        # Should be limited to 2 lines (lowest confidence first)
        assert len(low_conf) == 2
        # Should have the lowest confidence lines
        assert low_conf[0].confidence == 0.30
        assert low_conf[1].confidence == 0.45
    
    def test_crop_region_valid(self, sample_config, sample_image):
        """Test cropping a valid region."""
        reocr = TargetedReOCR(config=sample_config)
        bbox = BoundingBox(x1=50, y1=50, x2=150, y2=100)
        
        crop = reocr.crop_region(sample_image, bbox, padding=5)
        
        assert crop is not None
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0
    
    def test_crop_region_with_padding(self, sample_config, sample_image):
        """Test that padding is applied correctly."""
        reocr = TargetedReOCR(config=sample_config)
        bbox = BoundingBox(x1=50, y1=50, x2=100, y2=80)
        
        crop_no_pad = reocr.crop_region(sample_image, bbox, padding=0)
        crop_with_pad = reocr.crop_region(sample_image, bbox, padding=10)
        
        # Crop with padding should be larger
        assert crop_with_pad.shape[0] > crop_no_pad.shape[0]
        assert crop_with_pad.shape[1] > crop_no_pad.shape[1]
    
    def test_crop_region_too_small(self, sample_config, sample_image):
        """Test that small crops return None."""
        config = ReOCRConfig(min_crop_size=(50, 50))  # Large minimum
        reocr = TargetedReOCR(config=config)
        bbox = BoundingBox(x1=10, y1=10, x2=20, y2=20)  # 10x10 crop
        
        crop = reocr.crop_region(sample_image, bbox, padding=0)
        
        assert crop is None
    
    def test_crop_region_boundary_clipping(self, sample_config, sample_image):
        """Test that crops are clipped to image boundaries."""
        reocr = TargetedReOCR(config=sample_config)
        # Bbox that extends beyond image boundaries
        bbox = BoundingBox(x1=-10, y1=-10, x2=310, y2=210)
        
        crop = reocr.crop_region(sample_image, bbox, padding=0)
        
        # Should be clipped to image size
        assert crop is not None
        assert crop.shape[0] <= sample_image.shape[0]
        assert crop.shape[1] <= sample_image.shape[1]
    
    def test_apply_enhanced_preprocessing(self, sample_config, sample_image):
        """Test enhanced preprocessing pipeline runs without errors."""
        reocr = TargetedReOCR(config=sample_config)
        
        input_crop = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        
        # Should not raise - actually run preprocessing
        result = reocr.apply_enhanced_preprocessing(input_crop)
        
        assert result is not None
        assert result.shape[0] > 0
        assert result.shape[1] > 0
    
    def test_reocr_line_no_recognizers(self, sample_config, sample_image, sample_text_lines):
        """Test re-OCR when recognizers are not available."""
        reocr = TargetedReOCR(config=sample_config)
        reocr._trocr = False  # Simulate unavailable
        reocr._tesseract = False
        
        line = sample_text_lines[1]  # Medium confidence
        result = reocr.reocr_line(sample_image, line)
        
        # Should return original values unchanged
        assert result.original_text == line.text
        assert result.new_text == line.text
        assert result.improved is False
    
    def test_reocr_line_with_mock_trocr(self, sample_config, sample_image, sample_text_lines):
        """Test re-OCR with mocked TrOCR."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("Improved text", 0.90)
        
        reocr = TargetedReOCR(
            config=sample_config,
            trocr_recognizer=mock_trocr,
        )
        reocr._tesseract = False  # Disable tesseract
        
        line = sample_text_lines[2]  # Low confidence (0.45)
        result = reocr.reocr_line(sample_image, line)
        
        assert result.improved is True
        assert result.new_text == "Improved text"
        assert result.new_confidence == 0.90
    
    def test_process_local_no_low_confidence(self, sample_config, sample_image):
        """Test process_local when no lines are below threshold."""
        high_conf_lines = [
            TextLine(
                text="All high confidence",
                bbox=BoundingBox(x1=10, y1=10, x2=200, y2=30),
                confidence=0.95,
                source=SourceEngine.TROCR,
            ),
        ]
        
        reocr = TargetedReOCR(config=sample_config)
        result = reocr.process_local(sample_image, high_conf_lines)
        
        assert result == high_conf_lines
        assert reocr.stats["total_processed"] == 0
    
    def test_process_local_improves_lines(self, sample_config, sample_image, sample_text_lines):
        """Test that process_local improves low-confidence lines."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("Better text", 0.85)
        
        reocr = TargetedReOCR(
            config=sample_config,
            trocr_recognizer=mock_trocr,
        )
        reocr._tesseract = False
        
        result = reocr.process_local(sample_image, sample_text_lines)
        
        # Stats should reflect processing
        assert reocr.stats["total_processed"] > 0
    
    def test_stats_tracking(self, sample_config, sample_image, sample_text_lines):
        """Test that statistics are tracked correctly."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("Improved", 0.90)
        
        reocr = TargetedReOCR(
            config=sample_config,
            trocr_recognizer=mock_trocr,
        )
        reocr._tesseract = False
        
        reocr.process_local(sample_image, sample_text_lines)
        
        stats = reocr.get_stats()
        assert "total_processed" in stats
        assert "improved" in stats
        assert "improvement_rate" in stats
    
    def test_reset_stats(self, sample_config):
        """Test resetting statistics."""
        reocr = TargetedReOCR(config=sample_config)
        reocr.stats["total_processed"] = 100
        reocr.stats["improved"] = 50
        
        reocr.reset_stats()
        
        assert reocr.stats["total_processed"] == 0
        assert reocr.stats["improved"] == 0


class TestReOCRStrategies:
    """Tests for different re-OCR strategies."""
    
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    
    @pytest.fixture
    def low_conf_line(self):
        return TextLine(
            text="low conf text",
            bbox=BoundingBox(x1=10, y1=10, x2=200, y2=30),
            confidence=0.40,
            source=SourceEngine.TROCR,
        )
    
    def test_ensemble_strategy_picks_best(self, sample_image, low_conf_line):
        """Test that ensemble strategy picks the best result."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("TrOCR result", 0.75)
        
        mock_tesseract = Mock()
        mock_tesseract.recognize_line.return_value = ("Tesseract result", 0.85)
        
        config = ReOCRConfig(strategy=ReOCRStrategy.ENSEMBLE)
        reocr = TargetedReOCR(
            config=config,
            trocr_recognizer=mock_trocr,
            tesseract_recognizer=mock_tesseract,
        )
        
        result = reocr.reocr_line(sample_image, low_conf_line)
        
        # Should pick Tesseract (higher confidence)
        assert result.new_text == "Tesseract result"
        assert result.new_confidence == 0.85
        assert result.source == SourceEngine.TESSERACT
    
    def test_trocr_only_strategy(self, sample_image, low_conf_line):
        """Test TrOCR-only strategy."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("TrOCR only", 0.80)
        
        mock_tesseract = Mock()
        mock_tesseract.recognize_line.return_value = ("Should not use", 0.95)
        
        config = ReOCRConfig(strategy=ReOCRStrategy.TROCR_ONLY)
        reocr = TargetedReOCR(
            config=config,
            trocr_recognizer=mock_trocr,
            tesseract_recognizer=mock_tesseract,
        )
        
        result = reocr.reocr_line(sample_image, low_conf_line)
        
        # Should only use TrOCR
        assert "trocr" in result.engine_results
        assert "tesseract" not in result.engine_results
    
    def test_tesseract_only_strategy(self, sample_image, low_conf_line):
        """Test Tesseract-only strategy."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("Should not use", 0.95)
        
        mock_tesseract = Mock()
        mock_tesseract.recognize_line.return_value = ("Tesseract only", 0.80)
        
        config = ReOCRConfig(strategy=ReOCRStrategy.TESSERACT_ONLY)
        reocr = TargetedReOCR(
            config=config,
            trocr_recognizer=mock_trocr,
            tesseract_recognizer=mock_tesseract,
        )
        
        result = reocr.reocr_line(sample_image, low_conf_line)
        
        # Should only use Tesseract
        assert "tesseract" in result.engine_results
        assert "trocr" not in result.engine_results
    
    def test_sequential_strategy_trocr_sufficient(self, sample_image, low_conf_line):
        """Test sequential strategy when TrOCR provides sufficient improvement."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("TrOCR good enough", 0.85)
        
        mock_tesseract = Mock()
        mock_tesseract.recognize_line.return_value = ("Tesseract backup", 0.90)
        
        config = ReOCRConfig(
            strategy=ReOCRStrategy.SEQUENTIAL,
            improvement_threshold=0.05,
        )
        reocr = TargetedReOCR(
            config=config,
            trocr_recognizer=mock_trocr,
            tesseract_recognizer=mock_tesseract,
        )
        
        result = reocr.reocr_line(sample_image, low_conf_line)
        
        # TrOCR was sufficient (0.85 - 0.40 > 0.05), so Tesseract not called
        assert result.new_text == "TrOCR good enough"
    
    def test_sequential_strategy_fallback_to_tesseract(self, sample_image, low_conf_line):
        """Test sequential strategy falls back to Tesseract."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("TrOCR not enough", 0.42)  # Only 0.02 improvement
        
        mock_tesseract = Mock()
        mock_tesseract.recognize_line.return_value = ("Tesseract better", 0.80)
        
        config = ReOCRConfig(
            strategy=ReOCRStrategy.SEQUENTIAL,
            improvement_threshold=0.05,
        )
        reocr = TargetedReOCR(
            config=config,
            trocr_recognizer=mock_trocr,
            tesseract_recognizer=mock_tesseract,
        )
        
        result = reocr.reocr_line(sample_image, low_conf_line)
        
        # TrOCR insufficient, should fallback to Tesseract
        assert result.new_text == "Tesseract better"
        assert result.new_confidence == 0.80


class TestProcessModes:
    """Tests for different processing modes."""
    
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_lines(self):
        return [
            TextLine(
                text="Low conf",
                bbox=BoundingBox(x1=10, y1=10, x2=100, y2=30),
                confidence=0.50,
                source=SourceEngine.AZURE_DOC_INTELLIGENCE,
            ),
        ]
    
    def test_process_local_mode(self, sample_image, sample_lines):
        """Test process() with local mode."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("Improved", 0.85)
        
        config = ReOCRConfig(confidence_threshold=0.70)
        reocr = TargetedReOCR(config=config, trocr_recognizer=mock_trocr)
        reocr._tesseract = False
        
        result = reocr.process(sample_image, sample_lines, mode="local")
        
        assert reocr.stats["total_processed"] > 0
    
    def test_process_azure_mode_disabled(self, sample_image, sample_lines):
        """Test process() with Azure mode when disabled."""
        config = ReOCRConfig(azure_retry_enabled=False)
        reocr = TargetedReOCR(config=config)
        
        # Should return unchanged lines
        result = reocr.process(sample_image, sample_lines, mode="azure")
        
        assert result == sample_lines
    
    def test_process_hybrid_mode(self, sample_image, sample_lines):
        """Test process() with hybrid mode."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("Local improved", 0.85)
        
        config = ReOCRConfig(
            confidence_threshold=0.70,
            azure_retry_enabled=False,  # Disable Azure retry for this test
        )
        reocr = TargetedReOCR(config=config, trocr_recognizer=mock_trocr)
        reocr._tesseract = False
        
        result = reocr.process(sample_image, sample_lines, mode="hybrid")
        
        # Should process with local first
        assert reocr.stats["total_processed"] > 0
    
    def test_process_unknown_mode_defaults_to_local(self, sample_image, sample_lines):
        """Test that unknown mode defaults to local."""
        mock_trocr = Mock()
        mock_trocr.recognize.return_value = ("Local", 0.85)
        
        config = ReOCRConfig(confidence_threshold=0.70)
        reocr = TargetedReOCR(config=config, trocr_recognizer=mock_trocr)
        reocr._tesseract = False
        
        result = reocr.process(sample_image, sample_lines, mode="invalid_mode")
        
        # Should default to local processing
        assert reocr.stats["total_processed"] > 0


class TestConfigIntegration:
    """Tests for config.py ReOCRConfig integration."""
    
    def test_config_has_reocr_section(self):
        """Test that main Config has reocr section."""
        from docvision.config import Config, ReOCRConfig
        
        config = Config()
        assert hasattr(config, 'reocr')
        assert isinstance(config.reocr, ReOCRConfig)
    
    def test_config_reocr_defaults(self):
        """Test default ReOCRConfig values in main Config."""
        from docvision.config import Config
        
        config = Config()
        
        assert config.reocr.enable is True
        assert config.reocr.confidence_threshold == 0.70
        assert config.reocr.strategy == "ensemble"
        assert config.reocr.azure_retry_enabled is False
    
    def test_config_from_dict_with_reocr(self):
        """Test loading ReOCRConfig from dict."""
        from docvision.config import Config
        
        data = {
            "reocr": {
                "enable": False,
                "confidence_threshold": 0.60,
                "strategy": "trocr_only",
                "max_lines_per_page": 30,
            }
        }
        
        config = Config.from_dict(data)
        
        assert config.reocr.enable is False
        assert config.reocr.confidence_threshold == 0.60
        assert config.reocr.strategy == "trocr_only"
        assert config.reocr.max_lines_per_page == 30
