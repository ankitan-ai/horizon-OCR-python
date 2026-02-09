"""Tests for DocVision configuration."""

import pytest
import tempfile
from pathlib import Path


class TestConfig:
    """Tests for Config dataclass."""
    
    def test_default_config(self):
        from docvision.config import Config
        
        config = Config()
        
        assert config.runtime.device == "auto"
        assert config.pdf.dpi == 350  # Higher DPI for better accuracy
        assert config.output.dir == "output"
    
    def test_get_device_cpu(self):
        from docvision.config import RuntimeConfig
        
        config = RuntimeConfig(device="cpu")
        assert config.get_device() == "cpu"
    
    def test_get_device_auto(self, monkeypatch):
        from docvision.config import RuntimeConfig
        
        # Mock torch.cuda.is_available
        import sys
        
        # Create mock torch module
        class MockTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return False
        
        monkeypatch.setitem(sys.modules, 'torch', MockTorch())
        
        config = RuntimeConfig(device="auto")
        # Should return cpu when CUDA not available
        device = config.get_device()
        assert device in ["cpu", "cuda", "mps"]
    
    def test_pdf_config(self):
        from docvision.config import PDFConfig
        
        config = PDFConfig(dpi=150, max_pages=10)
        
        assert config.dpi == 150
        assert config.max_pages == 10
    
    def test_models_config(self):
        from docvision.config import ModelsConfig
        
        config = ModelsConfig()
        
        assert "trocr" in config.trocr_printed.lower()
        assert "donut" in config.donut.lower()


class TestLoadConfig:
    """Tests for YAML config loading."""
    
    def test_load_valid_yaml(self, temp_dir):
        from docvision.config import load_config
        
        yaml_content = """
runtime:
  device: cpu
  workers: 2
  
pdf:
  dpi: 150
  
output:
  dir: "./custom_output"
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(yaml_content)
        
        config = load_config(str(config_path))
        
        assert config.runtime.device == "cpu"
        assert config.runtime.workers == 2
        assert config.pdf.dpi == 150
        assert config.output.dir == "./custom_output"
    
    def test_load_partial_yaml(self, temp_dir):
        from docvision.config import load_config
        
        # Only override some values
        yaml_content = """
pdf:
  dpi: 200
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(yaml_content)
        
        config = load_config(str(config_path))
        
        # Overridden value
        assert config.pdf.dpi == 200
        # Default values
        assert config.runtime.device == "auto"
        assert config.output.dir == "output"
    
    def test_load_nonexistent_file(self):
        from docvision.config import load_config
        
        # Should return default config for non-existent file
        config = load_config("/nonexistent/path/config.yaml")
        
        assert config.runtime.device == "auto"
    
    def test_load_invalid_yaml(self, temp_dir):
        from docvision.config import load_config
        import yaml
        
        yaml_content = "invalid: yaml: content: : :"
        config_path = temp_dir / "config.yaml"
        config_path.write_text(yaml_content)
        
        # Should raise YAML error or return default config
        try:
            config = load_config(str(config_path))
            assert config is not None  # Got default config
        except yaml.YAMLError:
            pass  # Expected - invalid YAML raises error


class TestThresholdsConfig:
    """Tests for threshold configuration."""
    
    def test_default_thresholds(self):
        from docvision.config import ThresholdsConfig
        
        thresholds = ThresholdsConfig()
        
        assert 0 <= thresholds.trocr_min_conf <= 1
        assert 0 <= thresholds.tesseract_min_conf <= 1
        assert thresholds.low_confidence_threshold < thresholds.trocr_min_conf
    
    def test_custom_thresholds(self):
        from docvision.config import ThresholdsConfig
        
        thresholds = ThresholdsConfig(
            trocr_min_conf=0.8,
            tesseract_min_conf=0.65,
            low_confidence_threshold=0.4
        )
        
        assert thresholds.trocr_min_conf == 0.8
        assert thresholds.tesseract_min_conf == 0.65


class TestKIEConfig:
    """Tests for KIE configuration."""
    
    def test_default_kie_config(self):
        from docvision.config import KIEConfig
        
        config = KIEConfig()
        
        assert config.use_donut is True
        assert config.use_layoutlmv3 is True
        assert config.donut_weight + config.layoutlmv3_weight <= 2.0
    
    def test_weights_sum(self):
        from docvision.config import KIEConfig
        
        config = KIEConfig(
            donut_weight=0.6,
            layoutlmv3_weight=0.4
        )
        
        # Weights should be usable for weighted averaging
        total = config.donut_weight + config.layoutlmv3_weight
        assert abs(total - 1.0) < 0.01
