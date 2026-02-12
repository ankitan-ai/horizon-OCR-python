"""
Configuration management for DocVision.

Loads and validates configuration from YAML files with sensible defaults.
Supports runtime device detection (CPU/GPU).
"""

import os
from pathlib import Path
from typing import Optional, List, Literal
from dataclasses import dataclass, field
import yaml
import torch
from loguru import logger


class ProcessingMode:
    """Processing mode selector."""
    LOCAL = "local"       # All local models (YOLO, CRAFT, TrOCR, etc.)
    AZURE = "azure"       # Azure Document Intelligence + GPT-4o (cloud)
    HYBRID = "hybrid"     # Azure for OCR/layout, local for post-processing


@dataclass
class AzureConfig:
    """Azure AI Foundry configuration for cloud-based processing."""
    # Processing mode: "local", "azure", or "hybrid"
    processing_mode: str = ProcessingMode.LOCAL

    # Azure Document Intelligence (replaces YOLO + CRAFT + TrOCR + TATR + Tesseract)
    doc_intelligence_endpoint: str = ""  # e.g. https://xxx.cognitiveservices.azure.com/
    doc_intelligence_key: str = ""        # API key (prefer env var AZURE_DOC_INTELLIGENCE_KEY)
    doc_intelligence_model: str = "prebuilt-layout"  # prebuilt-layout, prebuilt-read, prebuilt-invoice

    # Azure OpenAI / GPT Vision (replaces Donut + LayoutLMv3 KIE)
    openai_endpoint: str = ""             # e.g. https://xxx.openai.azure.com/
    openai_key: str = ""                  # API key (prefer env var AZURE_OPENAI_KEY)
    openai_deployment: str = "gpt-4o-mini"  # gpt-5.2, gpt-5-nano, gpt-5-mini, gpt-4o-mini, gpt-4.1-mini
    openai_api_version: str = "2024-12-01-preview"  # Azure OpenAI API version
    use_gpt_vision_kie: bool = True       # Whether to use GPT-4o for field extraction

    # GPT-4o structured extraction settings
    gpt_max_tokens: int = 16384
    gpt_temperature: float = 0.0          # Deterministic for extraction
    document_type: str = "auto"           # auto, bol, invoice, receipt, delivery_ticket

    def __post_init__(self):
        """Load keys from environment variables if not set in config."""
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv()  # Load .env file if present
        except ImportError:
            pass  # python-dotenv not installed — rely on real env vars
        if not self.doc_intelligence_key:
            self.doc_intelligence_key = os.environ.get("AZURE_DOC_INTELLIGENCE_KEY", "")
        if not self.openai_key:
            self.openai_key = os.environ.get("AZURE_OPENAI_KEY", "")
        if not self.doc_intelligence_endpoint:
            self.doc_intelligence_endpoint = os.environ.get("AZURE_DOC_INTELLIGENCE_ENDPOINT", "")
        if not self.openai_endpoint:
            self.openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")

    @property
    def is_azure_ready(self) -> bool:
        """Check if Azure Document Intelligence is configured."""
        return bool(self.doc_intelligence_endpoint and self.doc_intelligence_key)

    @property
    def is_openai_ready(self) -> bool:
        """Check if Azure OpenAI is configured."""
        return bool(self.openai_endpoint and self.openai_key and self.openai_deployment)


@dataclass
class RuntimeConfig:
    """Runtime configuration settings."""
    device: str = "auto"  # auto, cuda, cpu
    workers: int = 0  # 0 means os.cpu_count()
    
    def get_device(self) -> str:
        """Resolve actual device to use."""
        if self.device == "auto":
            if torch.cuda.is_available():
                logger.info("CUDA available, using GPU")
                return "cuda"
            else:
                logger.info("CUDA not available, using CPU")
                return "cpu"
        return self.device
    
    def get_workers(self) -> int:
        """Resolve actual number of workers."""
        if self.workers <= 0:
            return os.cpu_count() or 4
        return self.workers


@dataclass
class PDFConfig:
    """PDF processing configuration."""
    dpi: int = 600  # Rasterization DPI (high for maximum accuracy)
    max_pages: Optional[int] = None  # None means process all pages


@dataclass
class PreprocessConfig:
    """Image preprocessing configuration."""
    denoise: bool = True  # NLM denoising
    clahe: bool = True  # Contrast Limited Adaptive Histogram Equalization
    sharpen: bool = True  # Unsharp masking
    deskew: bool = True  # Automatic deskew via Hough transform
    dewarp: bool = True  # Perspective correction


@dataclass
class ModelsConfig:
    """Model paths and identifiers."""
    # Layout detection (DocLayNet-trained)
    layout: str = "models/yolov8x-doclaynet.pt"
    layout_fallback: str = "yolov8n.pt"  # Fallback if custom not available
    
    # Text detection
    craft: str = "models/craft_mlt_25k.pth"
    
    # Table structure
    tatr: str = "models/table-transformer-structure"
    
    # OCR recognition
    trocr_printed: str = "models/trocr-base-printed"
    trocr_handwritten: str = "models/trocr-base-handwritten"
    
    # KIE models
    donut: str = "models/invoice-and-receipts-donut-v1"
    layoutlmv3: str = "models/layoutlmv3-base"


@dataclass
class KIEConfig:
    """Key Information Extraction configuration."""
    use_donut: bool = True
    use_layoutlmv3: bool = True
    use_ppstructure: bool = False  # Reserved for future PaddleOCR integration
    
    # Ensemble weights for rank-and-fuse
    donut_weight: float = 1.0
    layoutlmv3_weight: float = 0.9
    ocr_weight: float = 0.8


@dataclass
class ThresholdsConfig:
    """Confidence thresholds for decision making."""
    # OCR confidence thresholds
    trocr_min_conf: float = 0.75
    tesseract_min_conf: float = 0.70
    reroute_to_tesseract_below: float = 0.60
    
    # Handwriting detection threshold
    handwriting_detection_conf: float = 0.6
    
    # Field extraction
    low_confidence_threshold: float = 0.5  # Below this, mark as uncertain
    min_confidence_for_output: float = 0.2  # Below this, still include but flag


@dataclass
class ValidatorsConfig:
    """Validator configuration for post-processing checks."""
    enable: bool = True
    currency_codes: List[str] = field(default_factory=lambda: ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "INR"])
    amount_tolerance: float = 0.01  # Tolerance for total vs sum(lines) validation
    date_formats: List[str] = field(default_factory=lambda: [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"
    ])


@dataclass
class ArtifactsConfig:
    """Artifact generation configuration for debugging."""
    enable: bool = True
    dir: str = "artifacts"
    save_layout_overlay: bool = True
    save_text_polygons: bool = True
    save_table_structure: bool = True
    save_ocr_overlay: bool = True
    save_preprocessed: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""
    dir: str = "output"
    include_all_candidates: bool = True  # Option C: include all candidates
    include_page_images: bool = False  # Whether to embed base64 page images
    pretty_json: bool = True  # Pretty-print JSON output


@dataclass
class MarkdownConfig:
    """Markdown report generation configuration."""
    enable: bool = True
    dir: str = "markdown"


@dataclass
class SmartRoutingConfig:
    """Smart document classification and model routing."""
    enable: bool = True                     # Enable smart classifier before extraction
    classifier_deployment: str = "gpt-5-nano"  # Cheap model used only for classification
    default_gpt_deployment: str = "gpt-4o-mini"  # Default extraction model
    classify_on_auto_only: bool = True      # Only classify when document_type == "auto"


@dataclass
class ReOCRConfig:
    """Targeted re-OCR configuration for low-confidence regions."""
    enable: bool = True                     # Enable targeted re-OCR
    confidence_threshold: float = 0.70      # Re-process lines below this confidence
    improvement_threshold: float = 0.05     # Minimum improvement to accept new result
    strategy: str = "ensemble"              # ensemble, trocr_only, tesseract, sequential
    max_lines_per_page: int = 50            # Max lines to re-process per page
    
    # Enhanced preprocessing settings
    scale_factor: float = 2.0               # Upscale crop before OCR
    enhanced_denoise: int = 15              # Stronger denoising (default: 10)
    enhanced_clahe: float = 3.0             # Stronger CLAHE (default: 2.0)
    enhanced_sharpen: float = 2.0           # Stronger sharpening (default: 1.5)
    apply_binarization: bool = True         # Apply adaptive binarization
    apply_morphology: bool = True           # Apply morphological cleanup
    
    # Azure re-OCR (optional, adds API cost)
    azure_retry_enabled: bool = False       # Re-send very low confidence regions to Azure
    azure_retry_threshold: float = 0.50     # Threshold for Azure retry (lower = more selective)


@dataclass
class Config:
    """Main configuration container."""
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    kie: KIEConfig = field(default_factory=KIEConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    validators: ValidatorsConfig = field(default_factory=ValidatorsConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    markdown: MarkdownConfig = field(default_factory=MarkdownConfig)
    azure: AzureConfig = field(default_factory=AzureConfig)
    smart_routing: SmartRoutingConfig = field(default_factory=SmartRoutingConfig)
    reocr: ReOCRConfig = field(default_factory=ReOCRConfig)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create Config from dictionary."""
        config = cls()
        
        if "runtime" in data:
            config.runtime = RuntimeConfig(**data["runtime"])
        if "pdf" in data:
            config.pdf = PDFConfig(**data["pdf"])
        if "preprocess" in data:
            config.preprocess = PreprocessConfig(**data["preprocess"])
        if "models" in data:
            config.models = ModelsConfig(**data["models"])
        if "kie" in data:
            config.kie = KIEConfig(**data["kie"])
        if "thresholds" in data:
            config.thresholds = ThresholdsConfig(**data["thresholds"])
        if "validators" in data:
            config.validators = ValidatorsConfig(**data["validators"])
        if "artifacts" in data:
            config.artifacts = ArtifactsConfig(**data["artifacts"])
        if "output" in data:
            config.output = OutputConfig(**data["output"])
        if "markdown" in data:
            config.markdown = MarkdownConfig(**data["markdown"])
        if "azure" in data:
            config.azure = AzureConfig(**data["azure"])
        if "smart_routing" in data:
            config.smart_routing = SmartRoutingConfig(**data["smart_routing"])
        if "reocr" in data:
            config.reocr = ReOCRConfig(**data["reocr"])
        
        return config


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        
    Returns:
        Config object with loaded or default settings.
    """
    if config_path is None:
        logger.info("No config file specified, using defaults")
        return Config()
    
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return Config()
    
    logger.info(f"Loading config from: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        return Config()
    
    return Config.from_dict(data)


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save.
        config_path: Path to save YAML config file.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclasses to dict
    from dataclasses import asdict
    data = asdict(config)
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Config saved to: {config_path}")
