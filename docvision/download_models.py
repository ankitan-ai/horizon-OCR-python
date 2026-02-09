"""
Model download utility for DocVision.

Downloads required model weights from HuggingFace and other sources.
Run with: python -m docvision.download_models
"""

import os
import sys
from pathlib import Path
from loguru import logger


# Model registry: (repo_id, filename, local_path, description)
MODELS = [
    {
        "name": "DocLayNet YOLOv8",
        "repo_id": "hantian/yolo-doclaynet",
        "filename": "yolov8x-doclaynet.pt",
        "local_path": "models/yolov8x-doclaynet.pt",
        "description": "Document layout detection (header, text, table, figure, etc.)",
        "size_mb": 131,
    },
    {
        "name": "CRAFT Text Detector",
        "repo_id": "boomb0om/CRAFT-text-detector",
        "filename": "craft_mlt_25k.pth",
        "local_path": "models/craft_mlt_25k.pth",
        "description": "Character-level text region detection",
        "size_mb": 80,
    },
    {
        "name": "TrOCR Printed",
        "repo_id": "microsoft/trocr-base-printed",
        "filename": None,
        "local_path": "models/trocr-base-printed",
        "description": "Printed text OCR recognition",
        "size_mb": 1277,
        "type": "transformers",
    },
    {
        "name": "TrOCR Handwritten",
        "repo_id": "microsoft/trocr-base-handwritten",
        "filename": None,
        "local_path": "models/trocr-base-handwritten",
        "description": "Handwritten text OCR recognition",
        "size_mb": 1277,
        "type": "transformers",
    },
    {
        "name": "Table Transformer",
        "repo_id": "microsoft/table-transformer-structure-recognition",
        "filename": None,
        "local_path": "models/table-transformer-structure",
        "description": "Table structure detection",
        "size_mb": 110,
        "type": "transformers",
    },
]


def check_ssl():
    """Configure SSL certificates if needed."""
    try:
        from docvision.ssl_config import configure_ssl_certificates
        configure_ssl_certificates()
    except ImportError:
        try:
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass


def _dir_size_mb(path: Path) -> float:
    """Calculate total size of a directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024 / 1024


def _is_model_present(model_info: dict) -> bool:
    """Check if a model is already present locally."""
    local_path = Path(model_info["local_path"])
    model_type = model_info.get("type", "file")
    
    if model_type == "transformers":
        # A transformers model directory must contain config.json + weights
        return (
            local_path.is_dir()
            and (local_path / "config.json").exists()
            and any(local_path.glob("*.safetensors"))
        )
    else:
        return local_path.is_file()


def download_model(model_info: dict, force: bool = False) -> bool:
    """Download a single model to a local path."""
    name = model_info["name"]
    local_path = Path(model_info["local_path"])
    model_type = model_info.get("type", "file")
    
    if _is_model_present(model_info) and not force:
        size_mb = (
            _dir_size_mb(local_path) if local_path.is_dir()
            else local_path.stat().st_size / 1024 / 1024
        )
        logger.info(f"  {name}: already exists ({size_mb:.1f} MB)")
        return True
    
    logger.info(f"  Downloading {name} (~{model_info['size_mb']} MB)...")
    
    try:
        if model_type == "transformers":
            return _download_transformers_model(model_info)
        else:
            return _download_single_file(model_info)
    except Exception as e:
        logger.error(f"  {name}: download failed: {e}")
        return False


def _download_single_file(model_info: dict) -> bool:
    """Download a single-file model via huggingface_hub."""
    from huggingface_hub import hf_hub_download
    
    local_path = Path(model_info["local_path"])
    hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
        local_dir=str(local_path.parent),
        local_dir_use_symlinks=False,
    )
    
    if local_path.exists():
        size_mb = local_path.stat().st_size / 1024 / 1024
        logger.info(f"  {model_info['name']}: downloaded ({size_mb:.1f} MB)")
        return True
    
    logger.error(f"  {model_info['name']}: download completed but file not found")
    return False


def _download_transformers_model(model_info: dict) -> bool:
    """Download a transformers model and save_pretrained to local path."""
    repo_id = model_info["repo_id"]
    local_path = Path(model_info["local_path"])
    name = model_info["name"]
    
    # Determine which transformers classes to use based on model type
    if "table-transformer" in repo_id:
        from transformers import TableTransformerForObjectDetection, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = TableTransformerForObjectDetection.from_pretrained(repo_id)
    elif "trocr" in repo_id:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained(repo_id)
        model = VisionEncoderDecoderModel.from_pretrained(repo_id)
    else:
        from transformers import AutoModel, AutoProcessor
        processor = AutoProcessor.from_pretrained(repo_id)
        model = AutoModel.from_pretrained(repo_id)
    
    local_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(local_path))
    model.save_pretrained(str(local_path))
    
    size_mb = _dir_size_mb(local_path)
    logger.info(f"  {name}: saved locally ({size_mb:.1f} MB)")
    return True


def download_all(force: bool = False) -> None:
    """Download all required models to local paths."""
    check_ssl()
    
    logger.info("DocVision Model Downloader")
    logger.info("=" * 50)
    
    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)
    
    success = 0
    failed = 0
    
    for model_info in MODELS:
        if download_model(model_info, force):
            success += 1
        else:
            failed += 1
    
    logger.info("=" * 50)
    logger.info(f"Results: {success} downloaded, {failed} failed")
    
    if failed == 0:
        logger.info("All models are stored locally — the app can run fully offline.")


def check_models() -> dict:
    """Check which models are available locally."""
    status = {}
    
    for model_info in MODELS:
        name = model_info["name"]
        local_path = Path(model_info["local_path"])
        
        if _is_model_present(model_info):
            size_mb = (
                _dir_size_mb(local_path) if local_path.is_dir()
                else local_path.stat().st_size / 1024 / 1024
            )
            status[name] = {
                "available": True,
                "path": str(local_path),
                "size_mb": round(size_mb, 1),
            }
        else:
            status[name] = {"available": False, "path": str(local_path)}
    
    return status


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download DocVision models")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--check", action="store_true", help="Check model status only")
    args = parser.parse_args()
    
    if args.check:
        status = check_models()
        all_ok = all(info["available"] for info in status.values())
        
        print("\nModel Status:")
        print("=" * 60)
        total_mb = 0.0
        for name, info in status.items():
            if info["available"]:
                print(f"  ✅ {name}: {info['size_mb']} MB ({info['path']})")
                total_mb += info["size_mb"]
            else:
                print(f"  ❌ {name}: MISSING ({info['path']})")
        print("=" * 60)
        print(f"  Total: {total_mb/1024:.2f} GB")
        if all_ok:
            print("  All models local — fully offline capable ✅")
        else:
            print("  Run: python -m docvision.download_models")
        print()
    else:
        download_all(force=args.force)
