# DocVision - Accuracy-First Document AI

DocVision is a production-ready document AI system that maximizes field-level accuracy on invoices, receipts, contracts, purchase orders, forms, bank statements, delivery notes, and mixed scans/digital documents—including handwritten content.

## Features

- **Multi-Engine OCR**: TrOCR for printed and handwritten text, with Tesseract backup
- **Layout Analysis**: DocLayNet-based layout detection for document structure understanding
- **Table Extraction**: Table Transformer (TATR) for accurate table structure recognition
- **Key Information Extraction**: Dual KIE engines (Donut + LayoutLMv3) with rank-and-fuse
- **Field Validation**: Built-in validators for amounts, dates, currencies with consistency checks
- **All Candidates Preserved**: Low-confidence fields include all candidates with uncertainty status
- **REST API & CLI**: FastAPI server and Typer CLI for flexible integration
- **Docker Ready**: CPU and GPU Dockerfiles with docker-compose support

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/example/docvision.git
cd docvision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### CLI Usage

```bash
# Process a single document
docvision process invoice.pdf -o ./output

# Process with debug artifacts
docvision process invoice.pdf --artifacts

# Batch process a directory
docvision batch ./documents -o ./output --pattern "*.pdf" --parallel

# Start API server
docvision serve --port 8000

# Show configuration
docvision config --show

# Generate example config
docvision config --generate -o config.yaml
```

### API Usage

```bash
# Start the server
docvision serve

# Process a document
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.pdf"

# Health check
curl http://localhost:8000/health

# Async processing
curl -X POST "http://localhost:8000/process/async" \
  -F "file=@invoice.pdf"

# Check job status
curl http://localhost:8000/jobs/{job_id}
```

### Python API

```python
from docvision.pipeline import DocumentProcessor, ProcessingOptions
from docvision.config import load_config

# Load configuration
config = load_config("config.yaml")

# Create processor
processor = DocumentProcessor(config)

# Process document
result = processor.process("invoice.pdf")

if result.success:
    doc = result.document
    
    # Access extracted fields
    for field in doc.fields:
        print(f"{field.name}: {field.value} (confidence: {field.confidence:.2f})")
    
    # Access tables
    for table in doc.tables:
        print(f"Table: {table.rows}x{table.cols}")
        for row in table.to_list():
            print(row)
else:
    print(f"Error: {result.error}")
```

## Configuration

Create a `config.yaml` file to customize behavior:

```yaml
runtime:
  device: auto  # auto, cpu, cuda, mps
  workers: 4

pdf:
  dpi: 300
  max_pages: null

preprocess:
  deskew: true
  dewarp: true
  denoise: true
  clahe: true

models:
  trocr_printed: "microsoft/trocr-base-printed"
  trocr_handwritten: "microsoft/trocr-base-handwritten"
  donut: "naver-clova-ix/donut-base-finetuned-docvqa"
  layoutlmv3: "microsoft/layoutlmv3-base"

thresholds:
  field_confidence_high: 0.8
  field_confidence_low: 0.3

output:
  dir: "./output"
  pretty_json: true
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Document                           │
│                    (PDF / Image / Scan)                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Preprocessing                               │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │  Deskew  │  │  Denoise │  │  CLAHE   │  │  Sharpen │       │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Detection                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   │ Layout YOLO  │  │  CRAFT Text  │  │ Table TATR   │         │
│   │ (DocLayNet)  │  │  Detection   │  │              │         │
│   └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OCR Recognition                               │
│   ┌──────────────────────┐  ┌──────────────────────┐           │
│   │ TrOCR (Printed)      │  │ TrOCR (Handwritten)  │           │
│   │ + Tesseract Backup   │  │                      │           │
│   └──────────────────────┘  └──────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Key Information Extraction                     │
│   ┌──────────────────────┐  ┌──────────────────────┐           │
│   │  Donut (OCR-Free)    │  │  LayoutLMv3 (Token)  │           │
│   └──────────────────────┘  └──────────────────────┘           │
│                      │              │                           │
│                      └──────┬───────┘                           │
│                             ▼                                   │
│                    ┌──────────────┐                             │
│                    │ Rank & Fuse  │                             │
│                    └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Validation                                 │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │
│   │ Amount  │  │  Date   │  │Currency │  │   Consistency   │  │
│   └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Structured JSON Output                       │
│   • Document metadata   • All candidates preserved               │
│   • Page-level content  • Confidence scores                      │
│   • Extracted fields    • Validation results                     │
│   • Tables with cells   • Status flags                           │
└─────────────────────────────────────────────────────────────────┘
```

## Output JSON Schema

See [JSON_SCHEMA.md](JSON_SCHEMA.md) for the complete output schema documentation.

Example output:

```json
{
  "id": "doc-a1b2c3d4",
  "metadata": {
    "filename": "invoice.pdf",
    "file_type": "pdf",
    "processed_at": "2024-01-15T10:30:00Z",
    "processing_time_seconds": 2.5
  },
  "page_count": 1,
  "fields": [
    {
      "name": "invoice_number",
      "value": "INV-001",
      "confidence": 0.95,
      "status": "confident",
      "candidates": [
        {"value": "INV-001", "confidence": 0.95, "source": "donut"},
        {"value": "INV-001", "confidence": 0.92, "source": "layoutlmv3"}
      ],
      "page": 1
    },
    {
      "name": "total",
      "value": "$1,234.56",
      "confidence": 0.42,
      "status": "uncertain",
      "candidates": [
        {"value": "$1,234.56", "confidence": 0.42, "source": "donut"},
        {"value": "$1,234.00", "confidence": 0.38, "source": "layoutlmv3"}
      ],
      "page": 1
    }
  ],
  "tables": [...],
  "validation": {
    "passed": true,
    "total_checks": 5,
    "passed_checks": 5
  }
}
```

## Docker

### CPU

```bash
# Build
docker build -t docvision .

# Run API server
docker run -p 8000:8000 -v ./data:/app/data docvision

# Run CLI
docker run -v ./data:/app/data -v ./output:/app/output \
  docvision docvision process /app/data/invoice.pdf -o /app/output
```

### GPU

```bash
# Build
docker build -f Dockerfile.gpu -t docvision:gpu .

# Run with GPU
docker run --gpus all -p 8000:8000 -v ./data:/app/data docvision:gpu
```

### Docker Compose

```bash
# Development mode
docker-compose --profile dev up

# Production
docker-compose up docvision

# GPU production
docker-compose --profile gpu up docvision-gpu

# Batch processing
docker-compose --profile batch run docvision-worker
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=docvision --cov-report=html

# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration
```

## Project Structure

```
docvision/
├── __init__.py           # Package init, version
├── config.py             # Configuration dataclasses
├── types.py              # Pydantic schemas
├── io/
│   ├── pdf.py            # PDF loading
│   ├── image.py          # Image loading
│   └── artifacts.py      # Debug overlay generation
├── preprocess/
│   ├── geometry.py       # Deskew, dewarp
│   └── enhance.py        # Denoise, CLAHE, content detection
├── detect/
│   ├── layout_doclaynet.py  # Layout detection
│   ├── text_craft.py        # Text detection
│   └── table_tatr.py        # Table detection
├── ocr/
│   ├── trocr.py          # TrOCR recognizer
│   ├── tesseract.py      # Tesseract backup
│   └── crops.py          # Image cropping utilities
├── kie/
│   ├── donut_runner.py   # Donut KIE
│   ├── layoutlmv3_runner.py  # LayoutLMv3 KIE
│   ├── fuse.py           # Rank-and-fuse logic
│   └── validators.py     # Field validators
├── pipeline/
│   └── orchestrator.py   # Main pipeline
├── api/
│   └── server.py         # FastAPI server
└── cli/
    └── main.py           # Typer CLI
```

## License

MIT License - see [LICENSE](LICENSE) for details.
