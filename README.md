# DocVision — Accuracy-First Document AI

DocVision is a production-ready document AI system that maximizes field-level accuracy on invoices, receipts, contracts, purchase orders, forms, bank statements, delivery notes, and mixed scans/digital documents — including handwritten content.

It supports **three processing modes**: a fully local ML pipeline (8 models), Azure AI cloud APIs, or a hybrid of both.

---

## Features

### Core Pipeline (Local)
- **Multi-Engine OCR** — TrOCR for printed & handwritten text, with Tesseract fallback
- **Layout Analysis** — DocLayNet-based YOLO detection for document structure
- **Table Extraction** — Table Transformer (TATR) for accurate table structure recognition
- **Key Information Extraction** — Dual KIE engines (Donut + LayoutLMv3) with rank-and-fuse
- **Field Validation** — Built-in validators for amounts, dates, currencies with consistency checks
- **All Candidates Preserved** — Low-confidence fields include all candidates with uncertainty status

### Azure Cloud Integration
- **Azure Document Intelligence** — Replaces YOLO, CRAFT, TrOCR, TATR, and Tesseract with a single cloud call
- **Azure OpenAI GPT Vision** — Replaces Donut + LayoutLMv3 KIE with GPT-4o vision
- **Processing Modes** — `local`, `azure`, or `hybrid` (run both and merge results)
- **Cost Tracking** — Per-request cost estimation for Document Intelligence and GPT Vision calls
- **Response Caching** — SHA-256 content-addressed cache to avoid redundant API calls

### Web UI
- **Upload & Scan** — Drag-and-drop or file picker with PDF page preview
- **Multi-File Batch Upload** — Process multiple documents in parallel
- **Artifacts Viewer** — Browse intermediate outputs (layout overlays, text polygons, table structure, OCR)
- **Confidence Highlighting** — Color-coded badges (green ≥ 90%, yellow ≥ 70%, red < 70%) on every text line, field, and table cell
- **Field Editing** — Inline field editor with confidence badges, plus raw JSON editor with toggle
- **Processing History** — Browse and reload past jobs
- **Cost & Usage Dashboard** — Live cost breakdown by service, cache hit stats
- **Save to Disk** — Export final JSON to the output directory
- **Download** — Download processed JSON directly from the browser

### Infrastructure
- **REST API & CLI** — FastAPI server and Typer CLI for flexible integration
- **Docker Ready** — CPU and GPU Dockerfiles with docker-compose profiles (dev, prod, gpu, batch)
- **291 Tests** — Unit, integration, web feature, and Azure smoke tests

---

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) CUDA-capable GPU for faster local model inference
- (Optional) System Tesseract binary for OCR fallback
- (Optional) Poppler for PDF page preview via `pdf2image`

### Installation

```bash
# Clone the repository
git clone https://github.com/ankitan-ai/horizon-OCR-python.git
cd horizon-OCR-python

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .\.venv\Scripts\activate     # Windows

# Install core dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# (Optional) Install Azure cloud extras
pip install -e ".[azure]"

# (Optional) Install PDF page preview support
pip install -e ".[preview]"
```

### Web UI (Recommended)

```bash
# Start the web server
python -m uvicorn docvision.web.app:app --host 0.0.0.0 --port 8080

# Open in browser
# http://localhost:8080
```

The web UI provides five tabs:

| Tab | Description |
|-----|-------------|
| **Upload & Scan** | Upload documents, select processing mode (local/azure/hybrid), view PDF preview |
| **Artifacts** | Browse intermediate outputs — layout overlays, text polygons, tables, OCR |
| **Output** | View & edit extracted JSON — toggle between field editor and raw JSON |
| **History** | Browse past processing jobs, reload results |
| **Cost & Usage** | Azure API cost breakdown, cache statistics |

### CLI Usage

```bash
# Process a single document
docvision process invoice.pdf -o ./output

# Process with debug artifacts
docvision process invoice.pdf --artifacts

# Batch process a directory
docvision batch ./documents -o ./output --pattern "*.pdf" --parallel

# Start API server (legacy — prefer the web UI)
docvision serve --port 8080

# Show configuration
docvision config --show

# Generate example config
docvision config --generate -o config.yaml
```

### API Usage

```bash
# Process a document
curl -X POST "http://localhost:8080/api/process" \
  -F "file=@invoice.pdf" \
  -F "mode=local"

# Batch process multiple files
curl -X POST "http://localhost:8080/api/process/batch" \
  -F "files=@invoice1.pdf" \
  -F "files=@invoice2.pdf"

# Check job status
curl http://localhost:8080/api/jobs/{job_id}

# Get result
curl http://localhost:8080/api/jobs/{job_id}/result

# View processing history
curl http://localhost:8080/api/history

# View cost stats
curl http://localhost:8080/api/costs
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

---

## Configuration

### Processing Modes

| Mode | Description |
|------|-------------|
| `local` | All processing runs locally using 8 ML models (default) |
| `azure` | All processing uses Azure Document Intelligence + GPT Vision |
| `hybrid` | Runs both local and cloud pipelines, merges results for maximum accuracy |

### Config File

Create a `config.yaml` from the included example:

```bash
cp config.example.yaml config.yaml
```

Key sections:

```yaml
runtime:
  device: auto        # auto, cpu, cuda, mps
  workers: 4

pdf:
  dpi: 350
  max_pages: null

preprocess:
  deskew: true
  dewarp: true
  denoise: true
  clahe: true

thresholds:
  trocr_min_conf: 0.75
  low_confidence_threshold: 0.5
  min_confidence_for_output: 0.2

# Azure AI Foundry (optional)
azure:
  processing_mode: local    # local | azure | hybrid
  doc_intelligence_endpoint: ""
  doc_intelligence_key: ""
  openai_endpoint: ""
  openai_key: ""
  openai_deployment: gpt-4o
  use_gpt_vision_kie: true
```

### Azure Credentials

Set credentials in `config.yaml` or via environment variables (`.env` file supported):

```bash
# .env
AZURE_DOC_INTELLIGENCE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOC_INTELLIGENCE_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_KEY=your-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

See [.env.example](.env.example) for a complete template.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Document                           │
│                    (PDF / Image / Scan)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
┌──────────────────────┐   ┌──────────────────────────┐
│     LOCAL PIPELINE   │   │    AZURE CLOUD PIPELINE   │
│                      │   │                          │
│  Preprocessing       │   │  Document Intelligence   │
│   • Deskew/Dewarp    │   │   • Layout + OCR + Tables│
│   • Denoise/CLAHE    │   │                          │
│                      │   │  GPT Vision KIE          │
│  Detection           │   │   • Field extraction     │
│   • YOLO (DocLayNet) │   │   • Structured JSON      │
│   • CRAFT Text       │   │                          │
│   • Table Transformer│   │  Cost Tracker            │
│                      │   │   • Per-request costs    │
│  OCR Recognition     │   │                          │
│   • TrOCR Printed    │   │  Response Cache          │
│   • TrOCR Handwritten│   │   • SHA-256 dedup        │
│   • Tesseract Backup │   └──────────┬───────────────┘
│                      │              │
│  KIE Extraction      │              │
│   • Donut (OCR-Free) │              │
│   • LayoutLMv3       │              │
│   • Rank & Fuse      │              │
│                      │              │
│  Validation          │              │
│   • Amount/Date/Curr │              │
│   • Cross-field      │              │
└──────────┬───────────┘              │
           │                          │
           └──────────┬───────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Structured JSON Output                        │
│   • Document metadata    • Confidence scores                    │
│   • Page-level content   • All candidates preserved             │
│   • Extracted fields     • Validation results                   │
│   • Tables with cells    • Status flags                         │
└─────────────────────────────────────────────────────────────────┘
```

---

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
    "processing_time_seconds": 2.5,
    "processing_mode": "azure"
  },
  "page_count": 1,
  "fields": [
    {
      "name": "invoice_number",
      "value": "INV-001",
      "confidence": 0.95,
      "status": "confident",
      "source": "doc_intelligence",
      "candidates": [
        {"value": "INV-001", "confidence": 0.95, "source": "doc_intelligence"}
      ]
    },
    {
      "name": "total",
      "value": "$1,234.56",
      "confidence": 0.42,
      "status": "uncertain",
      "candidates": [
        {"value": "$1,234.56", "confidence": 0.42, "source": "donut"},
        {"value": "$1,234.00", "confidence": 0.38, "source": "layoutlmv3"}
      ]
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

---

## Docker

### CPU

```bash
docker build -t docvision .
docker run -p 8080:8080 -v ./data:/app/data docvision
```

### GPU

```bash
docker build -f Dockerfile.gpu -t docvision:gpu .
docker run --gpus all -p 8080:8080 -v ./data:/app/data docvision:gpu
```

### Docker Compose

```bash
# Development mode (with hot reload)
docker-compose --profile dev up

# Production
docker-compose up docvision

# GPU production
docker-compose --profile gpu up docvision-gpu

# Batch processing
docker-compose --profile batch run docvision-worker
```

Azure credentials are passed as environment variables in `docker-compose.yml` — set them in your `.env` file or export them before running.

---

## Testing

```bash
# Run all tests (291 tests)
pytest

# Run with coverage
pytest --cov=docvision --cov-report=html

# Run specific test suites
pytest tests/test_web_features.py     # Web UI feature tests (39)
pytest tests/test_azure_smoke.py      # Azure smoke tests (15)
pytest tests/test_cost_cache_batch.py # Cost/cache/batch tests (36)

# Run with verbose output
pytest -v --tb=short
```

---

## Project Structure

```
docvision/
├── __init__.py               # Package init, version
├── config.py                 # Configuration dataclasses
├── types.py                  # Pydantic schemas (Document, Page, Field, Table...)
├── ssl_config.py             # SSL/TLS certificate configuration (certifi)
├── download_models.py        # Model download utility
├── io/
│   ├── pdf.py                # PDF loading & rasterization
│   ├── image.py              # Image loading & normalization
│   └── artifacts.py          # Debug overlay generation
├── preprocess/
│   ├── geometry.py           # Deskew, dewarp
│   └── enhance.py            # Denoise, CLAHE, content detection
├── detect/
│   ├── layout_doclaynet.py   # YOLO layout detection
│   ├── text_craft.py         # CRAFT text detection
│   └── table_tatr.py         # Table Transformer detection
├── ocr/
│   ├── trocr.py              # TrOCR recognizer (printed + handwritten)
│   ├── tesseract.py          # Tesseract backup
│   └── crops.py              # Image cropping utilities
├── kie/
│   ├── donut_runner.py       # Donut KIE (OCR-free)
│   ├── layoutlmv3_runner.py  # LayoutLMv3 KIE (token classification)
│   ├── fuse.py               # Rank-and-fuse logic
│   └── validators.py         # Field validators (amount, date, currency)
├── pipeline/
│   └── orchestrator.py       # Main processing pipeline
├── azure/
│   ├── doc_intelligence.py   # Azure Document Intelligence client
│   ├── gpt_vision_kie.py     # Azure OpenAI GPT Vision KIE
│   ├── cost_tracker.py       # Per-request cost estimation
│   └── response_cache.py     # SHA-256 content-addressed response cache
├── web/
│   ├── app.py                # FastAPI web server (all API endpoints)
│   └── index.html            # Single-page application (inline CSS/JS)
├── api/
│   └── server.py             # Legacy FastAPI server (CLI-based)
└── cli/
    └── main.py               # Typer CLI entry point
```

---

## Dependencies

### Core (always required)

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning inference |
| `transformers`, `timm` | Model architectures (TrOCR, LayoutLMv3, Donut) |
| `ultralytics` | YOLO layout detection |
| `opencv-python-headless` | Image processing |
| `Pillow` | Image I/O |
| `PyMuPDF` | PDF rasterization & preview fallback |
| `fastapi`, `uvicorn` | Web server |
| `pydantic` | Data validation & schemas |
| `requests`, `httpx` | HTTP clients |
| `certifi` | SSL certificate handling |
| `typer`, `rich` | CLI framework |
| `PyYAML` | Configuration parsing |
| `loguru` | Structured logging |
| `numpy`, `scipy`, `tqdm` | Numerical utilities |

### Optional Extras

```bash
pip install -e ".[azure]"      # azure-ai-documentintelligence, openai
pip install -e ".[preview]"    # pdf2image (requires poppler)
pip install -e ".[tesseract]"  # pytesseract (requires system Tesseract)
pip install -e ".[dev]"        # pytest, black, ruff, mypy
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
