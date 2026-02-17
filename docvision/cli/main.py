"""
DocVision CLI - Command line interface for document processing.

Usage:
    docvision process <file> [options]
    docvision batch <directory> [options]
    docvision serve [options]
    docvision config [options]
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import json

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

from docvision import __version__

# Create CLI app
app = typer.Typer(
    name="docvision",
    help="DocVision - Accuracy-first document AI processing",
    add_completion=False
)

console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"DocVision version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    )
):
    """DocVision - Accuracy-first document AI processing."""
    pass


@app.command()
def process(
    file: Path = typer.Argument(..., help="Path to PDF or image file"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output directory (default: ./output)"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config file"
    ),
    artifacts: bool = typer.Option(
        False, "--artifacts", "-a",
        help="Save debug artifacts (overlays, preprocessed images)"
    ),
    no_preprocess: bool = typer.Option(
        False, "--no-preprocess",
        help="Skip preprocessing"
    ),
    no_ocr: bool = typer.Option(
        False, "--no-ocr",
        help="Skip OCR recognition"
    ),
    no_donut: bool = typer.Option(
        False, "--no-donut",
        help="Skip Donut KIE extraction"
    ),
    no_layoutlmv3: bool = typer.Option(
        False, "--no-layoutlmv3",
        help="Skip LayoutLMv3 KIE extraction"
    ),
    no_validators: bool = typer.Option(
        False, "--no-validators",
        help="Skip field validation"
    ),
    pretty: bool = typer.Option(
        True, "--pretty/--compact",
        help="Pretty-print JSON output"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Suppress progress output"
    ),
):
    """
    Process a single document and extract structured information.
    
    Example:
        docvision process invoice.pdf -o ./results --artifacts
    """
    # Validate input file
    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)
    
    # Load config
    from docvision.config import load_config, Config
    
    if config and config.exists():
        cfg = load_config(str(config))
    else:
        cfg = Config()
    
    # Override output dir if specified
    if output:
        cfg.output.dir = str(output)
    
    cfg.output.pretty_json = pretty
    
    # Pre-resolve Azure hostnames via DoH (bypasses VPN private link DNS issues)
    if cfg.azure.processing_mode in ("azure", "hybrid") or cfg.azure.is_azure_ready:
        try:
            from docvision.dns_config import configure_doh_for_azure
            configure_doh_for_azure(
                di_endpoint=cfg.azure.doc_intelligence_endpoint,
                openai_endpoint=cfg.azure.openai_endpoint,
            )
        except Exception:
            pass

    # Initialize processor
    from docvision.pipeline import DocumentProcessor, ProcessingOptions
    
    if not quiet:
        console.print(f"[blue]DocVision[/blue] Processing: {file}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=quiet
    ) as progress:
        task = progress.add_task("Initializing...", total=None)
        
        processor = DocumentProcessor(cfg)
        
        progress.update(task, description="Processing document...")
        
        options = ProcessingOptions(
            preprocess=not no_preprocess,
            detect_layout=True,
            detect_text=True,
            detect_tables=True,
            run_ocr=not no_ocr,
            run_donut=not no_donut,
            run_layoutlmv3=not no_layoutlmv3,
            run_validators=not no_validators,
            save_artifacts=artifacts,
            save_json=True,
            output_dir=str(output) if output else None
        )
        
        result = processor.process(str(file), options)
        
        progress.update(task, description="Done!")
    
    if result.success:
        doc = result.document
        
        if not quiet:
            # Print summary
            console.print()
            console.print(f"[green]✓ Success[/green]")
            console.print(f"  Document ID: {doc.id}")
            console.print(f"  Pages: {doc.page_count}")
            console.print(f"  Fields extracted: {len(doc.fields)}")
            console.print(f"  Tables detected: {len(doc.tables)}")
            console.print(f"  Processing time: {doc.metadata.processing_time_seconds:.2f}s")
            
            # Print fields summary
            if doc.fields:
                console.print()
                table = Table(title="Extracted Fields")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")
                table.add_column("Confidence", justify="right")
                table.add_column("Status", style="yellow")
                
                for field in doc.fields[:20]:  # Limit to 20
                    conf_color = "green" if field.confidence >= 0.8 else "yellow" if field.confidence >= 0.5 else "red"
                    table.add_row(
                        field.name,
                        str(field.value)[:50] + ("..." if len(str(field.value)) > 50 else ""),
                        f"[{conf_color}]{field.confidence:.2f}[/{conf_color}]",
                        field.status.value
                    )
                
                if len(doc.fields) > 20:
                    table.add_row("...", f"({len(doc.fields) - 20} more)", "", "")
                
                console.print(table)
            
            # Print validation summary
            if doc.validation:
                console.print()
                v = doc.validation
                status = "green" if v.passed else "red"
                console.print(f"[{status}]Validation: {v.passed_checks}/{v.total_checks} checks passed[/{status}]")
                
                if v.issues:
                    for issue in v.issues[:5]:
                        console.print(f"  [yellow]⚠[/yellow] {issue}")
    else:
        console.print(f"[red]✗ Error:[/red] {result.error}")
        raise typer.Exit(1)


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory containing documents"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output directory (default: ./output)"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config file"
    ),
    pattern: str = typer.Option(
        "*.pdf", "--pattern", "-p",
        help="File pattern to match (e.g., *.pdf, *.png)"
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r",
        help="Search subdirectories"
    ),
    parallel: bool = typer.Option(
        False, "--parallel",
        help="Process files in parallel"
    ),
):
    """
    Process multiple documents in a directory.
    
    Example:
        docvision batch ./invoices -o ./results --pattern "*.pdf" --parallel
    """
    if not directory.exists():
        console.print(f"[red]Error:[/red] Directory not found: {directory}")
        raise typer.Exit(1)
    
    # Find files
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    if not files:
        console.print(f"[yellow]No files found matching pattern:[/yellow] {pattern}")
        raise typer.Exit(0)
    
    console.print(f"[blue]DocVision[/blue] Found {len(files)} files to process")
    
    # Load config
    from docvision.config import load_config, Config
    from docvision.pipeline import DocumentProcessor, ProcessingOptions
    
    if config and config.exists():
        cfg = load_config(str(config))
    else:
        cfg = Config()
    
    if output:
        cfg.output.dir = str(output)
    
    # Initialize processor
    processor = DocumentProcessor(cfg)
    
    options = ProcessingOptions(
        save_json=True,
        output_dir=str(output) if output else None
    )
    
    # Process files
    with Progress(console=console) as progress:
        task = progress.add_task("Processing...", total=len(files))
        
        results = []
        success_count = 0
        
        for file in files:
            progress.update(task, description=f"Processing: {file.name}")
            
            result = processor.process(str(file), options)
            results.append((file, result))
            
            if result.success:
                success_count += 1
            
            progress.advance(task)
    
    # Print summary
    console.print()
    console.print(f"[green]Completed:[/green] {success_count}/{len(files)} files processed successfully")
    
    # Show failures
    failures = [(f, r) for f, r in results if not r.success]
    if failures:
        console.print()
        console.print("[red]Failed files:[/red]")
        for file, result in failures:
            console.print(f"  - {file.name}: {result.error}")


@app.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0", "--host", "-h",
        help="Host to bind to"
    ),
    port: int = typer.Option(
        8000, "--port", "-p",
        help="Port to listen on"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config file"
    ),
    reload: bool = typer.Option(
        False, "--reload",
        help="Enable auto-reload (development)"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w",
        help="Number of worker processes"
    ),
):
    """
    Start the DocVision API server.
    
    Example:
        docvision serve --port 8080 --workers 4
    """
    console.print(f"[blue]DocVision[/blue] Starting API server on {host}:{port}")
    
    # Set config path in environment
    if config:
        os.environ["DOCVISION_CONFIG"] = str(config)
    
    try:
        import uvicorn
        
        uvicorn.run(
            "docvision.api.server:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1
        )
    except ImportError:
        console.print("[red]Error:[/red] uvicorn not installed. Run: pip install uvicorn")
        raise typer.Exit(1)


@app.command("config")
def config_cmd(
    show: bool = typer.Option(
        False, "--show", "-s",
        help="Show current configuration"
    ),
    generate: bool = typer.Option(
        False, "--generate", "-g",
        help="Generate example config file"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path for generated config"
    ),
):
    """
    Configuration management.
    
    Example:
        docvision config --generate -o config.yaml
        docvision config --show
    """
    from docvision.config import Config
    
    if show:
        # Show current config
        config = Config()
        
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        
        # Runtime settings
        table.add_row("Device", config.runtime.device)
        table.add_row("Workers", str(config.runtime.workers))
        
        # PDF settings
        table.add_row("PDF DPI", str(config.pdf.dpi))
        table.add_row("PDF Max Pages", str(config.pdf.max_pages))
        
        # Model settings
        table.add_row("TrOCR Printed", config.models.trocr_printed)
        table.add_row("TrOCR Handwritten", config.models.trocr_handwritten)
        table.add_row("Donut", config.models.donut)
        table.add_row("LayoutLMv3", config.models.layoutlmv3)
        
        # Output settings
        table.add_row("Output Dir", config.output.dir)
        table.add_row("Artifacts Dir", config.artifacts.dir)
        
        console.print(table)
    
    elif generate:
        # Generate example config
        example_config = '''# DocVision Configuration
# =======================

runtime:
  device: auto  # auto, cpu, cuda, mps
  workers: 4
  log_level: INFO

pdf:
  dpi: 300
  max_pages: null  # null for no limit

preprocess:
  deskew: true
  dewarp: true
  denoise: true
  clahe: true
  sharpen: false

models:
  # Layout detection
  layout: null  # null = use DocLayNet YOLO default
  craft: null   # null = use CRAFT default
  
  # Table detection
  tatr: "microsoft/table-transformer-detection"
  
  # OCR models
  trocr_printed: "microsoft/trocr-base-printed"
  trocr_handwritten: "microsoft/trocr-base-handwritten"
  
  # KIE models
  donut: "naver-clova-ix/donut-base-finetuned-docvqa"
  layoutlmv3: "microsoft/layoutlmv3-base"

kie:
  use_donut: true
  use_layoutlmv3: true
  donut_weight: 0.6
  layoutlmv3_weight: 0.4
  ocr_weight: 0.3

thresholds:
  layout_confidence: 0.5
  text_detection_confidence: 0.5
  ocr_confidence: 0.5
  field_confidence_high: 0.8
  field_confidence_low: 0.3
  reroute_to_tesseract_below: 0.4

validators:
  enable_amount: true
  enable_date: true
  enable_currency: true
  enable_consistency: true

artifacts:
  enable: false
  dir: "./artifacts"
  save_layout_overlay: true
  save_text_polygons: true
  save_table_structure: true
  save_ocr_overlay: true
  save_preprocessed: true

output:
  dir: "./output"
  format: json
  pretty_json: true
  include_raw_text: true
  include_coordinates: true
'''
        
        if output:
            output.write_text(example_config)
            console.print(f"[green]Config written to:[/green] {output}")
        else:
            console.print(example_config)
    
    else:
        console.print("Use --show to display config or --generate to create example config")


@app.command()
def info():
    """Show system and model information."""
    import platform
    
    table = Table(title="DocVision System Info")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Version", __version__)
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())
    
    # Check PyTorch
    try:
        import torch
        table.add_row("PyTorch", torch.__version__)
        table.add_row("CUDA Available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            table.add_row("CUDA Device", torch.cuda.get_device_name(0))
    except ImportError:
        table.add_row("PyTorch", "[red]Not installed[/red]")
    
    # Check other dependencies
    deps = [
        ("transformers", "transformers"),
        ("opencv", "cv2"),
        ("pydantic", "pydantic"),
        ("fastapi", "fastapi"),
        ("tesseract", "pytesseract"),
    ]
    
    for name, module in deps:
        try:
            m = __import__(module)
            version = getattr(m, "__version__", "installed")
            table.add_row(name, version)
        except ImportError:
            table.add_row(name, "[red]Not installed[/red]")
    
    console.print(table)


def main_entry():
    """Entry point for console script."""
    app()


if __name__ == "__main__":
    main_entry()
