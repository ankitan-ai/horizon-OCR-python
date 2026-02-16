"""
Azure AI Foundry integration for DocVision.

Provides cloud-based document processing via:
- Azure Document Intelligence (OCR, layout, tables)
- Azure OpenAI GPT-4o Vision (key information extraction)
- Cost tracking and response caching for API call management

These replace the local model pipeline when processing_mode is "azure" or "hybrid".
"""

from docvision.azure.doc_intelligence import AzureDocIntelligenceProvider
from docvision.azure.gpt_vision_kie import GPTVisionExtractor
from docvision.azure.cost_tracker import CostTracker
from docvision.azure.response_cache import ResponseCache

__all__ = [
    "AzureDocIntelligenceProvider",
    "GPTVisionExtractor",
    "CostTracker",
    "ResponseCache",
]
