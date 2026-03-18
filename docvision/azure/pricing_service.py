"""
Azure Retail Prices API integration for live pricing.

Fetches ACTUAL pricing from https://prices.azure.com/api/retail/prices
for Azure Document Intelligence and Azure OpenAI services.

NO FALLBACKS - only real Microsoft pricing is used. If the API is
unreachable, an error is raised to ensure accurate cost tracking.

Usage::

    pricing = AzurePricingService()
    di_cost = await pricing.get_di_cost_per_page("prebuilt-layout")
    gpt_input, gpt_output = await pricing.get_gpt_costs("gpt-4o-mini")
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import httpx
from loguru import logger


class PricingUnavailableError(Exception):
    """Raised when Azure pricing cannot be fetched from the API."""
    pass


# ── API Response Logging ────────────────────────────────────────────────────
PRICING_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output", "pricing_logs")


def _log_api_response(
    service_description: str,
    filter_query: str,
    response_data: Dict[str, Any],
    selected_price: Optional[float] = None,
    selected_item: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log the full API response to a file for debugging and auditing.
    
    Creates a JSON file in output/pricing_logs/ with:
    - Timestamp
    - Filter query used
    - Full API response
    - Which price was selected
    """
    try:
        os.makedirs(PRICING_LOG_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_desc = service_description.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"{timestamp}_{safe_desc}.json"
        filepath = os.path.join(PRICING_LOG_DIR, filename)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "service_description": service_description,
            "filter_query": filter_query,
            "api_url": AZURE_PRICES_API_URL,
            "total_items_returned": len(response_data.get("Items", [])),
            "selected_price": selected_price,
            "selected_item": selected_item,
            "full_response": response_data,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, default=str)
        
        logger.info(f"Pricing API response logged to: {filepath}")
        
    except Exception as e:
        logger.warning(f"Failed to log pricing API response: {e}")


# ── Azure Retail Prices API configuration ───────────────────────────────────
AZURE_PRICES_API_URL = "https://prices.azure.com/api/retail/prices"

# Document Intelligence meter patterns
# Found under productName containing 'Document Intelligence'
# Prices are per 1K pages, need to divide by 1000 for per-page cost
DI_METER_PATTERNS: Dict[str, str] = {
    "prebuilt-layout": "Layout",
    "prebuilt-read": "Read",
    "prebuilt-invoice": "Pre-built",
    "prebuilt-receipt": "Pre-built",
    "prebuilt-document": "Pre-built",
    "prebuilt-idDocument": "Pre-built",
    "prebuilt-businessCard": "Pre-built",
    "prebuilt-tax.us.w2": "Pre-built",
}

# Azure OpenAI meter patterns for input/output tokens
# Based on actual API data: gpt-4o-mini uses "gpt-4o-mini-0718-Inp-regnl" style
# Prices are per 1K tokens
GPT_METER_PATTERNS: Dict[str, Tuple[str, str]] = {
    # model_name: (input_search_pattern, output_search_pattern)
    # Patterns that uniquely identify input vs output pricing
    "gpt-4o-mini": ("4o-mini", "Inp"),  # Looks for "4o-mini" and "Inp" 
    "gpt-4o": ("4o-0", "Inp"),  # e.g., gpt-4o-0806
    "gpt-4": ("gpt-4-", "Inp"),
    "gpt-4-turbo": ("4-turbo", "Inp"),
    "gpt-4.1": ("4.1", "Inp"),
    "gpt-4.1-mini": ("4.1-mini", "Inp"),
    "gpt-4.1-nano": ("4.1-nano", "Inp"),
    "gpt-35-turbo": ("35-turbo", "Inp"),
    "gpt-5": ("GPT 5", "inpt"),  # API uses "GPT 5 inpt"
    "gpt-5-mini": ("5-mini", "Inp"),
    "gpt-5-nano": ("5-nano", "Inp"),
    "gpt-5.2": ("5.2", "Inp"),
}


@dataclass
class PriceCache:
    """Simple in-memory cache with TTL."""
    
    data: Dict[str, float] = field(default_factory=dict)
    timestamps: Dict[str, float] = field(default_factory=dict)
    ttl_seconds: float = 3600  # Cache pricing for 1 hour by default
    
    def get(self, key: str) -> Optional[float]:
        """Get cached price if not expired."""
        if key not in self.data:
            return None
        if time.time() - self.timestamps.get(key, 0) > self.ttl_seconds:
            # Expired
            del self.data[key]
            del self.timestamps[key]
            return None
        return self.data[key]
    
    def set(self, key: str, value: float) -> None:
        """Cache a price."""
        self.data[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cached prices."""
        self.data.clear()
        self.timestamps.clear()


class AzurePricingService:
    """
    Fetches LIVE pricing from Azure Retail Prices API.
    
    Thread-safe and async-compatible. Caches prices in memory
    to minimize API calls.
    
    NO FALLBACKS - raises PricingUnavailableError if API fails.
    """
    
    def __init__(
        self,
        region: str = "eastus",
        currency: str = "USD",
        cache_ttl_seconds: float = 3600,
        timeout_seconds: float = 30.0,
        verify_ssl: bool = True,
    ):
        """
        Initialize the pricing service.
        
        Args:
            region: Azure region for pricing (e.g., "eastus", "westus2")
            currency: Currency code (e.g., "USD", "EUR")
            cache_ttl_seconds: How long to cache prices (default 1 hour)
            timeout_seconds: HTTP request timeout
            verify_ssl: Whether to verify SSL certificates (set False for testing)
        """
        self.region = region
        self.currency = currency
        self.timeout = timeout_seconds
        self.verify_ssl = verify_ssl
        self._cache = PriceCache(ttl_seconds=cache_ttl_seconds)
        self._lock = asyncio.Lock()
        self._sync_lock = __import__("threading").Lock()
        
    async def _fetch_price(
        self,
        filter_query: str,
        service_description: str = "Azure service",
    ) -> float:
        """
        Fetch price from Azure Retail Prices API.
        
        Args:
            filter_query: OData filter expression
            service_description: Human-readable description for error messages
            
        Returns:
            Unit price in the configured currency
            
        Raises:
            PricingUnavailableError: If pricing cannot be fetched
        """
        params = {
            "$filter": filter_query,
            "currencyCode": self.currency,
        }
        
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, 
                verify=self.verify_ssl
            ) as client:
                response = await client.get(AZURE_PRICES_API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                items = data.get("Items", [])
                if not items:
                    # Log empty response
                    _log_api_response(
                        service_description, filter_query, data,
                        selected_price=None, selected_item=None
                    )
                    raise PricingUnavailableError(
                        f"No pricing found for {service_description}. "
                        f"Filter: {filter_query}"
                    )
                
                # Find the retail price (prefer Consumption type)
                selected_item = None
                for item in items:
                    if item.get("type") == "Consumption":
                        price = item.get("unitPrice") or item.get("retailPrice")
                        if price is not None:
                            selected_item = item
                            # Log successful response
                            _log_api_response(
                                service_description, filter_query, data,
                                selected_price=price, selected_item=item
                            )
                            logger.debug(
                                f"Found {service_description} pricing: "
                                f"${price:.8f} ({item.get('meterName')})"
                            )
                            return price
                
                # Use first item if no Consumption type found
                price = items[0].get("unitPrice") or items[0].get("retailPrice")
                if price is not None:
                    # Log fallback selection
                    _log_api_response(
                        service_description, filter_query, data,
                        selected_price=price, selected_item=items[0]
                    )
                    return price
                
                raise PricingUnavailableError(
                    f"Price data missing for {service_description}"
                )
                
        except httpx.TimeoutException as e:
            raise PricingUnavailableError(
                f"Timeout fetching Azure pricing for {service_description}: {e}"
            )
        except httpx.HTTPStatusError as e:
            raise PricingUnavailableError(
                f"HTTP error fetching Azure pricing for {service_description}: {e}"
            )
        except PricingUnavailableError:
            raise
        except Exception as e:
            raise PricingUnavailableError(
                f"Error fetching Azure pricing for {service_description}: {e}"
            )
    
    async def _fetch_price_for_gpt(
        self,
        filter_query: str,
        service_description: str,
        is_input: bool,
    ) -> float:
        """
        Fetch GPT price with client-side filtering for regional (non-batch) pricing.
        
        Args:
            filter_query: OData filter expression
            service_description: Human-readable description for error messages
            is_input: True for input tokens, False for output tokens
            
        Returns:
            Price per 1K tokens
            
        Raises:
            PricingUnavailableError: If pricing cannot be fetched
        """
        params = {
            "$filter": filter_query,
            "currencyCode": self.currency,
        }
        
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, 
                verify=self.verify_ssl
            ) as client:
                response = await client.get(AZURE_PRICES_API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                items = data.get("Items", [])
                if not items:
                    # Log empty response
                    _log_api_response(
                        service_description, filter_query, data,
                        selected_price=None, selected_item=None
                    )
                    raise PricingUnavailableError(
                        f"No pricing found for {service_description}. "
                        f"Filter: {filter_query}"
                    )
                
                # Filter items client-side to get regional (non-batch, non-fine-tuned) pricing
                filtered = []
                for item in items:
                    meter = item.get("meterName", "").lower()
                    # Skip batch pricing
                    if "batch" in meter:
                        continue
                    # Skip fine-tuning pricing
                    if "ft" in meter or "fine" in meter:
                        continue
                    # Skip cached pricing
                    if "cchd" in meter or "cach" in meter:
                        continue
                    # Skip TTS/audio pricing
                    if "tts" in meter or "aud" in meter or "transcribe" in meter:
                        continue
                    # Prefer regional over global
                    if item.get("type") == "Consumption":
                        filtered.append(item)
                
                if not filtered:
                    # Fall back to any Consumption type if no filtered results
                    filtered = [i for i in items if i.get("type") == "Consumption"]
                
                if not filtered:
                    # Log no valid pricing
                    _log_api_response(
                        service_description, filter_query, data,
                        selected_price=None, selected_item=None
                    )
                    raise PricingUnavailableError(
                        f"No standard pricing found for {service_description}"
                    )
                
                # Prefer regional pricing
                for item in filtered:
                    meter = item.get("meterName", "").lower()
                    if "regnl" in meter or "regional" in meter:
                        price = item.get("unitPrice") or item.get("retailPrice")
                        # Log regional pricing selection
                        _log_api_response(
                            service_description, filter_query, data,
                            selected_price=price, selected_item=item
                        )
                        logger.debug(
                            f"Found {service_description} regional pricing: "
                            f"${price:.8f} ({item.get('meterName')})"
                        )
                        return price
                
                # Use first filtered item
                price = filtered[0].get("unitPrice") or filtered[0].get("retailPrice")
                # Log fallback selection
                _log_api_response(
                    service_description, filter_query, data,
                    selected_price=price, selected_item=filtered[0]
                )
                logger.debug(
                    f"Found {service_description} pricing: "
                    f"${price:.8f} ({filtered[0].get('meterName')})"
                )
                return price
                
        except PricingUnavailableError:
            raise
        except Exception as e:
            raise PricingUnavailableError(
                f"Error fetching Azure pricing for {service_description}: {e}"
            )
    
    async def get_di_cost_per_page(self, model: str = "prebuilt-layout") -> float:
        """
        Get Document Intelligence cost per page for a model.
        
        Args:
            model: DI model name (e.g., "prebuilt-layout")
            
        Returns:
            Cost per page in configured currency
            
        Raises:
            PricingUnavailableError: If pricing cannot be fetched
        """
        cache_key = f"di:{model}"
        
        # Check cache first
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Using cached DI pricing for {model}: ${cached:.6f}/page")
            return cached
        
        async with self._lock:
            # Double-check cache after acquiring lock
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Build filter query - DI is under 'Foundry Tools' service
            meter_pattern = DI_METER_PATTERNS.get(model, "S0 Pre-built Pages")
            filter_query = (
                f"contains(productName, 'Document Intelligence') and "
                f"armRegionName eq '{self.region}' and "
                f"contains(meterName, '{meter_pattern}')"
            )
            
            # Price is per 1K pages, convert to per page
            price_per_1k = await self._fetch_price(
                filter_query, 
                f"Document Intelligence ({model})"
            )
            price = price_per_1k / 1000.0
            
            logger.info(
                f"Fetched LIVE DI pricing for {model}: ${price:.6f}/page "
                f"(region: {self.region})"
            )
            self._cache.set(cache_key, price)
            
            return price
    
    async def get_gpt_costs(
        self, deployment: str = "gpt-4o-mini"
    ) -> Tuple[float, float]:
        """
        Get Azure OpenAI costs per 1K tokens.
        
        Args:
            deployment: Model deployment name (e.g., "gpt-4o-mini")
            
        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k) in configured currency
            
        Raises:
            PricingUnavailableError: If pricing cannot be fetched
        """
        input_cache_key = f"gpt_input:{deployment}"
        output_cache_key = f"gpt_output:{deployment}"
        
        # Check cache first
        cached_input = self._cache.get(input_cache_key)
        cached_output = self._cache.get(output_cache_key)
        if cached_input is not None and cached_output is not None:
            logger.debug(
                f"Using cached GPT pricing for {deployment}: "
                f"input=${cached_input:.6f}/1K, output=${cached_output:.6f}/1K"
            )
            return (cached_input, cached_output)
        
        async with self._lock:
            # Double-check cache after acquiring lock
            cached_input = self._cache.get(input_cache_key)
            cached_output = self._cache.get(output_cache_key)
            if cached_input is not None and cached_output is not None:
                return (cached_input, cached_output)
            
            # Get meter patterns for this model
            meter_patterns = GPT_METER_PATTERNS.get(deployment)
            
            if meter_patterns:
                model_pattern, _ = meter_patterns
            else:
                # Use deployment name directly for unknown models
                model_pattern = deployment
            
            # Fetch input pricing - search for model pattern + "Inp" in meter
            input_filter = (
                f"contains(productName, 'OpenAI') and "
                f"armRegionName eq '{self.region}' and "
                f"contains(meterName, '{model_pattern}') and "
                f"contains(meterName, 'Inp')"
            )
            input_cost = await self._fetch_price_for_gpt(
                input_filter,
                f"GPT ({deployment}) input tokens",
                is_input=True
            )
            
            # Fetch output pricing - search for model pattern + output patterns
            output_filter = (
                f"contains(productName, 'OpenAI') and "
                f"armRegionName eq '{self.region}' and "
                f"contains(meterName, '{model_pattern}') and "
                f"contains(meterName, 'Outp')"
            )
            output_cost = await self._fetch_price_for_gpt(
                output_filter,
                f"GPT ({deployment}) output tokens",
                is_input=False
            )
            
            logger.info(
                f"Fetched LIVE GPT pricing for {deployment}: "
                f"input=${input_cost:.6f}/1K, output=${output_cost:.6f}/1K "
                f"(region: {self.region})"
            )
            
            self._cache.set(input_cache_key, input_cost)
            self._cache.set(output_cache_key, output_cost)
            
            return (input_cost, output_cost)
    
    # ── Synchronous wrappers for non-async code ─────────────────────────────
    
    def get_di_cost_per_page_sync(self, model: str = "prebuilt-layout") -> float:
        """
        Synchronous wrapper for get_di_cost_per_page.
        
        Raises:
            PricingUnavailableError: If pricing cannot be fetched
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is not None:
            # We're in an async context, need to run in new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.get_di_cost_per_page(model)
                )
                return future.result()
        else:
            return asyncio.run(self.get_di_cost_per_page(model))
    
    def get_gpt_costs_sync(
        self, deployment: str = "gpt-4o-mini"
    ) -> Tuple[float, float]:
        """
        Synchronous wrapper for get_gpt_costs.
        
        Raises:
            PricingUnavailableError: If pricing cannot be fetched
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is not None:
            # We're in an async context, need to run in new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.get_gpt_costs(deployment)
                )
                return future.result()
        else:
            return asyncio.run(self.get_gpt_costs(deployment))
    
    def refresh_cache(self) -> None:
        """Clear the price cache to force fresh fetches."""
        self._cache.clear()
        logger.info("Pricing cache cleared - next calls will fetch live prices")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_prices": len(self._cache.data),
            "ttl_seconds": self._cache.ttl_seconds,
            "region": self.region,
            "currency": self.currency,
            "keys": list(self._cache.data.keys()),
        }
    
    async def verify_api_connection(self) -> bool:
        """
        Verify that the Azure Retail Prices API is reachable.
        
        Returns:
            True if API is reachable
            
        Raises:
            PricingUnavailableError: If API is not reachable
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                verify=self.verify_ssl
            ) as client:
                response = await client.get(
                    AZURE_PRICES_API_URL,
                    params={"$top": "1"}
                )
                response.raise_for_status()
                logger.info("Azure Retail Prices API connection verified")
                return True
        except Exception as e:
            raise PricingUnavailableError(
                f"Cannot connect to Azure Retail Prices API: {e}"
            )


# ── Singleton instance ──────────────────────────────────────────────────────
_pricing_service: Optional[AzurePricingService] = None


def get_pricing_service(
    region: str = None,
    currency: str = "USD",
    cache_ttl_seconds: float = 3600,
    verify_ssl: bool = None,
) -> AzurePricingService:
    """
    Get or create the global pricing service instance.
    
    Configuration can be set via environment variables:
        - AZURE_PRICING_REGION: Azure region (default: eastus)
        - AZURE_PRICING_VERIFY_SSL: Set to "false" to disable SSL verification
    
    Args:
        region: Azure region for pricing (or use AZURE_PRICING_REGION env var)
        currency: Currency code
        cache_ttl_seconds: Cache TTL
        verify_ssl: Whether to verify SSL (or use AZURE_PRICING_VERIFY_SSL env var)
        
    Returns:
        AzurePricingService singleton
    """
    import os
    
    global _pricing_service
    if _pricing_service is None:
        # Read from environment variables if not explicitly provided
        if region is None:
            region = os.environ.get("AZURE_PRICING_REGION", "eastus")
        
        if verify_ssl is None:
            ssl_env = os.environ.get("AZURE_PRICING_VERIFY_SSL", "true").lower()
            verify_ssl = ssl_env not in ("false", "0", "no")
        
        _pricing_service = AzurePricingService(
            region=region,
            currency=currency,
            cache_ttl_seconds=cache_ttl_seconds,
            verify_ssl=verify_ssl,
        )
    return _pricing_service


def reset_pricing_service() -> None:
    """Reset the global pricing service instance."""
    global _pricing_service
    _pricing_service = None
