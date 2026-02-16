"""
Field validators for post-processing and verification.

Validates extracted fields against rules for:
- Amount calculations (total = sum of lines)
- Date formats and validity
- Currency codes
- ID patterns (invoice numbers, PO numbers)
- Checksums and cross-references
"""

from typing import Any, Optional, List, Dict, Callable, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import re
from loguru import logger

from docvision.types import Field, ValidatorResult


class Validator(ABC):
    """Base class for field validators."""
    
    name: str = "base"
    
    @abstractmethod
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidatorResult:
        """
        Validate a field value.
        
        Args:
            value: Value to validate
            context: Optional context (other fields, document data)
            
        Returns:
            ValidatorResult with pass/fail and details
        """
        pass
    
    def __call__(self, value: Any, context: Optional[Dict] = None) -> ValidatorResult:
        return self.validate(value, context)


class AmountValidator(Validator):
    """Validates monetary amounts and calculations."""
    
    name = "amount"
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize amount validator.
        
        Args:
            tolerance: Tolerance for floating point comparisons
        """
        self.tolerance = tolerance
    
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidatorResult:
        """Validate amount format and value."""
        # Try to parse as number
        try:
            parsed = self._parse_amount(value)
            
            if parsed is None:
                return ValidatorResult(
                    name=self.name,
                    passed=False,
                    message="Could not parse as amount",
                    details={"original_value": str(value)}
                )
            
            # Basic validation: should be positive
            if parsed < 0:
                return ValidatorResult(
                    name=self.name,
                    passed=False,
                    message="Amount is negative",
                    details={"parsed_value": parsed}
                )
            
            return ValidatorResult(
                name=self.name,
                passed=True,
                message="Valid amount",
                details={"parsed_value": parsed}
            )
        
        except Exception as e:
            return ValidatorResult(
                name=self.name,
                passed=False,
                message=f"Validation error: {str(e)}",
                details={"original_value": str(value)}
            )
    
    def _parse_amount(self, value: Any) -> Optional[float]:
        """Parse amount from various formats."""
        if isinstance(value, (int, float)):
            return float(value)
        
        if not isinstance(value, str):
            return None
        
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[$€£¥₹]', '', str(value))
        cleaned = cleaned.strip()
        
        # Handle different decimal separators
        # E.g., "1,234.56" or "1.234,56"
        if ',' in cleaned and '.' in cleaned:
            # Determine which is decimal separator
            if cleaned.rfind(',') > cleaned.rfind('.'):
                # European format: 1.234,56
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                # US format: 1,234.56
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Could be decimal or thousand separator
            if re.match(r'^\d+,\d{2}$', cleaned):
                # Likely decimal: 123,45
                cleaned = cleaned.replace(',', '.')
            else:
                # Likely thousand: 1,234
                cleaned = cleaned.replace(',', '')
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def validate_total(
        self,
        total: float,
        items: List[float],
        tax: Optional[float] = None
    ) -> ValidatorResult:
        """
        Validate that total equals sum of items (+ tax).
        
        Args:
            total: Claimed total amount
            items: List of item amounts
            tax: Optional tax amount
            
        Returns:
            ValidatorResult
        """
        calculated = sum(items)
        if tax is not None:
            calculated += tax
        
        diff = abs(total - calculated)
        
        if diff <= self.tolerance:
            return ValidatorResult(
                name="total_check",
                passed=True,
                message="Total matches sum of items",
                details={
                    "total": total,
                    "calculated": calculated,
                    "difference": diff
                }
            )
        else:
            return ValidatorResult(
                name="total_check",
                passed=False,
                message=f"Total mismatch: {total} vs calculated {calculated}",
                details={
                    "total": total,
                    "calculated": calculated,
                    "difference": diff
                }
            )


class DateValidator(Validator):
    """Validates date formats and values."""
    
    name = "date"
    
    def __init__(self, formats: Optional[List[str]] = None):
        """
        Initialize date validator.
        
        Args:
            formats: List of accepted date formats
        """
        self.formats = formats or [
            # ── Date + time (most specific first) ───────────────
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            # ── Date only ───────────────────────────────────────
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%Y/%m/%d",
        ]
    
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidatorResult:
        """Validate date format and reasonableness."""
        if not value:
            return ValidatorResult(
                name=self.name,
                passed=False,
                message="Empty date value"
            )
        
        str_value = str(value).strip()
        
        # Try to parse date
        parsed_date, format_used = self._parse_date(str_value)
        
        if parsed_date is None:
            return ValidatorResult(
                name=self.name,
                passed=False,
                message="Could not parse date",
                details={"original_value": str_value}
            )
        
        # Validate reasonableness (not too far in past or future)
        now = datetime.now()
        years_diff = abs((now - parsed_date).days / 365)
        
        if years_diff > 50:
            return ValidatorResult(
                name=self.name,
                passed=False,
                message="Date seems unreasonable (>50 years from now)",
                details={
                    "parsed_date": parsed_date.isoformat(),
                    "format_used": format_used
                }
            )
        
        return ValidatorResult(
            name=self.name,
            passed=True,
            message="Valid date",
            details={
                "parsed_date": parsed_date.isoformat(),
                "format_used": format_used,
                "normalized": parsed_date.strftime("%Y-%m-%d")
            }
        )
    
    def _parse_date(self, value: str) -> Tuple[Optional[datetime], Optional[str]]:
        """Try to parse date using known formats."""
        for fmt in self.formats:
            try:
                return datetime.strptime(value, fmt), fmt
            except ValueError:
                continue
        
        # Try common variations
        # Handle 2-digit year
        for fmt in self.formats:
            fmt_2digit = fmt.replace("%Y", "%y")
            try:
                return datetime.strptime(value, fmt_2digit), fmt_2digit
            except ValueError:
                continue
        
        return None, None
    
    def normalize_date(self, value: str) -> Optional[str]:
        """Normalize date to ISO format (YYYY-MM-DD)."""
        parsed, _ = self._parse_date(value)
        if parsed:
            return parsed.strftime("%Y-%m-%d")
        return None


class CurrencyValidator(Validator):
    """Validates currency codes."""
    
    name = "currency"
    
    # ISO 4217 currency codes
    VALID_CURRENCIES = {
        "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY",
        "INR", "MXN", "BRL", "KRW", "SGD", "HKD", "NOK", "SEK",
        "DKK", "NZD", "ZAR", "RUB", "TRY", "PLN", "THB", "MYR",
        "IDR", "PHP", "CZK", "ILS", "CLP", "PKR", "AED", "SAR",
    }
    
    def __init__(self, allowed_codes: Optional[List[str]] = None):
        """
        Initialize currency validator.
        
        Args:
            allowed_codes: List of allowed currency codes (None = all valid ISO codes)
        """
        self.allowed_codes = set(allowed_codes) if allowed_codes else self.VALID_CURRENCIES
    
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidatorResult:
        """Validate currency code."""
        if not value:
            return ValidatorResult(
                name=self.name,
                passed=False,
                message="Empty currency value"
            )
        
        code = str(value).upper().strip()
        
        # Handle currency symbols
        symbol_map = {
            "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY",
            "₹": "INR", "₽": "RUB", "₩": "KRW",
        }
        
        if code in symbol_map:
            code = symbol_map[code]
        
        if code in self.allowed_codes:
            return ValidatorResult(
                name=self.name,
                passed=True,
                message="Valid currency code",
                details={"code": code}
            )
        
        return ValidatorResult(
            name=self.name,
            passed=False,
            message=f"Invalid or disallowed currency code: {code}",
            details={"code": code, "allowed": list(self.allowed_codes)}
        )


class RegexValidator(Validator):
    """Validates values against regex patterns."""
    
    name = "regex"
    
    # Common patterns — kept intentionally broad to avoid false negatives
    # on real-world documents (e.g. INV-2024/001, BOL-ABC-12345, #38291-A)
    PATTERNS = {
        "invoice_number": r'^[A-Za-z0-9#][\w\-/.# ]{1,40}$',
        "po_number": r'^[A-Za-z0-9#][\w\-/.# ]{1,30}$',
        "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "phone": r'^[\+]?[\d\s\-\(\)]{7,20}$',
        "zip_code": r'^\d{5}(-\d{4})?$',
        "tax_id": r'^\d{2}-?\d{7}$',
    }
    
    def __init__(self, pattern: Optional[str] = None, pattern_name: Optional[str] = None):
        """
        Initialize regex validator.
        
        Args:
            pattern: Regex pattern string
            pattern_name: Name of predefined pattern to use
        """
        if pattern:
            self.pattern = pattern
        elif pattern_name and pattern_name in self.PATTERNS:
            self.pattern = self.PATTERNS[pattern_name]
        else:
            self.pattern = r'.*'  # Match anything
        
        self.compiled = re.compile(self.pattern, re.IGNORECASE)
    
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidatorResult:
        """Validate value against regex pattern."""
        if value is None:
            return ValidatorResult(
                name=self.name,
                passed=False,
                message="Empty value"
            )
        
        str_value = str(value).strip()
        
        if self.compiled.match(str_value):
            return ValidatorResult(
                name=self.name,
                passed=True,
                message="Matches pattern",
                details={"value": str_value, "pattern": self.pattern}
            )
        
        return ValidatorResult(
            name=self.name,
            passed=False,
            message="Does not match expected pattern",
            details={"value": str_value, "pattern": self.pattern}
        )


class NonEmptyValidator(Validator):
    """Validates that a value is not empty."""
    
    name = "non_empty"
    
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidatorResult:
        """Validate that value is not empty."""
        if value is None:
            return ValidatorResult(
                name=self.name,
                passed=False,
                message="Value is None"
            )
        
        str_value = str(value).strip()
        
        if not str_value:
            return ValidatorResult(
                name=self.name,
                passed=False,
                message="Value is empty"
            )
        
        return ValidatorResult(
            name=self.name,
            passed=True,
            message="Value is not empty",
            details={"value": str_value}
        )


def run_all_validators(
    field: Field,
    validators: Optional[List[Validator]] = None,
    context: Optional[Dict] = None
) -> List[ValidatorResult]:
    """
    Run all applicable validators on a field.
    
    Args:
        field: Field to validate
        validators: List of validators (auto-detect if None)
        context: Optional context for validation
        
    Returns:
        List of ValidatorResults
    """
    results = []
    
    # Auto-select validators based on data type
    if validators is None:
        validators = []
        
        validators.append(NonEmptyValidator())
        
        if field.data_type == "date":
            validators.append(DateValidator())
        elif field.data_type == "currency":
            validators.append(AmountValidator())
        elif field.data_type == "string":
            # Check if it looks like a known pattern — be specific to
            # avoid false-matching fields like "phone_number", "reference_number"
            name_lower = field.name.lower()
            if "invoice" in name_lower and "number" in name_lower:
                validators.append(RegexValidator(pattern_name="invoice_number"))
            elif name_lower.startswith("po") and "number" in name_lower:
                validators.append(RegexValidator(pattern_name="po_number"))
            elif "email" in name_lower:
                validators.append(RegexValidator(pattern_name="email"))
            elif "phone" in name_lower:
                validators.append(RegexValidator(pattern_name="phone"))
    
    # Run validators
    for validator in validators:
        try:
            result = validator.validate(field.value, context)
            results.append(result)
        except Exception as e:
            logger.warning(f"Validator {validator.name} failed: {e}")
            results.append(ValidatorResult(
                name=validator.name,
                passed=False,
                message=f"Validator error: {str(e)}"
            ))
    
    return results


def validate_document_consistency(fields: List[Field]) -> List[ValidatorResult]:
    """
    Validate consistency across document fields.
    
    Checks cross-field relationships like:
    - Total = sum(items) + tax
    - Dates are in logical order
    - References match
    
    Args:
        fields: All fields in document
        
    Returns:
        List of consistency validation results
    """
    results = []
    
    # Create field lookup
    field_map = {f.name.lower(): f for f in fields}
    
    # Check total calculation
    total_field = field_map.get("total") or field_map.get("total_amount")
    subtotal_field = field_map.get("subtotal")
    tax_field = field_map.get("tax") or field_map.get("tax_amount")
    
    if total_field and subtotal_field:
        validator = AmountValidator()
        
        total = validator._parse_amount(total_field.value) or 0
        subtotal = validator._parse_amount(subtotal_field.value) or 0
        tax = validator._parse_amount(tax_field.value) if tax_field else 0
        
        result = validator.validate_total(total, [subtotal], tax)
        results.append(result)
    
    # Check date ordering
    invoice_date = field_map.get("invoice_date") or field_map.get("date")
    due_date = field_map.get("due_date")
    
    if invoice_date and due_date:
        date_validator = DateValidator()
        
        inv_parsed, _ = date_validator._parse_date(str(invoice_date.value))
        due_parsed, _ = date_validator._parse_date(str(due_date.value))
        
        if inv_parsed and due_parsed:
            if due_parsed < inv_parsed:
                results.append(ValidatorResult(
                    name="date_order",
                    passed=False,
                    message="Due date is before invoice date",
                    details={
                        "invoice_date": str(invoice_date.value),
                        "due_date": str(due_date.value)
                    }
                ))
            else:
                results.append(ValidatorResult(
                    name="date_order",
                    passed=True,
                    message="Date order is valid"
                ))
    
    return results
