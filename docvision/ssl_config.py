"""
SSL certificate configuration utility.

Automatically configures SSL certificates for environments where
the default certificate bundle is not available (corporate proxies, etc.).
Uses certifi's CA bundle as a fallback.
"""

import os
import sys
from pathlib import Path
from loguru import logger


def configure_ssl_certificates() -> bool:
    """
    Configure SSL certificates for the current process.
    
    Checks if SSL_CERT_FILE and REQUESTS_CA_BUNDLE are set.
    If not, attempts to use certifi's CA bundle.
    
    Returns:
        True if certificates were configured, False if already set or unavailable.
    """
    # Check if already configured
    if os.environ.get("SSL_CERT_FILE") and os.environ.get("REQUESTS_CA_BUNDLE"):
        return False
    
    try:
        import certifi
        ca_bundle = certifi.where()
        
        if Path(ca_bundle).exists():
            if not os.environ.get("SSL_CERT_FILE"):
                os.environ["SSL_CERT_FILE"] = ca_bundle
            if not os.environ.get("REQUESTS_CA_BUNDLE"):
                os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
            
            logger.debug(f"SSL certificates configured: {ca_bundle}")
            return True
    except ImportError:
        pass
    
    # Try to find system certificates
    if sys.platform == "win32":
        # On Windows, try common locations
        common_paths = [
            Path(sys.prefix) / "Lib" / "site-packages" / "certifi" / "cacert.pem",
        ]
        for cert_path in common_paths:
            if cert_path.exists():
                os.environ.setdefault("SSL_CERT_FILE", str(cert_path))
                os.environ.setdefault("REQUESTS_CA_BUNDLE", str(cert_path))
                logger.debug(f"SSL certificates configured from: {cert_path}")
                return True
    
    return False


# Auto-configure on import
_ssl_configured = configure_ssl_certificates()
