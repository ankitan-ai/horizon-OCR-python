"""
DNS-over-HTTPS (DoH) resolution for Azure private-link endpoints.

Problem:
    Corporate VPNs often return CNAME records pointing to
    *.privatelink.cognitiveservices.azure.com (or *.privatelink.openai.azure.com),
    which have no public A record.  This causes DNS resolution to fail when
    running outside the corporate network (e.g. at home on VPN).

Solution:
    Use Google's DNS-over-HTTPS API (by IP — 8.8.8.8) to resolve the
    public IP, then monkey-patch Python's socket.getaddrinfo so all
    Azure SDK / httpx / requests calls transparently use the resolved IP.

This module is safe to import at any time; it only patches DNS for
hostnames that fail normal resolution.
"""

from __future__ import annotations

import json
import socket
import ssl
import urllib.request
from typing import Dict, Optional
from urllib.parse import urlparse

from loguru import logger

# Cache of DoH-resolved hostnames: hostname → IP
_doh_cache: Dict[str, str] = {}

# Keep a reference to the original getaddrinfo
_original_getaddrinfo = socket.getaddrinfo


def _resolve_via_doh(hostname: str) -> Optional[str]:
    """Resolve *hostname* to an IPv4 address via Google DNS-over-HTTPS."""
    try:
        ctx = ssl.create_default_context()
        url = f"https://8.8.8.8/resolve?name={hostname}&type=A"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/dns-json", "Host": "dns.google"},
        )
        resp = urllib.request.urlopen(req, timeout=10, context=ctx)
        data = json.loads(resp.read())
        for answer in data.get("Answer", []):
            if answer.get("type") == 1:  # A record
                return answer["data"]
    except Exception as exc:
        logger.debug(f"DoH resolution failed for {hostname}: {exc}")
    return None


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """Drop-in replacement for socket.getaddrinfo with DoH fallback."""
    # Try the original resolver first
    try:
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        pass  # fall through to DoH

    # Check DoH cache
    if host in _doh_cache:
        ip = _doh_cache[host]
        return _original_getaddrinfo(ip, port, family, type, proto, flags)

    # Resolve via DoH
    ip = _resolve_via_doh(host)
    if ip:
        logger.info(f"DoH resolved {host} → {ip}")
        _doh_cache[host] = ip
        return _original_getaddrinfo(ip, port, family, type, proto, flags)

    # Nothing worked — raise the original error
    raise socket.gaierror(f"Failed to resolve '{host}' via both system DNS and DoH")


def configure_doh_for_azure(
    *,
    di_endpoint: str = "",
    openai_endpoint: str = "",
) -> int:
    """
    Pre-resolve Azure hostnames via DoH and install the patched getaddrinfo.

    Args:
        di_endpoint:     Azure Document Intelligence endpoint URL.
        openai_endpoint: Azure OpenAI endpoint URL.

    Returns:
        Number of hostnames successfully pre-resolved.
    """
    hostnames = set()
    for endpoint in (di_endpoint, openai_endpoint):
        if endpoint:
            parsed = urlparse(endpoint)
            if parsed.hostname:
                hostnames.add(parsed.hostname)

    if not hostnames:
        return 0

    resolved = 0
    for hostname in hostnames:
        # Check if normal DNS works first
        try:
            _original_getaddrinfo(hostname, 443)
            logger.debug(f"DNS OK for {hostname} — no DoH needed")
            continue
        except socket.gaierror:
            pass

        # Resolve via DoH
        ip = _resolve_via_doh(hostname)
        if ip:
            _doh_cache[hostname] = ip
            logger.info(f"DoH pre-resolved {hostname} → {ip}")
            resolved += 1
        else:
            logger.warning(f"Could not resolve {hostname} via DoH")

    # Install the patched getaddrinfo if we resolved anything
    if _doh_cache and socket.getaddrinfo is not _patched_getaddrinfo:
        socket.getaddrinfo = _patched_getaddrinfo
        logger.info(
            f"Installed DoH DNS patch for {len(_doh_cache)} hostname(s)"
        )

    return resolved
