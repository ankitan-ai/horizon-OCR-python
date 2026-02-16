#!/bin/bash
# Docker entrypoint that resolves Azure hostnames via DNS-over-HTTPS (DoH)
# to bypass corporate VPN DNS interception of private link endpoints.
#
# Problem: Corporate DNS returns CNAME to *.privatelink.cognitiveservices.azure.com
# which has no public A record, causing DNS resolution to fail in Docker.
# Solution: Use Google's DoH API (HTTPS to 8.8.8.8 by IP, not intercepted by VPN)
# to get the public IP and add it to /etc/hosts.

set -e

resolve_via_doh() {
    local hostname="$1"
    # Use Google's DNS-over-HTTPS API by IP (bypasses DNS interception for the DoH server itself)
    local ip
    ip=$(python3 -c "
import json, urllib.request, ssl, os, sys

hostname = sys.argv[1]
ctx = ssl.create_default_context()
url = f'https://8.8.8.8/resolve?name={hostname}&type=A'
req = urllib.request.Request(url, headers={'Accept': 'application/dns-json', 'Host': 'dns.google'})
try:
    resp = urllib.request.urlopen(req, timeout=10, context=ctx)
    data = json.loads(resp.read())
    answers = data.get('Answer', [])
    for ans in answers:
        if ans.get('type') == 1:
            print(ans['data'])
            break
except Exception:
    pass
" "$hostname" 2>/dev/null)
    echo "$ip"
}

# Extract Azure hostnames from environment variables
HOSTS_TO_RESOLVE=""

if [ -n "$AZURE_DOC_INTELLIGENCE_ENDPOINT" ]; then
    host=$(python3 -c "import os; from urllib.parse import urlparse; print(urlparse(os.environ.get('AZURE_DOC_INTELLIGENCE_ENDPOINT','')).hostname or '')" 2>/dev/null)
    [ -n "$host" ] && HOSTS_TO_RESOLVE="$HOSTS_TO_RESOLVE $host"
fi

if [ -n "$AZURE_OPENAI_ENDPOINT" ]; then
    host=$(python3 -c "import os; from urllib.parse import urlparse; print(urlparse(os.environ.get('AZURE_OPENAI_ENDPOINT','')).hostname or '')" 2>/dev/null)
    [ -n "$host" ] && HOSTS_TO_RESOLVE="$HOSTS_TO_RESOLVE $host"
fi

# Resolve each hostname and add to /etc/hosts if needed
for hostname in $HOSTS_TO_RESOLVE; do
    # Check if it already resolves
    if python3 -c "import socket, sys; socket.getaddrinfo(sys.argv[1], 443)" "$hostname" 2>/dev/null; then
        echo "[entrypoint] $hostname already resolves - OK"
    else
        echo "[entrypoint] $hostname failed DNS - resolving via DoH..."
        ip=$(resolve_via_doh "$hostname")
        if [ -n "$ip" ]; then
            echo "$ip $hostname" >> /etc/hosts
            echo "[entrypoint] Added $hostname -> $ip to /etc/hosts"
        else
            echo "[entrypoint] WARNING: Could not resolve $hostname via DoH"
        fi
    fi
done

# Drop privileges and run the main command
exec gosu docvision "$@"
