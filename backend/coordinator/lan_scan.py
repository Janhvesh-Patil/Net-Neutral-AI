"""
lan_scan.py — Discover active Net-Neutral coordinators on a local subnet.

Probes each host on the subnet for GET /api/session_info and returns
sessions that respond with valid metadata.
"""

from __future__ import annotations

import concurrent.futures
from typing import List, Optional

import requests


def _probe_host(ip: str, port: int, timeout: float) -> Optional[dict]:
    url = f"http://{ip}:{port}/api/session_info"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        data = response.json()
        if not data.get("session_id"):
            return None
        return data
    except requests.RequestException:
        return None


def scan_subnet(
    subnet_prefix: str,
    port: int = 5000,
    timeout: float = 0.35,
    max_workers: int = 32,
) -> List[dict]:
    """
    Scan subnet_prefix.* for coordinators.

    Args:
        subnet_prefix: e.g. "192.168.1" (without trailing dot)
        port: coordinator HTTP port
        timeout: per-host request timeout in seconds

    Returns:
        List of session_info dicts from responding coordinators
    """
    prefix = subnet_prefix.strip().rstrip(".")
    if not prefix:
        return []

    hosts = [f"{prefix}.{host}" for host in range(1, 255)]
    sessions: List[dict] = []
    seen_ids = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_probe_host, ip, port, timeout): ip
            for ip in hosts
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result and result["session_id"] not in seen_ids:
                seen_ids.add(result["session_id"])
                sessions.append(result)

    sessions.sort(key=lambda s: (s.get("session_name") or "", s.get("base_url") or ""))
    return sessions


def scan_hosts(hosts: List[str], timeout: float = 1.0) -> List[dict]:
    """Probe explicit host:port strings (e.g. ['192.168.1.10:5000'])."""
    sessions: List[dict] = []
    seen_ids = set()

    for host in hosts:
        host = host.strip()
        if not host:
            continue
        if "://" in host:
            host = host.split("://", 1)[1]
        if ":" in host:
            ip, port_str = host.rsplit(":", 1)
            port = int(port_str)
        else:
            ip, port = host, 5000

        result = _probe_host(ip, port, timeout)
        if result and result["session_id"] not in seen_ids:
            seen_ids.add(result["session_id"])
            sessions.append(result)

    return sessions
