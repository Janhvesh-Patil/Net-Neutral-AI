"""
ip_utils.py — Net-Neutral AI

Utility functions for IP address and hostname discovery on local networks.
Used by clients to report their IP address during registration.
"""

import socket
from typing import Optional


def get_local_ip() -> str:
    """
    Get the client's local IP address on the LAN.

    Strategy:
    1. Try to connect to public DNS (8.8.8.8:80) to find local interface
    2. Fallback to socket.gethostbyname(socket.gethostname())

    Returns:
        str: IP address (e.g., "192.168.1.100")
    """
    try:
        # Create a UDP socket (doesn't actually send data)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to Google's public DNS (port 80 is arbitrary, connection doesn't complete)
        s.connect(("8.8.8.8", 80))
        # Get the local IP that would be used for this connection
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback: use hostname resolution
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def get_local_hostname() -> str:
    """
    Get the machine's hostname.

    Returns:
        str: Hostname (e.g., "DESKTOP-ABC123")
    """
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


# Test suite (run directly)
if __name__ == "__main__":
    print("=" * 60)
    print("  ip_utils.py — IP Discovery Test")
    print("=" * 60)

    # Test 1: Get local IP
    print("\n[Test 1] Get local IP address")
    ip = get_local_ip()
    print(f"  [OK] Local IP: {ip}")
    assert ip, "Failed to get IP"
    assert isinstance(ip, str), "IP should be string"
    assert len(ip.split(".")) == 4, "Invalid IP format"
    print(f"  [OK] IP format valid")

    # Test 2: Get hostname
    print("\n[Test 2] Get hostname")
    hostname = get_local_hostname()
    print(f"  [OK] Hostname: {hostname}")
    assert hostname, "Failed to get hostname"
    assert isinstance(hostname, str), "Hostname should be string"
    print(f"  [OK] Hostname format valid")

    # Test 3: Consistency (multiple calls return same value)
    print("\n[Test 3] Consistency check")
    ip2 = get_local_ip()
    hostname2 = get_local_hostname()
    assert ip == ip2, "IP changed between calls"
    assert hostname == hostname2, "Hostname changed between calls"
    print(f"  [OK] Values consistent across multiple calls")

    print("\n" + "=" * 60)
    print("  All tests passed! ip_utils.py is ready")
    print("=" * 60 + "\n")
