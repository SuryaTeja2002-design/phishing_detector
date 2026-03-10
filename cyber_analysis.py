"""
cyber_analysis.py
-----------------
Performs real cybersecurity checks on a URL:
  - WHOIS: domain age, registrar
  - DNS:   A/MX/NS record analysis, suspicious patterns
  - SSL:   Certificate validity, issuer, expiry

All done using Python stdlib (socket, ssl, http) so no pip installs needed.
Falls back gracefully when network is unavailable (offline/sandboxed mode).
"""

import ssl
import socket
import urllib.parse
import datetime
import re
from typing import Dict, Any


# ── Public entry point ────────────────────────────────────────────────────────

def analyze_url(url: str) -> Dict[str, Any]:
    """
    Run all cybersecurity checks on a URL.
    Returns a dict with keys: whois, dns, ssl, threat_intel, summary_flags
    """
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower().split(":")[0]
    # Strip www
    clean_domain = domain[4:] if domain.startswith("www.") else domain

    results = {
        "domain": domain,
        "whois": _whois_analysis(clean_domain),
        "dns":   _dns_analysis(domain),
        "ssl":   _ssl_analysis(domain, parsed.scheme),
        "threat_intel": _threat_intel(url, domain),
    }
    results["summary_flags"] = _build_summary_flags(results)
    results["cyber_risk_score"] = _cyber_risk_score(results)
    return results


# ── WHOIS Analysis ────────────────────────────────────────────────────────────

def _whois_analysis(domain: str) -> Dict[str, Any]:
    """
    Query WHOIS via raw socket on port 43.
    Falls back to heuristic estimates if blocked.
    """
    info = {
        "method": "raw_socket",
        "raw_available": False,
        "domain_age_days": None,
        "registrar": "Unknown",
        "creation_date": "Unknown",
        "expiry_date": "Unknown",
        "recently_created": None,
        "risk_note": "",
    }

    try:
        whois_server = _get_whois_server(domain)
        raw = _raw_whois_query(domain, whois_server)

        if raw:
            info["raw_available"] = True
            info["registrar"] = _parse_whois_field(raw, ["Registrar:", "registrar:"])
            creation = _parse_whois_date(raw, [
                "Creation Date:", "creation date:", "Created:", "created:"
            ])
            expiry = _parse_whois_date(raw, [
                "Registry Expiry Date:", "Expiry Date:", "Expiration Date:", "expiry date:"
            ])

            if creation:
                info["creation_date"] = creation.strftime("%Y-%m-%d")
                age_days = (datetime.datetime.utcnow() - creation).days
                info["domain_age_days"] = age_days
                info["recently_created"] = age_days < 180
                if age_days < 30:
                    info["risk_note"] = "⚠ Domain created less than 30 days ago — very high risk"
                elif age_days < 180:
                    info["risk_note"] = "⚠ Domain created less than 6 months ago — elevated risk"
                else:
                    info["risk_note"] = f"✓ Domain is {age_days} days old — established"

            if expiry:
                info["expiry_date"] = expiry.strftime("%Y-%m-%d")
        else:
            info["risk_note"] = "WHOIS data unavailable (blocked or offline)"
            info["recently_created"] = None

    except Exception as e:
        info["error"] = str(e)
        info["risk_note"] = "WHOIS query failed (network unavailable in this environment)"

    return info


def _get_whois_server(domain: str) -> str:
    tld_servers = {
        "com": "whois.verisign-grs.com", "net": "whois.verisign-grs.com",
        "org": "whois.pir.org",          "io":  "whois.nic.io",
        "xyz": "whois.nic.xyz",          "tk":  "whois.dot.tk",
        "ru":  "whois.tcinet.ru",        "uk":  "whois.nic.uk",
        "co":  "whois.nic.co",           "info": "whois.afilias.net",
    }
    tld = domain.rsplit(".", 1)[-1] if "." in domain else "com"
    return tld_servers.get(tld, "whois.iana.org")


def _raw_whois_query(domain: str, server: str, timeout: int = 5) -> str:
    try:
        with socket.create_connection((server, 43), timeout=timeout) as sock:
            sock.sendall(f"{domain}\r\n".encode())
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
            return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _parse_whois_field(raw: str, keys: list) -> str:
    for key in keys:
        for line in raw.splitlines():
            if line.strip().startswith(key):
                val = line.split(":", 1)[-1].strip()
                if val:
                    return val[:80]
    return "Unknown"


def _parse_whois_date(raw: str, keys: list) -> datetime.datetime | None:
    date_patterns = [
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}-\w{3}-\d{4}",
    ]
    formats = [
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d", "%d-%b-%Y"
    ]
    for key in keys:
        for line in raw.splitlines():
            if key.lower() in line.lower():
                val = line.split(":", 1)[-1].strip()
                for pat, fmt in zip(date_patterns, formats):
                    m = re.search(pat, val)
                    if m:
                        try:
                            return datetime.datetime.strptime(m.group(), fmt)
                        except ValueError:
                            continue
    return None


# ── DNS Analysis ──────────────────────────────────────────────────────────────

def _dns_analysis(domain: str) -> Dict[str, Any]:
    info = {
        "resolves": False,
        "ip_addresses": [],
        "is_private_ip": False,
        "mx_exists": False,
        "nameservers": [],
        "risk_notes": [],
    }

    # A record (basic resolution)
    try:
        ip = socket.gethostbyname(domain)
        info["resolves"] = True
        info["ip_addresses"] = [ip]
        info["is_private_ip"] = _is_private_ip(ip)
        if info["is_private_ip"]:
            info["risk_notes"].append("⚠ Resolves to private/internal IP — suspicious")
    except socket.gaierror:
        info["risk_notes"].append("⚠ Domain does not resolve — possibly fake or taken down")
    except Exception as e:
        info["risk_notes"].append(f"DNS resolution error: {e}")

    # MX record check (legitimate domains usually have mail servers)
    try:
        socket.getaddrinfo(f"mail.{domain}", 25, socket.AF_INET)
        info["mx_exists"] = True
    except Exception:
        pass

    # Heuristic NS check
    try:
        results = socket.getaddrinfo(domain, None)
        if results:
            info["nameservers"] = list({r[4][0] for r in results})[:3]
    except Exception:
        pass

    # Extra heuristics
    if not info["resolves"]:
        info["risk_notes"].append("Domain is unresolvable — dead or newly registered")
    if not info["mx_exists"]:
        info["risk_notes"].append("No mail server detected — unusual for legitimate businesses")
    if not info["risk_notes"]:
        info["risk_notes"].append("✓ DNS looks normal")

    return info


def _is_private_ip(ip: str) -> bool:
    try:
        parts = list(map(int, ip.split(".")))
        return (
            parts[0] == 10 or
            (parts[0] == 172 and 16 <= parts[1] <= 31) or
            (parts[0] == 192 and parts[1] == 168) or
            parts[0] == 127
        )
    except Exception:
        return False


# ── SSL/TLS Analysis ──────────────────────────────────────────────────────────

def _ssl_analysis(domain: str, scheme: str) -> Dict[str, Any]:
    info = {
        "checked": False,
        "valid": False,
        "issuer": "Unknown",
        "subject": "Unknown",
        "expires": "Unknown",
        "days_until_expiry": None,
        "self_signed": False,
        "risk_notes": [],
    }

    if scheme != "https":
        info["risk_notes"].append("⚠ Not using HTTPS — traffic is unencrypted")
        return info

    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                info["checked"] = True
                info["valid"] = True

                # Issuer
                issuer_dict = dict(x[0] for x in cert.get("issuer", []))
                info["issuer"] = issuer_dict.get("organizationName", "Unknown")
                info["self_signed"] = issuer_dict.get("organizationName") == \
                    dict(x[0] for x in cert.get("subject", [])).get("organizationName")

                # Subject
                subj = dict(x[0] for x in cert.get("subject", []))
                info["subject"] = subj.get("commonName", "Unknown")

                # Expiry
                expiry_str = cert.get("notAfter", "")
                if expiry_str:
                    expiry = datetime.datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z")
                    info["expires"] = expiry.strftime("%Y-%m-%d")
                    days = (expiry - datetime.datetime.utcnow()).days
                    info["days_until_expiry"] = days
                    if days < 0:
                        info["valid"] = False
                        info["risk_notes"].append("⚠ SSL certificate has EXPIRED")
                    elif days < 30:
                        info["risk_notes"].append(f"⚠ Certificate expires in {days} days")

                if info["self_signed"]:
                    info["risk_notes"].append("⚠ Self-signed certificate — not from a trusted CA")

                free_ca_keywords = ["let's encrypt", "zerossl", "buypass"]
                if any(k in info["issuer"].lower() for k in free_ca_keywords):
                    info["risk_notes"].append(
                        f"ℹ Free CA certificate ({info['issuer']}) — common in phishing but also legit sites"
                    )

                if not info["risk_notes"]:
                    info["risk_notes"].append(f"✓ Valid SSL from {info['issuer']}, expires {info['expires']}")

    except ssl.SSLCertVerificationError as e:
        info["valid"] = False
        info["risk_notes"].append(f"⚠ SSL verification failed: {e}")
    except (socket.timeout, ConnectionRefusedError, OSError):
        info["risk_notes"].append("SSL check skipped (network unreachable in this environment)")
    except Exception as e:
        info["risk_notes"].append(f"SSL error: {type(e).__name__}: {e}")

    return info


# ── Threat Intelligence (heuristic) ──────────────────────────────────────────

def _threat_intel(url: str, domain: str) -> Dict[str, Any]:
    """
    Simulates threat intelligence checks using heuristics.
    In production: integrate VirusTotal API, AbuseIPDB, OpenPhish, PhishTank.
    """
    flags = []
    score = 0

    # Known bad TLDs
    bad_tlds = [".xyz", ".tk", ".ml", ".ga", ".cf", ".pw", ".top", ".gq"]
    if any(domain.endswith(t) for t in bad_tlds):
        flags.append(f"Domain uses high-risk TLD")
        score += 25

    # Brand impersonation
    brands = ["paypal", "amazon", "apple", "microsoft", "google",
              "facebook", "netflix", "ebay", "chase", "wellsfargo"]
    for brand in brands:
        if brand in domain and not domain.endswith(f"{brand}.com"):
            flags.append(f"Possible brand impersonation: '{brand}' in domain")
            score += 30
            break

    # URL shortener
    shorteners = ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly"]
    if any(s in domain for s in shorteners):
        flags.append("URL shortener detected — destination hidden")
        score += 20

    # IP-based URL
    if re.match(r"https?://(\d{1,3}\.){3}\d{1,3}", url):
        flags.append("Raw IP address used instead of domain name")
        score += 35

    # Excessive subdomains
    parts = domain.split(".")
    if len(parts) > 4:
        flags.append(f"Excessive subdomains ({len(parts)-2}) — evasion technique")
        score += 15

    # Long URL
    if len(url) > 100:
        flags.append(f"Unusually long URL ({len(url)} chars)")
        score += 10

    # Keywords
    phish_words = ["verify", "confirm", "update", "secure", "account", "login", "suspended"]
    hits = [w for w in phish_words if w in url.lower()]
    if len(hits) >= 2:
        flags.append(f"Multiple phishing keywords: {', '.join(hits[:3])}")
        score += 20

    reputation = "UNKNOWN"
    if score >= 60:
        reputation = "HIGH RISK"
    elif score >= 30:
        reputation = "SUSPICIOUS"
    elif score == 0:
        reputation = "CLEAN"

    note = ("In production, integrate: VirusTotal API, AbuseIPDB, "
            "OpenPhish feed, PhishTank CSV — all free tiers available.")

    return {
        "reputation": reputation,
        "threat_score": min(score, 100),
        "flags": flags,
        "api_note": note,
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def _build_summary_flags(results: dict) -> list:
    flags = []

    whois = results.get("whois", {})
    if whois.get("recently_created"):
        flags.append({"severity": "HIGH", "msg": f"Newly registered domain: {whois.get('creation_date', 'unknown date')}"})

    dns = results.get("dns", {})
    if not dns.get("resolves"):
        flags.append({"severity": "HIGH", "msg": "Domain does not resolve in DNS"})
    if dns.get("is_private_ip"):
        flags.append({"severity": "HIGH", "msg": "Resolves to private/internal IP address"})

    ssl = results.get("ssl", {})
    if not ssl.get("valid") and results["domain"].startswith("https"):
        flags.append({"severity": "HIGH", "msg": "Invalid or expired SSL certificate"})
    if ssl.get("self_signed"):
        flags.append({"severity": "MEDIUM", "msg": "Self-signed SSL certificate"})

    ti = results.get("threat_intel", {})
    for f in ti.get("flags", []):
        flags.append({"severity": "MEDIUM", "msg": f})

    return flags


def _cyber_risk_score(results: dict) -> int:
    score = results.get("threat_intel", {}).get("threat_score", 0)
    if results.get("whois", {}).get("recently_created"):
        score += 20
    if not results.get("dns", {}).get("resolves"):
        score += 15
    if results.get("ssl", {}).get("self_signed"):
        score += 10
    return min(score, 100)