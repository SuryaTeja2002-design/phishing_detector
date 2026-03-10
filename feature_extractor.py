"""
feature_extractor.py
--------------------
Extracts 35 numerical features from a URL for ML classification.
Covers lexical, domain, path, and entropy-based signals.
"""

import re
import math
import urllib.parse
from typing import Dict, List


SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "mcaf.ee", "rebrand.ly", "cutt.ly"
}

SUSPICIOUS_KEYWORDS = [
    "secure", "account", "update", "banking", "login", "signin", "verify",
    "confirm", "paypal", "amazon", "apple", "microsoft", "ebay", "password",
    "credential", "support", "alert", "suspended", "urgent", "validate",
    "billing", "payment", "recover", "unlock", "limited", "unusual"
]

TRUSTED_TLDS = {".com", ".org", ".edu", ".gov", ".net", ".io", ".co"}
SUSPICIOUS_TLDS = {".xyz", ".tk", ".ml", ".ga", ".cf", ".pw", ".ru", ".online",
                   ".top", ".gq", ".link", ".click", ".download", ".win"}


def extract_features(url: str) -> Dict[str, float]:
    features = {}
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower().split(":")[0]
    path = parsed.path.lower()
    query = parsed.query
    full_lower = url.lower()

    # --- URL-level ---
    features["url_length"] = len(url)
    features["has_ip_address"] = _has_ip_address(url)
    features["has_at_symbol"] = 1 if "@" in url else 0
    features["has_double_slash"] = 1 if "//" in url[8:] else 0
    features["has_encoded_chars"] = 1 if "%" in url else 0
    features["special_char_count"] = sum(1 for c in url if c in "!#$&*+=[]{}|;'\"<>?\\`~")
    features["digit_ratio_url"] = sum(c.isdigit() for c in url) / max(len(url), 1)

    # --- Domain-level ---
    features["domain_length"] = len(domain)
    features["subdomain_count"] = _count_subdomains(domain)
    features["has_hyphen_domain"] = 1 if "-" in domain else 0
    features["hyphen_count"] = domain.count("-")
    features["dot_count"] = domain.count(".")
    features["digit_count_domain"] = sum(c.isdigit() for c in domain)
    features["is_url_shortener"] = 1 if any(s in domain for s in SHORTENING_SERVICES) else 0
    features["suspicious_tld"] = _suspicious_tld(domain)
    features["trusted_tld"] = _trusted_tld(domain)
    features["domain_entropy"] = _entropy(domain)
    features["domain_vowel_ratio"] = _vowel_ratio(domain)
    features["domain_consonant_ratio"] = _consonant_ratio(domain)
    features["repeated_chars"] = _max_repeated_char(domain)

    # --- Path-level ---
    features["path_length"] = len(path)
    features["path_depth"] = path.count("/")
    features["digit_count_path"] = sum(c.isdigit() for c in path)
    features["path_entropy"] = _entropy(path)

    # --- Query-level ---
    features["query_length"] = len(query)
    features["query_param_count"] = len(urllib.parse.parse_qs(query))

    # --- Content signals ---
    features["suspicious_keyword_count"] = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full_lower)
    features["brand_in_subdomain"] = _brand_in_subdomain(domain)
    features["has_https"] = 1 if url.startswith("https://") else 0
    features["consecutive_digits"] = _max_consecutive_digits(url)
    features["url_entropy"] = _entropy(url)
    features["token_count"] = len(re.split(r"[.\-_/=?&]", url))
    features["long_token"] = max((len(t) for t in re.split(r"[.\-_/=?&]", url)), default=0)
    features["tld_in_path"] = _tld_in_path(path)

    return features


def get_feature_names() -> List[str]:
    sample = extract_features("http://example.com/path")
    return list(sample.keys())


def features_to_vector(url: str) -> list:
    return list(extract_features(url).values())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_ip_address(url: str) -> int:
    return 1 if re.match(r"https?://(\d{1,3}\.){3}\d{1,3}", url) else 0

def _count_subdomains(domain: str) -> int:
    parts = domain.split(".")
    return max(0, len(parts) - 2)

def _suspicious_tld(domain: str) -> int:
    return 1 if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS) else 0

def _trusted_tld(domain: str) -> int:
    return 1 if any(domain.endswith(tld) for tld in TRUSTED_TLDS) else 0

def _entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())

def _vowel_ratio(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in "aeiou") / len(letters)

def _consonant_ratio(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c not in "aeiou") / len(letters)

def _max_repeated_char(text: str) -> int:
    if not text:
        return 0
    max_r = cur = 1
    for i in range(1, len(text)):
        cur = cur + 1 if text[i] == text[i-1] else 1
        max_r = max(max_r, cur)
    return max_r

def _max_consecutive_digits(url: str) -> int:
    max_r = cur = 0
    for ch in url:
        if ch.isdigit():
            cur += 1
            max_r = max(max_r, cur)
        else:
            cur = 0
    return max_r

BRANDS = ["paypal", "amazon", "apple", "microsoft", "google", "facebook",
          "netflix", "ebay", "instagram", "twitter", "linkedin", "chase",
          "wellsfargo", "bankofamerica", "dropbox", "icloud"]

def _brand_in_subdomain(domain: str) -> int:
    parts = domain.split(".")
    subdomains = parts[:-2] if len(parts) > 2 else []
    sub_str = ".".join(subdomains)
    return 1 if any(b in sub_str for b in BRANDS) else 0

def _tld_in_path(path: str) -> int:
    tlds = [".com", ".net", ".org", ".xyz", ".tk", ".ru"]
    return 1 if any(t in path for t in tlds) else 0