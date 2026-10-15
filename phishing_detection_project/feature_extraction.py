"""Extract simple structural features from a URL for the demo classifier."""
import ipaddress
import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = (
    "login", "verify", "update", "secure", "account", "bank", "confirm",
    "password", "signin", "wallet", "payment", "free", "bonus", "recover",
)


def extract_url_features(raw_url: str) -> dict:
    raw_url = (raw_url or "").strip()
    candidate = raw_url if "://" in raw_url else "http://" + raw_url
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    full = raw_url.lower()
    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0
    labels = [part for part in host.split(".") if part]
    return {
        "url_length": len(raw_url),
        "hostname_length": len(host),
        "path_length": len(path),
        "count_dots": raw_url.count("."),
        "count_hyphens": raw_url.count("-"),
        "count_at": raw_url.count("@"),
        "count_question": raw_url.count("?"),
        "count_equal": raw_url.count("="),
        "count_percent": raw_url.count("%"),
        "count_digits": sum(ch.isdigit() for ch in raw_url),
        "subdomain_count": max(0, len(labels) - 2),
        "https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": has_ip,
        "has_port": int(parsed.port is not None) if parsed.port else 0,
        "has_double_slash_path": int("//" in path),
        "suspicious_word_count": sum(full.count(word) for word in SUSPICIOUS_WORDS),
        "is_shortener": int(any(host == d or host.endswith("." + d) for d in
            ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly"))),
    }


def features_for_urls(urls):
    import pandas as pd
    return pd.DataFrame([extract_url_features(url) for url in urls])
