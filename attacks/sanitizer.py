import re
import unicodedata

# List of zero-width and invisible unicode characters to strip
ZERO_WIDTH_CHARS = [
    '\u200B',  # Zero Width Space
    '\u200C',  # Zero Width Non-Joiner
    '\u200D',  # Zero Width Joiner
    '\uFEFF',  # Zero Width No-Break Space / BOM
    '\u200E',  # Left-to-Right Mark
    '\u200F',  # Right-to-Left Mark
    '\u202A',  # Left-to-Right Embedding
    '\u202B',  # Right-to-Left Embedding
    '\u202C',  # Pop Directional Formatting
    '\u202D',  # Left-to-Right Override
    '\u202E',  # Right-to-Left Override
]

_INVISIBLE_REGEX = re.compile('|'.join(re.escape(c) for c in ZERO_WIDTH_CHARS))


def analyze_unicode_anomaly(text: str) -> dict:
    """Scan text for hidden characters without modifying it."""
    found = []
    for char in ZERO_WIDTH_CHARS:
        count = text.count(char)
        if count > 0:
            found.append({"char": repr(char), "count": count})

    total_invisible = sum(item["count"] for item in found)
    anomaly_score = min(total_invisible / max(len(text), 1) * 100, 1.0)

    return {
        "found_chars": found,
        "total_invisible": total_invisible,
        "anomaly_score": round(anomaly_score, 4),
        "is_suspicious": anomaly_score > 0.01,
    }


def sanitize_text(text: str) -> dict:
    """
    Step 1 - SANITIZE
    Strips hidden zero-width and invisible unicode characters injected at the character level.
    Returns sanitized text and stats on removed characters.
    """
    if not text:
        return {
            "sanitized_text": "",
            "removed_count": 0,
            "anomaly_score": 0.0,
        }

    analysis = analyze_unicode_anomaly(text)
    sanitized = _INVISIBLE_REGEX.sub('', text)
    sanitized = unicodedata.normalize("NFC", sanitized)
    sanitized = re.sub(r' {2,}', ' ', sanitized).strip()

    return {
        "sanitized_text": sanitized,
        "removed_count": analysis["total_invisible"],
        "anomaly_score": analysis["anomaly_score"],
    }
