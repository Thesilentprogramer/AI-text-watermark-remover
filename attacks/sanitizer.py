import re

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


def sanitize_text(text: str) -> dict:
    """
    Step 1 - SANITIZE
    Strips hidden zero-width and invisible unicode characters injected at the character level.
    Returns sanitized text and stats on removed characters.
    """
    if not text:
        return {"sanitized_text": "", "removed_count": 0}

    initial_len = len(text)
    sanitized = _INVISIBLE_REGEX.sub('', text)
    removed_count = initial_len - len(sanitized)

    return {
        "sanitized_text": sanitized,
        "removed_count": removed_count
    }
