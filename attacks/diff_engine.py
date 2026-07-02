"""
Word-Level Diff Engine
Generates word-by-word visual diffs comparing original input text and clean output text.
Returns HTML string with highlighted insertions (<span class="diff-add">) and deletions (<span class="diff-del">).
"""

import difflib
import html


def generate_word_diff_html(original_text: str, clean_text: str) -> str:
    """
    Computes a word-level diff and returns sanitized HTML with diff highlighting.
    """
    if not original_text and not clean_text:
        return ""
    if not original_text:
        return f'<span class="diff-add">{html.escape(clean_text)}</span>'
    if not clean_text:
        return f'<span class="diff-del">{html.escape(original_text)}</span>'

    orig_words = original_text.split()
    clean_words = clean_text.split()

    matcher = difflib.SequenceMatcher(None, orig_words, clean_words)
    diff_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            diff_parts.append(html.escape(" ".join(orig_words[i1:i2])))
        elif tag == 'replace':
            del_text = html.escape(" ".join(orig_words[i1:i2]))
            add_text = html.escape(" ".join(clean_words[j1:j2]))
            diff_parts.append(f'<span class="diff-del">{del_text}</span> <span class="diff-add">{add_text}</span>')
        elif tag == 'delete':
            del_text = html.escape(" ".join(orig_words[i1:i2]))
            diff_parts.append(f'<span class="diff-del">{del_text}</span>')
        elif tag == 'insert':
            add_text = html.escape(" ".join(clean_words[j1:j2]))
            diff_parts.append(f'<span class="diff-add">{add_text}</span>')

    return " ".join(diff_parts)
