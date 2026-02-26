"""Base64 figure embedding utility for self-contained HTML reports.

Converts PNG and SVG figure files into data URI strings that can be
inlined directly in HTML <img> tags, eliminating external file dependencies.
"""

import base64
from pathlib import Path

# Supported MIME types by file extension
_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def embed_figure(fig_path: Path | str) -> str:
    """Read a figure file and return a base64 data URI string.

    Args:
        fig_path: Path to a PNG, SVG, or JPEG figure file.

    Returns:
        A ``data:{mime};base64,{data}`` URI string suitable for use in
        an HTML ``<img src="...">`` attribute.  Returns an empty string
        if the file does not exist or has an unsupported extension.
    """
    fig_path = Path(fig_path)

    if not fig_path.exists():
        return ""

    suffix = fig_path.suffix.lower()
    mime = _MIME_TYPES.get(suffix)
    if mime is None:
        return ""

    raw = fig_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"
