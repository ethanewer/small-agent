# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Any

FETCH_TRUNCATION_LIMIT = 40000


def execute(args: dict[str, Any], env: dict[str, Any]) -> str:
    url = args.get("url", "")
    raw = args.get("raw", False)

    if not url:
        return "Error: url is required"

    try:
        import httpx
    except ImportError:
        return "Error: httpx is not installed. Run: pip install httpx"

    try:
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            response = client.get(url)
    except httpx.TimeoutException:
        return f"Error: Request timed out fetching {url}"
    except Exception as e:
        return f"Error: Failed to fetch URL {url}: {e}"

    if response.status_code != 200:
        return f"Error: Failed to fetch {url} - status code {response.status_code}"

    content_type = response.headers.get("content-type", "")
    page_raw = response.text

    is_html = (
        "<html" in page_raw[:200].lower()
        or "text/html" in content_type
        or not content_type
    )

    if is_html and not raw:
        try:
            import html2text

            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.body_width = 0
            content = h.handle(page_raw)
        except ImportError:
            content = page_raw
    else:
        content = page_raw

    if len(content) > FETCH_TRUNCATION_LIMIT:
        content = content[:FETCH_TRUNCATION_LIMIT]
        content += f"\n\n[Content truncated at {FETCH_TRUNCATION_LIMIT} characters]"

    return content
