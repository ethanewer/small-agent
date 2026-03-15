# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportUnannotatedClassAttribute=false, reportPossiblyUnboundVariable=false, reportUnusedVariable=false

from __future__ import annotations

from pathlib import Path
import tempfile
from urllib.parse import urlparse
from typing import Any

FETCH_TRUNCATION_LIMIT = 40000


def _is_disallowed_by_robots(*, url: str, client: Any) -> tuple[bool, str | None]:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False, None

    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        robots_response = client.get(robots_url)
    except Exception:
        return False, None

    if robots_response.status_code != 200:
        return False, None

    raw_path = parsed.path or "/"
    path = raw_path if raw_path.startswith("/") else f"/{raw_path}"

    for raw_line in robots_response.text.splitlines():
        line = raw_line.strip()
        if not line.startswith("Disallow: "):
            continue
        disallowed = line[len("Disallow: ") :].strip()
        if not disallowed:
            continue
        disallowed = disallowed if disallowed.startswith("/") else f"/{disallowed}"
        if path.startswith(disallowed):
            return True, f"URL {url} cannot be fetched due to robots.txt restrictions"

    return False, None


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
            blocked, robots_error = _is_disallowed_by_robots(url=url, client=client)
            if blocked and robots_error:
                return f"Error: {robots_error}"
            response = client.get(url)
    except httpx.TimeoutException:
        return f"Error: Request timed out fetching {url}"
    except Exception as e:
        return f"Error: Failed to fetch URL {url}: {e}"

    if response.status_code != 200:
        return f"Error: Failed to fetch {url} - status code {response.status_code}"

    content_type = response.headers.get("content-type", "")
    page_raw = response.text

    html_probe = page_raw[: min(100, len(page_raw))]
    is_html = "<html" in html_probe or "text/html" in content_type or not content_type

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
        full_content = content
        content = content[:FETCH_TRUNCATION_LIMIT]
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                prefix="forge_fetch_",
                suffix=".txt",
            ) as tmp:
                tmp.write(full_content)
                temp_path = Path(tmp.name)
            content += (
                "\n\n"
                f"[Content truncated at {FETCH_TRUNCATION_LIMIT} characters; "
                f"remaining content can be read from path: {temp_path}]"
            )
        except Exception:
            content += f"\n\n[Content truncated at {FETCH_TRUNCATION_LIMIT} characters]"

    return content
