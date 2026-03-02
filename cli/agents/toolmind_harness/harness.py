#!/usr/bin/env python3
"""ToolMind-Web-QA style trajectory harness.

Generates conversations in a runtime-oriented schema close to `open-wiki-traj.jsonl`:
{
  "key": "...",
  "id": "...",
  "conversations": [{"role": "...", "content": "..."}, ...]
}

Design goal: emulate the protocol observed in the dataset:
- system prompt describing MCP tool-use
- one tool call per assistant turn in XML tags
- tool result returned as the next user turn JSON payload
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import secrets
import subprocess
import tempfile
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPAIR_MSG_MISSING_TOOL = json.dumps(
    {
        "success": False,
        "error": "ProtocolError: No <use_mcp_tool> block found. You must call exactly one tool unless you are providing the final answer.",
        "results": [],
    },
    ensure_ascii=False,
    indent=2,
)

REPAIR_MSG_MULTIPLE_TOOLS = json.dumps(
    {
        "success": False,
        "error": "ProtocolError: Multiple <use_mcp_tool> blocks found. You can only call one tool per message.",
        "results": [],
    },
    ensure_ascii=False,
    indent=2,
)


TOOL_SCHEMAS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "tool-python": {
        "create_sandbox": {
            "description": "Create a local sandbox working directory.",
            "schema": {
                "type": "object",
                "properties": {"timeout": {"default": 600, "title": "Timeout"}},
            },
        },
        "run_command": {
            "description": "Execute a lightweight shell command in sandbox.",
            "schema": {
                "type": "object",
                "required": ["command", "sandbox_id"],
                "properties": {
                    "command": {"type": "string", "title": "Command"},
                    "sandbox_id": {"type": "string", "title": "Sandbox Id"},
                },
            },
        },
        "run_python_code": {
            "description": "Run short python code in sandbox.",
            "schema": {
                "type": "object",
                "required": ["code_block", "sandbox_id"],
                "properties": {
                    "code_block": {"type": "string", "title": "Code Block"},
                    "sandbox_id": {"type": "string", "title": "Sandbox Id"},
                },
            },
        },
        "upload_file_from_local_to_sandbox": {
            "description": "Copy a local file into sandbox directory.",
            "schema": {
                "type": "object",
                "required": ["sandbox_id", "local_file_path"],
                "properties": {
                    "sandbox_id": {"type": "string", "title": "Sandbox Id"},
                    "local_file_path": {"type": "string", "title": "Local File Path"},
                    "sandbox_file_path": {
                        "type": "string",
                        "default": "/home/user",
                        "title": "Sandbox File Path",
                    },
                },
            },
        },
        "download_file_from_internet_to_sandbox": {
            "description": "Download internet file into sandbox.",
            "schema": {
                "type": "object",
                "required": ["sandbox_id", "url"],
                "properties": {
                    "sandbox_id": {"type": "string", "title": "Sandbox Id"},
                    "url": {"type": "string", "title": "Url"},
                    "sandbox_file_path": {
                        "type": "string",
                        "default": "/home/user",
                        "title": "Sandbox File Path",
                    },
                },
            },
        },
        "download_file_from_sandbox_to_local": {
            "description": "Copy file from sandbox to local machine.",
            "schema": {
                "type": "object",
                "required": ["sandbox_id", "sandbox_file_path"],
                "properties": {
                    "sandbox_id": {"type": "string", "title": "Sandbox Id"},
                    "sandbox_file_path": {
                        "type": "string",
                        "title": "Sandbox File Path",
                    },
                    "local_filename": {
                        "type": "string",
                        "default": None,
                        "title": "Local Filename",
                    },
                },
            },
        },
    },
    "search_and_scrape_webpage": {
        "google_search": {
            "description": "Google-like search via Serper API if configured.",
            "schema": {
                "type": "object",
                "required": ["q"],
                "properties": {
                    "q": {"type": "string", "title": "Q"},
                    "gl": {"type": "string", "default": "us", "title": "Gl"},
                    "hl": {"type": "string", "default": "en", "title": "Hl"},
                    "location": {
                        "type": "string",
                        "default": None,
                        "title": "Location",
                    },
                    "num": {"type": "integer", "default": None, "title": "Num"},
                    "tbs": {"type": "string", "default": None, "title": "Tbs"},
                    "page": {"type": "integer", "default": None, "title": "Page"},
                    "autocorrect": {
                        "type": "boolean",
                        "default": None,
                        "title": "Autocorrect",
                    },
                },
            },
        }
    },
    "jina_scrape_llm_summary": {
        "scrape_and_extract_info": {
            "description": "Fetch URL and extract a short answer.",
            "schema": {
                "type": "object",
                "required": ["url", "info_to_extract"],
                "properties": {
                    "url": {"type": "string", "title": "Url"},
                    "info_to_extract": {"type": "string", "title": "Info To Extract"},
                    "custom_headers": {
                        "type": "object",
                        "default": None,
                        "title": "Custom Headers",
                        "additionalProperties": {"type": "string"},
                    },
                },
            },
        }
    },
}


def build_system_prompt(today: str) -> str:
    schemas_blob = []
    for server_name, tools in TOOL_SCHEMAS.items():
        schemas_blob.append(f"## Server name: {server_name}")
        for tool_name, spec in tools.items():
            schemas_blob.append(f"### Tool name: {tool_name}")
            schemas_blob.append(f"Description: {spec['description']}")
            schemas_blob.append(f"Input JSON schema: {spec['schema']}")
    tools_txt = "\n".join(schemas_blob)
    return (
        "In this environment you have access to a set of tools you can use to answer the user's question.\n\n"
        "You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: "
        f"{today}\n\n"
        "# Tool-Use Formatting Instructions\n\n"
        "Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.\n\n"
        "The Model Context Protocol (MCP) connects to servers that provide additional tools and resources to extend your capabilities. You can use the server's tools via the `use_mcp_tool`.\n\n"
        "Description:\n"
        "Request to use a tool provided by a MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.\n\n"
        "Parameters:\n"
        "- server_name: (required) The name of the MCP server providing the tool\n"
        "- tool_name: (required) The name of the tool to execute\n"
        "- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema, quotes within string must be properly escaped, ensure it's valid JSON\n\n"
        "Usage:\n"
        "<use_mcp_tool>\n"
        "<server_name>server name here</server_name>\n"
        "<tool_name>tool name here</tool_name>\n"
        "<arguments>\n"
        "{\n"
        '"param1": "value1",\n'
        '"param2": "value2 \\"escaped string\\""\n'
        "}\n"
        "</arguments>\n"
        "</use_mcp_tool>\n\n"
        "Important Notes:\n"
        "- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.\n"
        "- Always adhere to this format for the tool use to ensure proper parsing and execution.\n\n"
        "String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\n"
        "Here are the functions available in JSONSchema format:\n\n"
        f"{tools_txt}\n\n"
        "# General Objective\n\n"
        "You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.\n\n"
        "# Agent Specific Objective\n\n"
        "You are a task-solving agent that uses tools step-by-step to answer the user's question. Your goal is to provide complete, accurate and well-reasoned answers using additional tools.\n"
    )


@dataclass
class ToolCall:
    server_name: str
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class CompletionResult:
    content: str
    reasoning: str


@dataclass
class HarnessCallbacks:
    on_reasoning: Callable[[int, str, str], None] | None = None
    on_tool_call: Callable[[int, ToolCall], None] | None = None
    on_tool_result: Callable[[int, ToolCall, Dict[str, Any]], None] | None = None
    on_issue: Callable[[str, str], None] | None = None
    on_done: Callable[[str], None] | None = None
    on_stopped: Callable[[int], None] | None = None


class MCPParser:
    BLOCK_RE = re.compile(r"<use_mcp_tool>(.*?)</use_mcp_tool>", re.S)

    @staticmethod
    def parse(text: str) -> Optional[ToolCall]:
        calls = MCPParser.parse_all(text)
        if not calls:
            return None
        return calls[0]

    @staticmethod
    def parse_all(text: str) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for m in MCPParser.BLOCK_RE.finditer(text):
            block = m.group(1)
            server = MCPParser._tag(block, "server_name")
            tool = MCPParser._tag(block, "tool_name")
            args_raw = MCPParser._tag(block, "arguments")
            if not server or not tool or args_raw is None:
                continue
            try:
                args = json.loads(args_raw.strip())
            except json.JSONDecodeError:
                args = {"_raw_arguments": args_raw}
            calls.append(
                ToolCall(
                    server_name=server.strip(), tool_name=tool.strip(), arguments=args
                )
            )
        return calls

    @staticmethod
    def _tag(s: str, name: str) -> Optional[str]:
        m = re.search(rf"<{name}>(.*?)</{name}>", s, re.S)
        return m.group(1) if m else None


class OpenAIChatClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        include_reasoning: bool = True,
    ) -> CompletionResult:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        if include_reasoning:
            payload["include_reasoning"] = True
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat completion failed: HTTP {e.code} {detail}") from e
        except Exception as e:
            raise RuntimeError(f"Chat completion failed: {e}") from e
        try:
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = self._extract_reasoning(msg)
            return CompletionResult(content=content, reasoning=reasoning)
        except Exception as e:
            raise RuntimeError(f"Unexpected completion payload: {data}") from e

    @staticmethod
    def _extract_reasoning(msg: Dict[str, Any]) -> str:
        if isinstance(msg.get("reasoning"), str):
            return msg["reasoning"]
        if isinstance(msg.get("reasoning_content"), str):
            return msg["reasoning_content"]
        details = msg.get("reasoning_details")
        if isinstance(details, list):
            chunks: List[str] = []
            for item in details:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            if chunks:
                return "\n".join(chunks)
        return ""


class ToolExecutor:
    def __init__(
        self,
        scratch_dir: Path,
        allow_fallback_search: bool = False,
        block_huggingface: bool = True,
        extractor_model: Optional[str] = None,
    ):
        self.scratch_dir = scratch_dir
        self.sandboxes: Dict[str, Path] = {}
        self.allow_fallback_search = allow_fallback_search
        self.block_huggingface = block_huggingface
        self.extractor_client: Optional[OpenAIChatClient] = None
        if extractor_model and os.getenv("OPENAI_API_KEY"):
            self.extractor_client = OpenAIChatClient(model=extractor_model)

    def _blocked(self, s: str) -> bool:
        if not self.block_huggingface:
            return False
        low = s.lower()
        return "huggingface.co" in low or "hf.co" in low

    def execute(self, call: ToolCall) -> Dict[str, Any]:
        try:
            if (
                call.server_name == "search_and_scrape_webpage"
                and call.tool_name == "google_search"
            ):
                return self._google_search(call.arguments)
            if (
                call.server_name == "jina_scrape_llm_summary"
                and call.tool_name == "scrape_and_extract_info"
            ):
                return self._scrape_and_extract(call.arguments)
            if call.server_name == "tool-python":
                return self._tool_python(call.tool_name, call.arguments)
            return {
                "success": False,
                "error": f"Unknown tool: {call.server_name}.{call.tool_name}",
                "results": [],
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {e}",
                "traceback": traceback.format_exc(limit=1),
            }

    def _google_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        q = args.get("q", "")
        if not q:
            return {
                "success": False,
                "error": "Missing required argument 'q'.",
                "results": [],
            }
        if self._blocked(q):
            return {
                "success": False,
                "error": "Access to HuggingFace is disabled in this tool.",
                "results": [],
            }
        key = os.getenv("SERPER_API_KEY", "")
        if key:
            base = os.getenv("SERPER_BASE_URL", "https://google.serper.dev/search")
            payload = {
                "q": q,
                "gl": args.get("gl", "us"),
                "hl": args.get("hl", "en"),
                "location": args.get("location"),
                "num": args.get("num"),
                "page": args.get("page"),
                "tbs": args.get("tbs"),
                "autocorrect": args.get("autocorrect"),
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            req = urllib.request.Request(
                base,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            return {
                "success": True,
                "error": "",
                "results": data.get("organic", []),
                "peopleAlsoAsk": data.get("peopleAlsoAsk", []),
                "relatedSearches": data.get("relatedSearches", []),
                "knowledgeGraph": data.get("knowledgeGraph", {}),
            }

        if not self.allow_fallback_search:
            return {
                "success": False,
                "error": "SERPER_API_KEY not set. Configure Serper for high-fidelity behavior.",
                "results": [],
            }

        # Optional fallback mode: lightweight DDG HTML scrape.
        qs = urllib.parse.urlencode({"q": q})
        url = f"https://duckduckgo.com/html/?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            html = r.read().decode("utf-8", errors="replace")
        links = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="(.*?)"[^>]*>(.*?)</a>',
            html,
            flags=re.S,
        )[:10]
        results = []
        for href, title_html in links:
            title = re.sub(r"<.*?>", "", title_html).strip()
            results.append({"title": title, "link": href})
        return {
            "success": True,
            "error": "",
            "results": results,
            "note": "SERPER_API_KEY not set; used DuckDuckGo fallback.",
        }

    def _scrape_and_extract(self, args: Dict[str, Any]) -> Dict[str, Any]:
        url = args.get("url")
        info = args.get("info_to_extract", "")
        if not url:
            return {"success": False, "error": "Missing required argument 'url'."}
        if self._blocked(str(url)):
            return {
                "success": False,
                "error": "Access to HuggingFace is disabled in this tool.",
            }

        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/plain, text/html, */*"}
        custom = args.get("custom_headers") or {}
        if isinstance(custom, dict):
            for k, v in custom.items():
                headers[str(k)] = str(v)
        jina_key = os.getenv("JINA_API_KEY", "")
        jina_base = os.getenv("JINA_BASE_URL", "").strip()
        fetch_url = str(url)
        if jina_base:
            if "{url}" in jina_base:
                fetch_url = jina_base.format(url=str(url))
            else:
                fetch_url = jina_base.rstrip("/") + "/" + str(url)
            if jina_key:
                headers["Authorization"] = f"Bearer {jina_key}"

        req = urllib.request.Request(fetch_url, headers=headers)
        with urllib.request.urlopen(req, timeout=45) as r:
            text = r.read().decode("utf-8", errors="replace")
        lines = text.splitlines()
        # Heuristic extraction: first non-empty lines + optional keyword windows.
        brief = []
        for ln in lines:
            s = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", ln)).strip()
            if len(s) >= 40:
                brief.append(s)
            if len(brief) >= 20:
                break
        extracted = "\n".join(brief[:12])[:6000]
        if info and self.extractor_client is not None:
            prompt = (
                "Extract the requested information from the scraped content.\n"
                "If uncertain, say so briefly.\n\n"
                f"Request: {info}\n\n"
                f"Content:\n{text[:12000]}"
            )
            try:
                extraction = self.extractor_client.complete(
                    messages=[
                        {
                            "role": "system",
                            "content": "You extract factual information from web content.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                extracted = extraction.content or extracted
            except Exception:
                extracted = f"Requested extraction target: {info}\n\n{extracted}"
        elif info:
            extracted = f"Requested extraction target: {info}\n\n{extracted}"
        return {
            "success": True,
            "url": url,
            "extracted_info": extracted or "(No extractable text found.)",
            "error": "",
            "scrape_stats": {
                "line_count": len(lines),
                "char_count": len(text),
                "last_char_line": len(lines),
                "all_content_displayed": False,
            },
            "model_used": "heuristic-local",
            "tokens_used": None,
        }

    def _tool_python(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "create_sandbox":
            sid = f"sandbox_{secrets.token_hex(4)}"
            d = Path(tempfile.mkdtemp(prefix=f"{sid}_", dir=self.scratch_dir))
            self.sandboxes[sid] = d
            return {"sandbox_id": sid}

        sid = str(args.get("sandbox_id", ""))
        if sid not in self.sandboxes:
            return {"success": False, "error": f"Unknown sandbox_id: {sid}"}
        sdir = self.sandboxes[sid]

        if tool_name == "run_command":
            cmd = args.get("command", "")
            p = subprocess.run(
                cmd,
                shell=True,
                cwd=sdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "stdout": p.stdout,
                "stderr": p.stderr,
                "exit_code": p.returncode,
                "error": "",
            }

        if tool_name == "run_python_code":
            code = args.get("code_block", "")
            script = sdir / "_tmp_run.py"
            script.write_text(code, encoding="utf-8")
            p = subprocess.run(
                ["python3", str(script)],
                cwd=sdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "stdout": p.stdout,
                "stderr": p.stderr,
                "exit_code": p.returncode,
                "error": "",
            }

        if tool_name == "upload_file_from_local_to_sandbox":
            src = Path(args["local_file_path"]).expanduser()
            dst = sdir / Path(src.name)
            dst.write_bytes(src.read_bytes())
            return {"uploaded_path": str(dst)}

        if tool_name == "download_file_from_internet_to_sandbox":
            url = str(args["url"])
            name = Path(urllib.parse.urlparse(url).path).name or "download.bin"
            dst = sdir / name
            with urllib.request.urlopen(url, timeout=45) as r:
                dst.write_bytes(r.read())
            return {"downloaded_path": str(dst)}

        if tool_name == "download_file_from_sandbox_to_local":
            src = sdir / Path(args["sandbox_file_path"]).name
            name = args.get("local_filename") or src.name
            dst = self.scratch_dir / name
            dst.write_bytes(src.read_bytes())
            return {"local_path": str(dst)}

        return {
            "success": False,
            "error": f"Unsupported tool-python method: {tool_name}",
        }


def run_harness(
    question: str,
    model: str,
    output_path: Path,
    key: str,
    row_id: str,
    max_assistant_turns: int,
    temperature: float,
    strict_protocol: bool,
    min_tool_turns: int,
    repair_attempts: int,
    allow_fallback_search: bool,
    force_think_tag: bool,
    request_reasoning: bool,
    internal_protocol_retry: bool,
    max_internal_protocol_retries: int,
    record_protocol_repairs: bool,
    api_key: str | None = None,
    api_base: str | None = None,
    callbacks: HarnessCallbacks | None = None,
) -> Dict[str, Any]:
    today = dt.date.today().isoformat()
    system_prompt = build_system_prompt(today=today)

    convo: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    callbacks = callbacks or HarnessCallbacks()
    client = OpenAIChatClient(model=model, api_key=api_key, base_url=api_base)
    tools = ToolExecutor(
        scratch_dir=output_path.parent,
        allow_fallback_search=allow_fallback_search,
        block_huggingface=True,
        extractor_model=os.getenv("EXTRACTOR_MODEL", model),
    )
    assistant_tool_turns = 0
    consecutive_repairs = 0

    for turn in range(1, max_assistant_turns + 1):
        retry_count = 0
        while True:
            msgs = [{"role": c["role"], "content": c["content"]} for c in convo]
            if retry_count > 0:
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            "Protocol reminder: return exactly one <use_mcp_tool> block with valid JSON arguments. "
                            "Do not provide final answer yet."
                        ),
                    }
                )
            completion = client.complete(
                msgs,
                temperature=temperature,
                include_reasoning=request_reasoning,
            )
            assistant_content = completion.content
            if callbacks.on_reasoning:
                callbacks.on_reasoning(turn, completion.reasoning, assistant_content)
            if completion.reasoning and "<think>" not in assistant_content:
                assistant_content = (
                    f"<think>\n{completion.reasoning}\n</think>\n\n{assistant_content}"
                )
            if force_think_tag and "<think>" not in assistant_content:
                assistant_content = (
                    "<think>\nReasoning process omitted in this trace.\n</think>\n\n"
                    + assistant_content
                )

            calls = MCPParser.parse_all(assistant_content)
            too_many_calls = strict_protocol and len(calls) > 1
            missing_required_call = (
                strict_protocol
                and len(calls) == 0
                and assistant_tool_turns < min_tool_turns
            )
            protocol_violation = too_many_calls or missing_required_call

            if (
                protocol_violation
                and internal_protocol_retry
                and retry_count < max_internal_protocol_retries
            ):
                retry_count += 1
                continue

            convo.append({"role": "assistant", "content": assistant_content})

            if too_many_calls:
                if callbacks.on_issue:
                    callbacks.on_issue(
                        "protocol",
                        "Multiple <use_mcp_tool> blocks found in a single turn.",
                    )
                if record_protocol_repairs:
                    convo.append({"role": "user", "content": REPAIR_MSG_MULTIPLE_TOOLS})
                consecutive_repairs += 1
                if consecutive_repairs > repair_attempts:
                    row = {"key": key, "id": row_id, "conversations": convo}
                    output_path.write_text(
                        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    return row
                break

            call = calls[0] if calls else None
            if call is None:
                if missing_required_call:
                    if callbacks.on_issue:
                        callbacks.on_issue(
                            "protocol",
                            "No <use_mcp_tool> block found before minimum tool turns.",
                        )
                    if record_protocol_repairs:
                        convo.append(
                            {"role": "user", "content": REPAIR_MSG_MISSING_TOOL}
                        )
                    consecutive_repairs += 1
                    if consecutive_repairs > repair_attempts:
                        row = {"key": key, "id": row_id, "conversations": convo}
                        output_path.write_text(
                            json.dumps(row, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        return row
                    break
                row = {"key": key, "id": row_id, "conversations": convo}
                output_path.write_text(
                    json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if callbacks.on_done:
                    callbacks.on_done(assistant_content)
                return row

            if callbacks.on_tool_call:
                callbacks.on_tool_call(turn, call)
            tool_result = tools.execute(call)
            if callbacks.on_tool_result:
                callbacks.on_tool_result(turn, call, tool_result)
            assistant_tool_turns += 1
            consecutive_repairs = 0
            convo.append(
                {
                    "role": "user",
                    "content": json.dumps(tool_result, ensure_ascii=False, indent=2),
                }
            )
            break

    row = {"key": key, "id": row_id, "conversations": convo}
    output_path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if callbacks.on_stopped:
        callbacks.on_stopped(max_assistant_turns)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ToolMind-style trajectories.")
    p.add_argument("--question", required=True, help="Initial user question.")
    p.add_argument("--output", required=True, help="Output JSON file path.")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--key", default="wikiQA-en-local")
    p.add_argument("--id", dest="row_id", default=f"local_{secrets.token_hex(4)}")
    p.add_argument("--max-assistant-turns", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--strict-protocol",
        action="store_true",
        help="Enforce one-tool-per-turn protocol.",
    )
    p.add_argument(
        "--min-tool-turns",
        type=int,
        default=8,
        help="Minimum tool turns before final answer.",
    )
    p.add_argument(
        "--repair-attempts",
        type=int,
        default=3,
        help="Max consecutive protocol repair turns.",
    )
    p.add_argument(
        "--allow-fallback-search",
        action="store_true",
        help="Allow DuckDuckGo fallback when SERPER_API_KEY is missing.",
    )
    p.add_argument(
        "--force-think-tag",
        action="store_true",
        help="Inject a <think> block when the model omits it.",
    )
    p.add_argument(
        "--request-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request separate reasoning output from API and inject into <think> (default: true).",
    )
    p.add_argument(
        "--internal-protocol-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry protocol-invalid assistant outputs internally (default: true).",
    )
    p.add_argument(
        "--max-internal-protocol-retries",
        type=int,
        default=2,
        help="Max hidden retries per turn for protocol-invalid outputs.",
    )
    p.add_argument(
        "--record-protocol-repairs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record protocol repair user messages in trajectory (default: false).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    row = run_harness(
        question=args.question,
        model=args.model,
        output_path=out,
        key=args.key,
        row_id=args.row_id,
        max_assistant_turns=args.max_assistant_turns,
        temperature=args.temperature,
        strict_protocol=args.strict_protocol,
        min_tool_turns=args.min_tool_turns,
        repair_attempts=args.repair_attempts,
        allow_fallback_search=args.allow_fallback_search,
        force_think_tag=args.force_think_tag,
        request_reasoning=args.request_reasoning,
        internal_protocol_retry=args.internal_protocol_retry,
        max_internal_protocol_retries=args.max_internal_protocol_retries,
        record_protocol_repairs=args.record_protocol_repairs,
    )
    print(f"Wrote {out} with {len(row['conversations'])} turns.")


if __name__ == "__main__":
    main()
