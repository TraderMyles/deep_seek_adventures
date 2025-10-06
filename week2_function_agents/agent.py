# agent.py

import os
import json
import sys
import re
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI
from rich import print
from rich.console import Console
from rich.prompt import Prompt

from tools import tool_calc, tool_search, tool_fetch_url

console = Console()

# -------- DeepSeek inline tool-call parsing -----------------------------------
DEEPSEEK_TOOL_BLOCK = re.compile(
    r"<\｜tool▁call▁begin｜>(.*?)<\｜tool▁sep｜>(.*?)<\｜tool▁call▁end｜>",
    re.DOTALL
)

def extract_deepseek_tool_calls(text: str):
    """
    Extract tool calls from DeepSeek's inline markup.
    Returns: [{"name": str, "arguments": dict}, ...]
    """
    calls = []
    if not text:
        return calls
    for m in DEEPSEEK_TOOL_BLOCK.finditer(text):
        raw_name = (m.group(1) or "").strip()
        raw_args = (m.group(2) or "").strip()
        # Map aliases to your tool names
        name = "fetch_url" if raw_name == "fetch" else raw_name
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": name, "arguments": args})
    return calls

def append_tool_exchange(messages, tool_name, tool_args, tool_result, tool_id_hint=None):
    """
    Append a tool call and its result to the conversation in an OpenAI-compatible way.
    """
    tool_call_obj = {
        "id": tool_id_hint or f"{tool_name}-{len(messages)}",
        "type": "function",
        "function": {"name": tool_name, "arguments": json.dumps(tool_args, ensure_ascii=False)}
    }
    messages.append({"role": "assistant", "tool_calls": [tool_call_obj]})
    messages.append({"role": "tool", "tool_call_id": tool_call_obj["id"], "content": json.dumps(tool_result, ensure_ascii=False)})

# -------- NEW: collect sources from prior tool outputs ------------------------
def collect_sources_from_messages(messages):
    """
    Scan prior tool results in `messages` and collect domains/urls.
    Works with outputs from tool_search() and tool_fetch_url().
    Returns an ordered, de-duplicated list of domains.
    """
    domains = []
    urls = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        try:
            payload = json.loads(m.get("content") or "{}")
        except json.JSONDecodeError:
            continue

        # From search: {"results": [{"title","url","snippet"}...]}
        if isinstance(payload, dict) and "results" in payload:
            for r in payload.get("results", []):
                u = r.get("url")
                if u:
                    urls.append(u)

        # From fetch_url: {"domain","url","excerpt"...}
        if isinstance(payload, dict) and ("domain" in payload or "url" in payload):
            if payload.get("url"):
                urls.append(payload["url"])
            if payload.get("domain"):
                domains.append(payload["domain"])

    # Normalize any URLs to domains
    def _domain(u: str) -> str:
        m = re.match(r"https?://([^/]+)/?", u or "")
        return m.group(1) if m else u

    domains.extend(_domain(u) for u in urls if u)

    # De-duplicate, preserve order
    seen, ordered = set(), []
    for d in domains:
        if d and d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered

# -------- Prompts / Client / Tools -------------------------------------------
def load_prompts() -> Dict[str, Any]:
    p = Path("prompts.json")
    if not p.exists():
        return {"system": "You are a helpful research assistant.", "style_rules": []}
    return json.loads(p.read_text(encoding="utf-8"))

def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        console.print("[red]Missing DEEPSEEK_API_KEY in .env[/red]")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=base_url)

def tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Evaluate a short math expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "e.g., '2+2*3' or 'sqrt(9)+10'"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch a URL and extract readable text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "max_chars": {"type": "integer", "minimum": 500, "maximum": 8000, "default": 2000}
                    },
                    "required": ["url"]
                }
            }
        }
    ]

def dispatch_tool(name: str, arguments: dict) -> dict | list | str:
    try:
        if name == "calc":
            return {"result": tool_calc(arguments["expression"])}
        if name == "search":
            return {"results": tool_search(arguments["query"], arguments.get("num_results", 5))}
        if name == "fetch_url":
            return tool_fetch_url(arguments["url"], arguments.get("max_chars", 2000))
        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": f"TOOL_ERROR: {e}"}

def save_run(transcript: dict) -> Path:
    runs = Path("runs"); runs.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    p = runs / f"research-{ts}.json"
    p.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

# -------- Main loop -----------------------------------------------------------
def main():
    prompts = load_prompts()
    system_msg = prompts["system"]
    style_rules = prompts.get("style_rules", [])

    client = build_client()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not user_query:
        user_query = Prompt.ask("[bold]Ask your research question[/bold]")

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_query},
        {"role": "system", "content": "Style rules: " + "; ".join(style_rules)}
    ]

    # First call (let the model choose tools)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tool_specs(),
        tool_choice="auto",
        temperature=0.2,
    )

    while True:
        choice = response.choices[0]
        msg = choice.message

        handled_any_tools = False

        # 1) Standard OpenAI tool_calls
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                tool_result = dispatch_tool(name, args)
                append_tool_exchange(messages, name, args, tool_result, tool_id_hint=tool_call.id)
                handled_any_tools = True

        # 2) DeepSeek inline tool tags (when tool_calls is empty)
        else:
            ds_calls = extract_deepseek_tool_calls(msg.content or "")
            if ds_calls:
                for call in ds_calls:
                    name, args = call["name"], call["arguments"]
                    tool_result = dispatch_tool(name, args)
                    append_tool_exchange(messages, name, args, tool_result)
                handled_any_tools = True

        if handled_any_tools:
            # Continue now that tools have returned results
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2
            )
            continue

        # No tool calls → final answer
        final = msg.content or ""

        # --- NEW: auto-append Sources (domains) if any tools were used
        sources = collect_sources_from_messages(messages)
        if sources:
            final = final.rstrip() + "\n\n**Sources:** " + ", ".join(sources[:6])

        print("\n[bold green]Answer[/bold green]\n" + final)
        transcript = {"query": user_query, "messages": messages, "final": final}
        path = save_run(transcript)
        print(f"\n[green]Saved →[/green] {path}")
        break

if __name__ == "__main__":
    main()
