# tools.py

import os
import math
import re
import requests
from dataclasses import dataclass
from typing import List, Dict, Any

from bs4 import BeautifulSoup
from readability import Document

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

# ---- Tool: calculator --------------------------------------------------------
def tool_calc(expression: str) -> str:
    allowed = {"__builtins__": {}}
    safe_funcs = {
        "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
        "sin": math.sin, "cos": math.cos, "tan": math.tan
    }
    allowed.update(safe_funcs)
    try:
        value = eval(expression, allowed)
        return str(value)
    except Exception as e:
        return f"CALC_ERROR: {e}"

# ---- Tool: web search (Tavily) ----------------------------------------------
def tool_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Live web search using Tavily API.
    Returns a list of dicts: {title, url, snippet}.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [{"error": "Missing TAVILY_API_KEY in .env"}]

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max(1, min(num_results, 10)),
                "include_answer": False,
                "include_images": False,
                "include_domains": [],
                "search_depth": "advanced"  # or "basic" to go faster/cheaper
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        out = []
        for r in results:
            out.append({
                "title": (r.get("title") or "")[:200],
                "url": r.get("url") or "",
                "snippet": (r.get("content") or "")[:500],
            })
        return out
    except Exception as e:
        return [{"error": f"TAVILY_ERROR: {e}"}]

# ---- Tool: fetch + extract main text ----------------------------------------
def tool_fetch_url(url: str, max_chars: int = 2000) -> Dict[str, str]:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        doc = Document(resp.text)
        title = doc.short_title()
        html = doc.summary()
        soup = BeautifulSoup(html, "html.parser")
        text = " ".join(soup.get_text(separator=" ").split())
        return {
            "title": title[:200] if title else "",
            "excerpt": text[:max_chars],
            "domain": _domain(url),
            "url": url
        }
    except Exception as e:
        return {"error": f"FETCH_ERROR: {e}", "url": url}

def _domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url)
    return m.group(1) if m else url
