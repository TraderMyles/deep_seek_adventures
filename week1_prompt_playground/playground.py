import json
import os
import sys
from datetime import datetime, UTC
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rich import print
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

# OpenAI SDK works with DeepSeek's OpenAI-compatible API
from openai import OpenAI
import argparse

console = Console()

# ---- New: creativity presets -------------------------------------------------
PRESETS = {
    # conservative, repeatable
    "low":        {"temperature": 0.1, "top_p": 1.0},
    # balanced default
    "medium":     {"temperature": 0.3, "top_p": 1.0},
    # more exploratory/varied
    "high":       {"temperature": 0.7, "top_p": 1.0},
    # for structured outputs like JSON, lists, codey stuff
    "deterministic": {"temperature": 0.0, "top_p": 1.0},
    # extra creative for ideation/brainstorms
    "creative":   {"temperature": 0.9, "top_p": 1.0},
}

def load_prompts(path: Path) -> dict:
    if not path.exists():
        return {"system_presets": {"neutral": "You are a helpful assistant."}, "user_samples": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def render_template(tmpl: str, **kwargs) -> str:
    out = tmpl
    for k, v in kwargs.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out

def ensure_runs_dir() -> Path:
    runs = Path("runs")
    runs.mkdir(exist_ok=True)
    return runs

def save_transcript(runs_dir: Path, data: dict) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = runs_dir / f"run-{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def build_client():
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        console.print("[red]Missing DEEPSEEK_API_KEY in .env[/red]")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=base_url)

@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
)
def chat_once(client, model, system_msg, user_msg, temperature, max_tokens, top_p):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=False,
    )
    return response.choices[0].message.content

def list_presets(prompts: dict):
    table = Table(title="System Presets")
    table.add_column("Key", style="bold cyan")
    table.add_column("System Message")
    for k, v in prompts.get("system_presets", {}).items():
        table.add_row(k, v)
    console.print(table)

    table2 = Table(title="User Templates")
    table2.add_column("Key", style="bold magenta")
    table2.add_column("Template (use {{var}})")
    for k, v in prompts.get("user_samples", {}).items():
        table2.add_row(k, v)
    console.print(table2)

    table3 = Table(title="Creativity Presets")
    table3.add_column("Preset", style="bold green")
    table3.add_column("temperature")
    table3.add_column("top_p")
    for name, cfg in PRESETS.items():
        table3.add_row(name, str(cfg["temperature"]), str(cfg["top_p"]))
    console.print(table3)

def apply_preset(args):
    """Apply a preset if provided. Then let explicit flags override the preset."""
    cfg = {"temperature": args.temperature, "top_p": args.top_p}
    if args.preset:
        preset = PRESETS.get(args.preset)
        if not preset:
            console.print(f"[red]Unknown preset:[/red] {args.preset}")
            sys.exit(2)
        cfg.update(preset)
    # explicit flags should win (only override if user supplied them)
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.top_p is not None:
        cfg["top_p"] = args.top_p
    return cfg

def main():
    parser = argparse.ArgumentParser(description="DeepSeek Prompt Playground CLI")
    parser.add_argument("--system", default="neutral", help="System preset key or raw text")
    parser.add_argument("--user", default=None, help="User template key or raw text")
    parser.add_argument("--text", default="", help="Value for {{text}}")
    parser.add_argument("--topic", default="", help="Value for {{topic}}")

    # Note: keep defaults None so presets can fill them; we show a computed value later
    parser.add_argument("--temperature", type=float, default=None, help="Creativity 0.0–1.0 (overrides preset)")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling 0.0–1.0 (overrides preset)")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Creativity profile (e.g., low, medium, high, deterministic, creative)")
    parser.add_argument("--list", action="store_true", help="List available presets/templates")
    parser.add_argument("--raw", action="store_true", help="Treat --system/--user as raw strings")
    args = parser.parse_args()

    prompts = load_prompts(Path("prompts.json"))
    if args.list:
        list_presets(prompts)
        sys.exit(0)

    # Resolve system message
    if args.raw:
        system_msg = args.system
        user_tmpl = args.user or ""
    else:
        system_msg = prompts.get("system_presets", {}).get(args.system, args.system)
        user_tmpl = prompts.get("user_samples", {}).get(args.user, args.user or "")

    # If no user provided via arg, read from stdin
    if not user_tmpl:
        console.print("[bold]Enter your prompt (end with Ctrl+D/Ctrl+Z):[/bold]\n")
        try:
            user_tmpl = sys.stdin.read().strip()
        except KeyboardInterrupt:
            console.print("\n[red]Cancelled[/red]")
            sys.exit(1)

    user_msg = render_template(user_tmpl, text=args.text, topic=args.topic)

    # Apply preset + overrides
    cfg = apply_preset(args)
    # Show effective values (fall back to a nice default for display if still None)
    effective_temperature = cfg["temperature"] if cfg["temperature"] is not None else 0.3
    effective_top_p = cfg["top_p"] if cfg["top_p"] is not None else 1.0

    console.rule("[bold green]DeepSeek Prompt Playground[/bold green]")
    console.print(f"[b]Model:[/b] {args.model}")
    console.print(f"[b]Preset:[/b] {args.preset or '—'}   [b]Temperature:[/b] {effective_temperature}   [b]Top-p:[/b] {effective_top_p}   [b]Max tokens:[/b] {args.max_tokens}")
    console.print("\n[bold cyan]System[/bold cyan]:\n" + system_msg)
    console.print("\n[bold magenta]User[/bold magenta]:\n" + user_msg)
    console.print("")

    if not Confirm.ask("Send this to the model?", default=True):
        console.print("[yellow]Aborted.[/yellow]")
        sys.exit(0)

    client = build_client()

    try:
        output = chat_once(
            client=client,
            model=args.model,
            system_msg=system_msg,
            user_msg=user_msg,
            temperature=effective_temperature,
            max_tokens=args.max_tokens,
            top_p=effective_top_p,
        )
    except Exception as e:
        console.print(f"[red]Request failed:[/red] {e}")
        sys.exit(1)

    console.rule("[bold]Response[/bold]")
    print(output)

    runs_dir = ensure_runs_dir()
    transcript = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "preset": args.preset,
        "temperature": effective_temperature,
        "top_p": effective_top_p,
        "max_tokens": args.max_tokens,
        "system": system_msg,
        "user": user_msg,
        "response": output,
    }
    path = save_transcript(runs_dir, transcript)
    console.print(f"\n[green]Saved transcript →[/green] {path}")

if __name__ == "__main__":
    main()
