"""Command line: `python -m ragguard ask "..."` and `python -m ragguard eval`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate, format_table
from .pipeline import build_pipeline

CONFIGS = {
    "baseline (no gate, no guard)": {"use_gate": False, "use_guard": False},
    "gate only": {"use_gate": True, "use_guard": False},
    "gate + guard": {"use_gate": True, "use_guard": True},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragguard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--backend", default="extractive")

    ev = sub.add_parser("eval", help="run the evaluation set under each configuration")
    ev.add_argument("--backend", default="extractive")
    ev.add_argument("--out", type=Path, help="write the full JSON report here")

    args = parser.parse_args(argv)

    if args.command == "ask":
        result = build_pipeline(backend=args.backend).ask(args.question)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    reports = {
        label: evaluate(build_pipeline(backend=args.backend, **options))
        for label, options in CONFIGS.items()
    }
    print(f"Backend: {args.backend}\n")
    print(format_table(reports))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
