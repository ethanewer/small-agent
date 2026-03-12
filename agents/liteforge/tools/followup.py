from __future__ import annotations

from typing import Any


def execute(args: dict[str, Any], env: dict[str, Any]) -> str:
    question = args.get("question", "")
    multiple = args.get("multiple", False)
    options = []
    for i in range(1, 6):
        opt = args.get(f"option{i}")
        if opt:
            options.append(opt)

    if not question:
        return "Error: question is required"

    print(f"\n{'=' * 60}")
    print(f"Agent asks: {question}")

    if options:
        print("\nOptions:")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        if multiple:
            print("\n(You may select multiple, comma-separated)")
        print()

    try:
        response = input("Your response: ").strip()
    except (EOFError, KeyboardInterrupt):
        response = ""

    if not response:
        return "(No response provided)"

    if options and response.replace(",", "").replace(" ", "").isdigit():
        indices = [int(x.strip()) for x in response.split(",") if x.strip().isdigit()]
        selected = []
        for idx in indices:
            if 1 <= idx <= len(options):
                selected.append(options[idx - 1])
        if selected:
            return ", ".join(selected)

    return response
