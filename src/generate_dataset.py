"""
Generate synthetic multi-turn conversations using the OpenAI API (GPT-3.5-turbo).

Each conversation is stored as a JSONL record:
  { "attribute": "age", "subcategory": "adult", "turns": [ {"role": "user"|"assistant", "content": "..."}, ... ] }

Usage:
    python src/generate_dataset.py --attribute age --n 1000
    python src/generate_dataset.py --all          # generate all attributes
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

from config import ALL_ATTRIBUTES, CONV_DIR, GEN_MODEL, N_TURNS, SEED

random.seed(SEED)

# System prompt template (matches paper footnote 3)
SYSTEM_PROMPT = (
    "You are a helpful AI assistant participating in a realistic multi-turn chat "
    "conversation with a human user."
)

USER_GEN_PROMPT = (
    "Generate a realistic multi-turn conversation between a human user and an AI assistant. "
    "The human user is {desc}. Make sure the conversation authentically reflects this "
    "user's background. Be creative with the topics. "
    "The conversation should have exactly {n_turns} turns (one user message + one assistant "
    "reply = one turn). "
    "Return ONLY a JSON array of objects with 'role' (\"user\" or \"assistant\") and "
    "'content' keys, starting with the user."
)


def generate_conversation(client: OpenAI, desc: str, n_turns: int = N_TURNS) -> list[dict]:
    """Call the API and return a list of {role, content} dicts."""
    prompt = USER_GEN_PROMPT.format(desc=desc, n_turns=n_turns)
    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    # The model may return {"conversation": [...]} or just [...]
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        turns = parsed
    else:
        # find the first list value
        turns = next(v for v in parsed.values() if isinstance(v, list))
    return turns


def generate_for_attribute(
    client: OpenAI,
    attr_name: str,
    n_total: int,
    out_path: Path,
    resume: bool = True,
) -> None:
    attr = ALL_ATTRIBUTES[attr_name]
    subcategories = attr["subcategories"]
    prompt_descs  = attr["prompt_desc"]
    n_per_class   = n_total // len(subcategories)

    # Load already-generated counts if resuming
    existing: dict[str, int] = {s: 0 for s in subcategories}
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["subcategory"]] += 1
        print(f"[{attr_name}] Resuming — existing counts: {existing}")

    with open(out_path, "a") as f:
        for sub in subcategories:
            needed = n_per_class - existing[sub]
            if needed <= 0:
                print(f"[{attr_name}/{sub}] Already complete, skipping.")
                continue
            print(f"[{attr_name}/{sub}] Generating {needed} conversations…")
            desc = prompt_descs[sub]
            for i in range(needed):
                for attempt in range(5):
                    try:
                        turns = generate_conversation(client, desc)
                        record = {
                            "attribute":    attr_name,
                            "subcategory":  sub,
                            "label":        subcategories.index(sub),
                            "turns":        turns,
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        break
                    except Exception as e:
                        wait = 2 ** attempt
                        print(f"  Attempt {attempt+1} failed: {type(e).__name__}: {e}. Retrying in {wait}s…")
                        time.sleep(wait)
                if (i + 1) % 50 == 0:
                    print(f"  [{attr_name}/{sub}] {i+1}/{needed} done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute", choices=list(ALL_ATTRIBUTES.keys()), help="Single attribute to generate")
    parser.add_argument("--all", action="store_true", help="Generate all attributes")
    parser.add_argument("--n", type=int, default=None, help="Override n_convos for the attribute")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the OPENAI_API_KEY environment variable before running.")
    client = OpenAI(api_key=api_key)

    targets = list(ALL_ATTRIBUTES.keys()) if args.all else [args.attribute]
    if not targets or targets == [None]:
        parser.error("Specify --attribute or --all")

    for attr_name in targets:
        n = args.n or ALL_ATTRIBUTES[attr_name]["n_convos"]
        out_path = CONV_DIR / f"{attr_name}.jsonl"
        if args.no_resume and out_path.exists():
            out_path.unlink()
        generate_for_attribute(client, attr_name, n, out_path, resume=not args.no_resume)
        print(f"[{attr_name}] Done → {out_path}")


if __name__ == "__main__":
    main()
