"""
Extract residual-stream activations from LLaMA-2-Chat-13B.

For each conversation the last token of the probe suffix
  "I think the {attribute} of this user is"
is appended after the last user message (matching §4.2 of Chen et al.).
The hidden state at that token position is saved for every layer.

Output layout:
    data/activations/{attribute}.pt
        dict with keys:
            "X"     : FloatTensor [N, num_layers+1, hidden_dim]
            "y"     : LongTensor  [N]   (class index)
            "meta"  : list of {"subcategory": str, "conv_index": int}

Usage:
    python src/extract_activations.py --attribute age
    python src/extract_activations.py --all
    python src/extract_activations.py --all --device cuda:1
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ACT_DIR,
    ALL_ATTRIBUTES,
    ATTRIBUTES,
    CONV_DIR,
    HIDDEN_DIM,
    MODEL_ID,
    NUM_LAYERS,
)

# LLaMA-2-Chat conversation template (used as fallback)
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS   = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM  = "You are a helpful, respectful and honest assistant."

# Module-level tokenizer reference — set once in main(), used by format_chat
_TOKENIZER = None


def _normalise_turn(t) -> dict | None:
    """Coerce a turn to {"role": str, "content": str} or return None if unusable."""
    if isinstance(t, dict):
        role    = t.get("role") or t.get("speaker") or t.get("type", "")
        content = t.get("content") or t.get("text") or t.get("message", "")
        if role and content:
            role = role.lower().strip()
            if "user" in role or "human" in role:
                role = "user"
            elif "assistant" in role or "ai" in role or "bot" in role:
                role = "assistant"
            return {"role": role, "content": str(content)}
    if isinstance(t, str) and t.strip():
        # bare string — can't determine role, skip
        return None
    return None


def _build_rounds(turns: list) -> list[tuple[str, str | None]]:
    """Parse flat turn list into (user, assistant|None) round pairs."""
    # Normalise and drop unusable entries
    clean = [n for t in turns if (n := _normalise_turn(t)) is not None]
    rounds: list[tuple[str, str | None]] = []
    i = 0
    while i < len(clean):
        if clean[i]["role"] == "user":
            user_text = clean[i]["content"]
            asst_text = (
                clean[i + 1]["content"]
                if (i + 1 < len(clean) and clean[i + 1]["role"] == "assistant")
                else None
            )
            rounds.append((user_text, asst_text))
            i += 2 if asst_text is not None else 1
        else:
            i += 1
    return rounds


def format_chat(turns: list[dict], probe_suffix: str) -> str:
    """
    Build a prompt ending with the probe suffix after the last user message.

    Uses the tokenizer's apply_chat_template when available (handles Mistral,
    LLaMA-3, etc.), otherwise falls back to the LLaMA-2 manual template.
    """
    rounds = _build_rounds(turns)

    if _TOKENIZER is not None and hasattr(_TOKENIZER, "apply_chat_template"):
        # Build a strictly alternating user/assistant message list.
        # Embed system prompt in the first user message to avoid role issues.
        raw_messages: list[dict] = []
        for idx, (user_text, asst_text) in enumerate(rounds):
            if idx == 0:
                user_text = f"{DEFAULT_SYSTEM}\n\n{user_text}"
            raw_messages.append({"role": "user", "content": user_text})
            if idx < len(rounds) - 1:
                # Use a placeholder if assistant reply is missing
                reply = asst_text if asst_text else "I understand."
                raw_messages.append({"role": "assistant", "content": reply})

        # Merge any accidental consecutive same-role messages
        messages: list[dict] = []
        for msg in raw_messages:
            if messages and messages[-1]["role"] == msg["role"]:
                messages[-1]["content"] += " " + msg["content"]
            else:
                messages.append(dict(msg))

        prompt = _TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt + probe_suffix

    # ── LLaMA-2 manual fallback ────────────────────────────────────────────
    tokens_str = ""
    for idx, (user_text, asst_text) in enumerate(rounds):
        user_part = f"{B_SYS}{DEFAULT_SYSTEM}{E_SYS}{user_text}" if idx == 0 else user_text
        if idx < len(rounds) - 1:
            tokens_str += f"{B_INST} {user_part.strip()} {E_INST} {asst_text.strip()} </s><s>"
        else:
            tokens_str += f"{B_INST} {user_part.strip()} {E_INST} {probe_suffix}"
    return tokens_str


@torch.inference_mode()
def extract_last_token_hidden_states(
    model,
    tokenizer,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns FloatTensor [num_layers+1, hidden_dim] — hidden states of the
    last token at every layer (0 = embedding, 1..N = transformer blocks).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (num_layers+1) tensors each [1, seq_len, hidden_dim]
    hidden_states = outputs.hidden_states
    last_pos = inputs["input_ids"].shape[1] - 1
    # Stack across layers → [num_layers+1, hidden_dim]
    stacked = torch.stack([h[0, last_pos, :].cpu().float() for h in hidden_states])
    return stacked


def format_chat_up_to_turn(turns: list[dict], turn_idx: int, probe_suffix: str) -> str:
    """
    Build a LLaMA-2-Chat prompt that includes only turns 0..turn_idx (inclusive)
    and appends the probe suffix after the last included user message.

    turn_idx is 0-based and refers to *user* turns only.  So turn_idx=0 gives
    only the first user message + probe suffix; turn_idx=1 gives the first
    full exchange plus the second user message + probe suffix, etc.
    """
    # Collect user turns only, preserving interleaved assistant replies
    rounds: list[tuple[str, str | None]] = []
    i = 0
    while i < len(turns):
        if turns[i]["role"] == "user":
            user_text = turns[i]["content"]
            asst_text = (
                turns[i + 1]["content"]
                if (i + 1 < len(turns) and turns[i + 1]["role"] == "assistant")
                else None
            )
            rounds.append((user_text, asst_text))
            i += 2 if asst_text is not None else 1
        else:
            i += 1

    # Clip to requested turn
    rounds = rounds[: turn_idx + 1]

    tokens_str = ""
    for idx, (user_text, asst_text) in enumerate(rounds):
        user_part = f"{B_SYS}{DEFAULT_SYSTEM}{E_SYS}{user_text}" if idx == 0 else user_text
        if idx < len(rounds) - 1:
            tokens_str += f"{B_INST} {user_part.strip()} {E_INST} {asst_text.strip()} </s><s>"
        else:
            tokens_str += f"{B_INST} {user_part.strip()} {E_INST} {probe_suffix}"
    return tokens_str


def extract_for_attribute(
    attr_name: str,
    model,
    tokenizer,
    device: torch.device,
    overwrite: bool = False,
) -> None:
    out_path = ACT_DIR / f"{attr_name}.pt"
    if out_path.exists() and not overwrite:
        print(f"[{attr_name}] Activations already exist at {out_path}, skipping.")
        return

    conv_path = CONV_DIR / f"{attr_name}.jsonl"
    if not conv_path.exists():
        raise FileNotFoundError(f"No conversation file found at {conv_path}. Run generate_dataset.py first.")

    attr      = ALL_ATTRIBUTES[attr_name]
    probe_sfx = attr["probe_suffix"]

    records: list[dict] = []
    with open(conv_path) as f:
        for line in f:
            records.append(json.loads(line))

    X_list: list[torch.Tensor] = []
    y_list: list[int]          = []
    meta:   list[dict]         = []

    for idx, rec in enumerate(tqdm(records, desc=f"[{attr_name}] extracting")):
        if not rec.get("turns"):
            continue
        try:
            text   = format_chat(rec["turns"], probe_sfx)
            hidden = extract_last_token_hidden_states(model, tokenizer, text, device)
        except Exception as e:
            print(f"\n  Skipping conv {idx} ({rec.get('subcategory')}): {type(e).__name__}: {e}")
            continue
        # hidden: [num_layers+1, hidden_dim]
        X_list.append(hidden)
        y_list.append(rec["label"])
        meta.append({"subcategory": rec["subcategory"], "conv_index": idx})

    X = torch.stack(X_list)   # [N, num_layers+1, hidden_dim]
    y = torch.tensor(y_list, dtype=torch.long)  # [N]

    torch.save({"X": X, "y": y, "meta": meta}, out_path)
    print(f"[{attr_name}] Saved {len(records)} activations → {out_path}")
    print(f"  X shape: {X.shape}  |  classes: { {s: (y == i).sum().item() for i, s in enumerate(attr['subcategories'])} }")


def extract_turn_level_for_attribute(
    attr_name: str,
    model,
    tokenizer,
    device: torch.device,
    max_turns: int = 7,
    overwrite: bool = False,
) -> None:
    """
    Turn-level extraction: for each conversation, extract the probe-suffix
    hidden state after *each* user turn (not just the last).

    Output: data/activations/{attribute}_turn_level.pt
        {
          "X"    : FloatTensor [N, max_turns, num_layers+1, hidden_dim],
          "y"    : LongTensor  [N]   (static attribute label),
          "mask" : BoolTensor  [N, max_turns]  (True where turn exists),
          "meta" : list of {"subcategory": str, "n_turns": int, "conv_index": int}
        }

    This answers: does the representation *update* as the conversation develops?
    For dynamic attributes (knowledge_level, confusion) the label may change
    per turn — in that case each record should carry a "turn_labels" list.
    """
    out_path = ACT_DIR / f"{attr_name}_turn_level.pt"
    if out_path.exists() and not overwrite:
        print(f"[{attr_name}] Turn-level activations already exist at {out_path}, skipping.")
        return

    conv_path = CONV_DIR / f"{attr_name}.jsonl"
    if not conv_path.exists():
        raise FileNotFoundError(f"{conv_path} not found. Run generate_dataset.py first.")

    attr      = ALL_ATTRIBUTES[attr_name]
    probe_sfx = attr["probe_suffix"]
    subcats   = attr["subcategories"]

    records: list[dict] = []
    with open(conv_path) as f:
        for line in f:
            records.append(json.loads(line))

    num_layers_plus1 = NUM_LAYERS + 1
    X_list:    list[torch.Tensor] = []
    y_list:    list[int]          = []
    mask_list: list[torch.Tensor] = []
    meta:      list[dict]         = []

    for idx, rec in enumerate(tqdm(records, desc=f"[{attr_name}] turn-level extracting")):
        turns     = rec["turns"]
        # Count user turns
        user_turns = [t for t in turns if t["role"] == "user"]
        n_user    = min(len(user_turns), max_turns)

        # [max_turns, num_layers+1, hidden_dim] — padded with zeros
        X_conv  = torch.zeros(max_turns, num_layers_plus1, HIDDEN_DIM)
        mask    = torch.zeros(max_turns, dtype=torch.bool)

        for t in range(n_user):
            text   = format_chat_up_to_turn(turns, t, probe_sfx)
            hidden = extract_last_token_hidden_states(model, tokenizer, text, device)
            X_conv[t]  = hidden
            mask[t]    = True

        X_list.append(X_conv)
        # For dynamic attributes, "turn_labels" overrides the static label
        if "turn_labels" in rec:
            y_list.append(rec["turn_labels"][0])  # anchor on turn-0 label
        else:
            y_list.append(rec["label"])
        mask_list.append(mask)
        meta.append({
            "subcategory": rec["subcategory"],
            "n_turns": n_user,
            "conv_index": idx,
            "turn_labels": rec.get("turn_labels", [rec["label"]] * n_user),
        })

    X    = torch.stack(X_list)    # [N, max_turns, num_layers+1, hidden_dim]
    y    = torch.tensor(y_list, dtype=torch.long)
    mask = torch.stack(mask_list)  # [N, max_turns]

    torch.save({"X": X, "y": y, "mask": mask, "meta": meta}, out_path)
    print(f"[{attr_name}] Saved turn-level activations → {out_path}")
    print(f"  X shape: {X.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute", choices=list(ALL_ATTRIBUTES.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--turn-level", action="store_true",
                        help="Extract activations at every user turn (for dynamic tracking)")
    parser.add_argument("--max-turns", type=int, default=7,
                        help="Maximum turns to extract per conversation (turn-level mode)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization (bitsandbytes). "
                             "Cuts VRAM/RAM to ~7GB, works on CPU or low-VRAM GPU.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    targets = list(ALL_ATTRIBUTES.keys()) if args.all else [args.attribute]
    if not targets or targets == [None]:
        parser.error("Specify --attribute or --all")

    device = torch.device(args.device)
    print(f"Loading {MODEL_ID} on {device}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Make the tokenizer available to format_chat for apply_chat_template support
    global _TOKENIZER
    _TOKENIZER = tokenizer

    load_kwargs: dict = {"device_map": args.device}
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        print("  (8-bit quantization enabled via bitsandbytes)")
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()

    for attr_name in targets:
        if args.turn_level:
            extract_turn_level_for_attribute(
                attr_name, model, tokenizer, device,
                max_turns=args.max_turns,
                overwrite=args.overwrite,
            )
        else:
            extract_for_attribute(attr_name, model, tokenizer, device, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
