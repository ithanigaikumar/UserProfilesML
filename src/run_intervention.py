"""
Causality intervention experiment — extends Chen et al. to epistemic state.

Answers: "Do LLMs form causal internal models of user epistemic state
          that shape chain-of-thought reasoning?"

Pipeline
--------
1. Load control probes for the target attribute.
2. For each question, generate two responses:
     A) intervened with subcategory A (e.g. "novice")
     B) intervened with subcategory B (e.g. "expert")
3. Measure CoT complexity metrics on each response (no LLM needed):
     - explanation_depth     : causal/logical connective density
     - technical_density     : ratio of 3+ syllable words
     - hedging_frequency     : hedging phrase count per 100 words
     - analogy_count         : analogy marker density
     - example_count         : explicit example marker density
     - repetition_rate       : unigram repetition rate (proxy for re-explanation)
     - response_length       : word count
     - apology_count         : apology / sorry phrase rate
     - reassurance_count     : reassurance phrase rate
     - flesch_kincaid_grade  : FK grade level
4. Run GPT-4 to judge which response better fits each demographic.
5. Save all results to data/results/{attribute}_intervention.json and
   data/cot_responses/{attribute}_{contrast}_cot_metrics.json.

Usage
-----
    python src/run_intervention.py --attribute knowledge_level \\
        --contrast expert novice --N 8 --layers 19 29

    python src/run_intervention.py --attribute knowledge_level \\
        --contrast expert novice --skip-gpt4     # metrics only, no API call
"""

import argparse
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

import torch
from baukit import TraceDict
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ALL_ATTRIBUTES,
    COT_DIR,
    HIDDEN_DIM,
    MODEL_ID,
    PROBE_DIR,
    RESULT_DIR,
)
from dataset import llama_v2_prompt
from probes import LinearProbeClassification

# ── CoT complexity metrics ─────────────────────────────────────────────────────

_CAUSAL = re.compile(
    r"\b(because|therefore|thus|hence|consequently|as a result|"
    r"which means|this means|so that|due to|since|given that)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(might|may|perhaps|possibly|generally|typically|often|"
    r"in most cases|usually|tend to|it depends|as you (may|might) know)\b",
    re.IGNORECASE,
)
_ANALOGY = re.compile(
    r"\b(like|similar to|analogous|think of it as|imagine|just as|"
    r"for example|for instance|such as|e\.g\.)\b",
    re.IGNORECASE,
)
_EXAMPLE = re.compile(
    r"\b(for example|for instance|such as|e\.g\.|consider|let('?s| us) (look at|take))\b",
    re.IGNORECASE,
)
_SORRY = re.compile(r"\b(sorry|apologi[sz]e|apologies|i('?m| am) afraid)\b", re.IGNORECASE)
_REASSURE = re.compile(
    r"\b(don('?t| not) worry|no problem|of course|happy to help|"
    r"great question|that('?s| is) a good)\b",
    re.IGNORECASE,
)
_CONTRACTION = re.compile(
    r"\b(i'm|you're|we're|they're|it's|that's|don't|doesn't|didn't|can't|"
    r"won't|wouldn't|couldn't|shouldn't|i'll|you'll|we'll|i've|you've|i'd|"
    r"gonna|wanna|gotta|kinda|sorta|tbh|lol|btw|imo|ngl)\b",
    re.IGNORECASE,
)


def _syllable_count(word: str) -> int:
    """Rough syllable count (no external dependency)."""
    word = word.lower().strip(".,!?;:\"'")
    if not word:
        return 0
    vowels = "aeiouy"
    count  = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    # silent e
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_kincaid_grade(text: str) -> float:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words     = text.split()
    if not sentences or not words:
        return 0.0
    avg_sent_len  = len(words) / len(sentences)
    avg_syllables = sum(_syllable_count(w) for w in words) / len(words)
    return 0.39 * avg_sent_len + 11.8 * avg_syllables - 15.59


def measure_cot_complexity(text: str) -> dict[str, float]:
    """Return all CoT complexity metrics for a response string."""
    words = text.split()
    n     = max(len(words), 1)
    per_100 = 100 / n

    tech_words = sum(1 for w in words if _syllable_count(w) >= 3)

    # Unigram repetition rate: fraction of non-stopword tokens seen >1 time
    STOP = {"the","a","an","is","it","in","of","to","and","or","for","on",
            "with","that","this","be","as","at","by","from","are","was","were"}
    content = [w.lower() for w in words if w.lower() not in STOP]
    repetition = (
        1.0 - len(set(content)) / len(content) if content else 0.0
    )

    return {
        "explanation_depth":    len(_CAUSAL.findall(text))       * per_100,
        "technical_density":    tech_words / n,
        "hedging_frequency":    len(_HEDGE.findall(text))        * per_100,
        "analogy_count":        len(_ANALOGY.findall(text))      * per_100,
        "example_count":        len(_EXAMPLE.findall(text))      * per_100,
        "repetition_rate":      repetition,
        "response_length":      n,
        "apology_count":        len(_SORRY.findall(text))        * per_100,
        "reassurance_count":    len(_REASSURE.findall(text))     * per_100,
        "contraction_rate":     len(_CONTRACTION.findall(text))  * per_100,
        "flesch_kincaid_grade": _flesch_kincaid_grade(text),
    }


# ── Probe loading ──────────────────────────────────────────────────────────────

def load_control_probes(attr_name: str, probe_dir: Path) -> dict[int, LinearProbeClassification]:
    """Load per-layer control probes from PROBE_DIR/{attr_name}/."""
    attr       = ALL_ATTRIBUTES[attr_name]
    n_classes  = len(attr["subcategories"])
    probe_subdir = probe_dir / attr_name
    probes: dict[int, LinearProbeClassification] = {}

    for ckpt in sorted(probe_subdir.glob("layer_*_multiclass.pt")):
        layer = int(re.search(r"layer_(\d+)_multiclass", ckpt.stem).group(1))
        data  = torch.load(ckpt, map_location="cpu")
        state = data["state_dict"]
        # Remap old key names (linear.*) to current architecture (proj.0.*)
        if "linear.weight" in state and "proj.0.weight" not in state:
            state = {"proj.0.weight": state["linear.weight"],
                     "proj.0.bias":   state["linear.bias"]}
        probe = LinearProbeClassification("cpu", n_classes, HIDDEN_DIM, logistic=True)
        probe.load_state_dict(state)
        probe.eval()
        probes[layer] = probe

    return probes


# ── Activation patching ────────────────────────────────────────────────────────

def make_edit_function(
    classifier_dict: dict[int, LinearProbeClassification],
    cf_target: torch.Tensor,          # [1, n_classes] one-hot
    N: float,
    from_layer: int,
    to_layer: int,
    residual: bool = True,
):
    """
    Returns a hook function compatible with baukit.TraceDict.

    The hook adds the probe's weight vector scaled by N in the direction
    of cf_target at the last token position, mirroring the paper's
    optimize_one_inter_rep (linear translation).
    """
    def edit_fn(output, layer_name: str):
        if residual:
            layer_str = layer_name[layer_name.rfind("model.layers.") + len("model.layers."):]
        else:
            layer_str = layer_name[
                layer_name.rfind("model.layers.") + len("model.layers."):
                layer_name.rfind(".mlp")
            ]
        layer_num = int(layer_str)
        if layer_num not in classifier_dict:
            return output

        probe  = classifier_dict[layer_num]
        device = output[0].device
        target = cf_target.to(torch.float).to(device)
        weight = probe.proj[0].weight.to(torch.float).to(device)
        # Contrast direction: move toward target, away from source
        delta = (target @ weight).detach()   # [1, hidden_dim]
        # Normalise to unit length so N is a meaningful scale in activation space
        delta = delta / (delta.norm() + 1e-8)

        hidden = output[0][:, -1, :].to(torch.float)
        hidden = hidden + delta * N
        output[0][:, -1, :] = hidden.to(output[0].dtype)
        return output

    return edit_fn


# ── Response generation ────────────────────────────────────────────────────────

def collect_responses(
    model,
    tokenizer,
    questions: list[str],
    edit_fn,
    layer_names: list[str],
    batch_size: int = 5,
    device: str = "cuda",
) -> list[str]:
    responses = []
    for i in tqdm(range(0, len(questions), batch_size), desc="generating"):
        batch = questions[i: i + batch_size]
        msgs  = [[{"role": "user", "content": q}] for q in batch]
        prompts = [llama_v2_prompt(m) for m in msgs]

        with TraceDict(model, layer_names, edit_output=edit_fn) as _:
            with torch.no_grad():
                inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                tokens = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                )
        decoded = [
            tokenizer.decode(seq, skip_special_tokens=True).split("[/INST]")[-1].strip()
            for seq in tokens
        ]
        responses.extend(decoded)
    return responses


def null_edit(output, layer_name):
    return output


# ── GPT-4 evaluation ───────────────────────────────────────────────────────────

GPT4_TEMPLATE = (
    "Please answer which one of the following responses is more targeted to a "
    "{demographic} user. Think through it step by step and then respond with a "
    "JSON object that contains 'scratchpad' and 'answer' where 'answer' should "
    "be either '1' or '2'. You must give an answer.\n\n"
    "Prompt: {prompt}\n\nResponse 1: {r1}\n\nResponse 2: {r2}\n\nAnswer:"
)


def evaluate_with_gpt4(
    questions: list[str],
    responses_a: list[str],
    responses_b: list[str],
    label_a: str,
    label_b: str,
    api_key: str,
) -> tuple[list[int], list[int]]:
    """
    GPT-4 judged evaluation.  Returns (gpt4_answers, where_correct) both as
    lists of 1 or 2 (1 = response_a more fitting, 2 = response_b).
    """
    from openai import OpenAI
    import numpy as np

    client = OpenAI(api_key=api_key)
    rng    = np.random.default_rng(42)

    gpt4_answers: list[int] = []
    where_correct: list[int] = []

    for q, ra, rb in tqdm(zip(questions, responses_a, responses_b), total=len(questions),
                          desc="GPT-4 eval"):
        # Randomly decide which target demographic to ask about to avoid position bias.
        # ra = response generated under label_a activation patch
        # rb = response generated under label_b activation patch
        if rng.integers(2) == 0:
            demographic = f"a user who is highly {label_a}"
            correct     = 1   # ra should match better
        else:
            demographic = f"a user who is highly {label_b}"
            correct     = 2   # rb should match better

        prompt_text = GPT4_TEMPLATE.format(
            demographic=demographic, prompt=q, r1=ra, r2=rb
        )

        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user",   "content": prompt_text},
                    ],
                    temperature=0.0,
                )
                raw = (resp.choices[0].message.content or "").strip()
                print(f"\n  [GPT-4 raw] {repr(raw[:200])}")
                # Try plain digit first
                if raw in ("1", "2"):
                    answer = int(raw)
                    break
                # Try JSON parse
                raw_clean = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                if raw_clean:
                    try:
                        parsed = json.loads(raw_clean)
                        answer = int(str(parsed["answer"]).strip())
                        break
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
                # Fallback: find first digit 1 or 2 anywhere in response
                import re as _re
                m = _re.search(r'\b([12])\b', raw)
                if m:
                    answer = int(m.group(1))
                    break
                raise ValueError(f"Cannot parse answer from: {repr(raw[:200])}")
            except Exception as e:
                print(f"\n  GPT-4 attempt {attempt+1} failed: {type(e).__name__}: {e}")
                time.sleep(2 ** attempt)
                answer = -1

        gpt4_answers.append(answer)
        where_correct.append(correct)

    return gpt4_answers, where_correct


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute", required=True, choices=list(ALL_ATTRIBUTES.keys()))
    parser.add_argument("--contrast", nargs=2, metavar=("SUB_A", "SUB_B"),
                        help="Two subcategories to contrast, e.g. --contrast expert novice")
    parser.add_argument("--N", type=float, default=15.0, help="Intervention strength (applied to unit-normalised delta)")
    parser.add_argument("--layers", nargs=2, type=int, default=[19, 29],
                        metavar=("FROM", "TO"), help="Layer range [from, to)")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-gpt4", action="store_true",
                        help="Skip GPT-4 judged eval; only compute CoT metrics")
    parser.add_argument("--probe-type", choices=["control", "reading"], default="control")
    args = parser.parse_args()

    attr      = ALL_ATTRIBUTES[args.attribute]
    subcats   = attr["subcategories"]
    contrast  = args.contrast or attr.get("transition_pairs", [None])[0]
    if contrast is None:
        parser.error("Specify --contrast SUB_A SUB_B")

    sub_a, sub_b = contrast
    assert sub_a in subcats, f"{sub_a} not in {subcats}"
    assert sub_b in subcats, f"{sub_b} not in {subcats}"

    # Load questions
    q_file = Path(attr.get("cot_questions_file", f"data/causality_test_questions/{args.attribute}.txt"))
    if not q_file.exists():
        raise FileNotFoundError(f"Questions file not found: {q_file}")
    questions = q_file.read_text().splitlines()
    questions = [q for q in questions if q.strip()]

    # Load model
    print(f"Loading {MODEL_ID} on {args.device}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=args.device
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # Select probe checkpoint directory
    if args.probe_type == "control":
        probes = load_control_probes(args.attribute, PROBE_DIR / "controlling_probe")
    else:
        probes = load_control_probes(args.attribute, PROBE_DIR)
    print(f"  Loaded {len(probes)} probe checkpoints for probe_type={args.probe_type}")

    # Build layer names for baukit hook
    from_l, to_l = args.layers
    layer_names = []
    for name, _ in model.named_modules():
        if "model.layers." not in name:
            continue
        suffix = name[name.rfind("model.layers.") + len("model.layers."):]
        # Only top-level layer containers: suffix is purely digits
        if not suffix.isdigit():
            continue
        layer_num = int(suffix)
        if from_l <= layer_num < to_l:
            layer_names.append(name)
    print(f"  Hooking {len(layer_names)} layers: {layer_names[:3]}...")

    n_classes = len(subcats)

    def make_contrast(src: str, tgt: str) -> torch.Tensor:
        """One-hot contrast vector: +1 at target index, -1 at source index."""
        s_idx = subcats.index(src)
        t_idx = subcats.index(tgt)
        t = torch.zeros(1, n_classes)
        t[0, t_idx] =  1.0
        t[0, s_idx] = -1.0
        return t

    # Generate responses
    print(f"\n[{args.attribute}] Generating responses: unintervened…")
    responses_base = collect_responses(
        model, tokenizer, questions, null_edit, [], args.batch_size, args.device
    )

    print(f"\n[{args.attribute}] Generating responses: intervened → {sub_a} (from {sub_b})…")
    edit_a = make_edit_function(probes, make_contrast(sub_b, sub_a), args.N, from_l, to_l)
    responses_a = collect_responses(
        model, tokenizer, questions, edit_a, layer_names, args.batch_size, args.device
    )

    print(f"\n[{args.attribute}] Generating responses: intervened → {sub_b} (from {sub_a})…")
    edit_b = make_edit_function(probes, make_contrast(sub_a, sub_b), args.N, from_l, to_l)
    responses_b = collect_responses(
        model, tokenizer, questions, edit_b, layer_names, args.batch_size, args.device
    )

    # Compute CoT complexity metrics
    print("\nComputing CoT complexity metrics…")
    metrics_base = [measure_cot_complexity(r) for r in responses_base]
    metrics_a    = [measure_cot_complexity(r) for r in responses_a]
    metrics_b    = [measure_cot_complexity(r) for r in responses_b]

    contrast_key = f"{sub_a}/{sub_b}"
    cot_output = {
        "attribute":    args.attribute,
        "contrast":     contrast_key,
        "probe_type":   args.probe_type,
        "N":            args.N,
        "layers":       args.layers,
        "questions":    questions,
        "responses": {
            "base":  responses_base,
            sub_a:   responses_a,
            sub_b:   responses_b,
        },
        "cot_metrics": {
            "base":  metrics_base,
            sub_a:   metrics_a,
            sub_b:   metrics_b,
        },
    }

    # Save responses and metrics
    cot_path = COT_DIR / f"{args.attribute}_{sub_a}_vs_{sub_b}_cot.json"
    with open(cot_path, "w") as f:
        json.dump(cot_output, f, indent=2)
    print(f"CoT results saved → {cot_path}")

    # GPT-4 intervention success rate
    success_rate = None
    if not args.skip_gpt4:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not set — skipping GPT-4 eval.")
        else:
            gpt4_answers, where_correct = evaluate_with_gpt4(
                questions, responses_a, responses_b, sub_a, sub_b, api_key
            )
            import numpy as np
            arr_g = np.array(gpt4_answers)
            arr_c = np.array(where_correct)
            valid = arr_g != -1
            success_rate = (arr_g[valid] == arr_c[valid]).mean()
            print(f"\nGPT-4 intervention success rate ({sub_a} vs {sub_b}): {success_rate:.3f}")

            cot_output["gpt4_answers"]    = gpt4_answers
            cot_output["where_correct"]   = where_correct
            cot_output["success_rate"]    = success_rate
            with open(cot_path, "w") as f:
                json.dump(cot_output, f, indent=2)

    # Save to results for evaluate.py / plot_all_intervention_success()
    result_path = RESULT_DIR / f"{args.attribute}_intervention.json"
    existing = {}
    if result_path.exists():
        with open(result_path) as f:
            existing = json.load(f)

    existing.setdefault("attribute", args.attribute)
    existing.setdefault("subcategory_pairs", [])
    existing.setdefault("probe_types", {"control": {}, "reading": {}})

    if [sub_a, sub_b] not in existing["subcategory_pairs"]:
        existing["subcategory_pairs"].append([sub_a, sub_b])
    if success_rate is not None:
        existing["probe_types"][args.probe_type][contrast_key] = success_rate

    with open(result_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Intervention results saved → {result_path}")


if __name__ == "__main__":
    main()
