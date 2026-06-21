"""
Probe-score intervention evaluation — runs entirely locally, no LLM needed.

Uses saved activation tensors (data/activations/{attr}.pt) to directly measure
whether the probe-weight steering direction causally shifts internal representations.

Experiment:
  For each class pair (sub_a, sub_b) at a target layer:
  1. Take held-out samples labelled as sub_a.
  2. Apply the sub_b steering vector: h' = h + N * W[b,:] / ||W[b,:]||
  3. Run the layer probe on h and h'.
  4. Report: how many samples flip from sub_a → sub_b?
  Repeat for the reverse direction.

  Control: apply a random unit vector instead and report the flip rate.

  A genuine causal direction should flip >> control rate.

Usage:
    python src/probe_score_eval.py --attribute confusion --contrast clear confused
    python src/probe_score_eval.py --attribute knowledge_level --contrast expert novice
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import ACT_DIR, ALL_ATTRIBUTES, HIDDEN_DIM, PROBE_DIR, RESULT_DIR
from probes import LinearProbeClassification


def load_control_probes(attr_name, probe_dir):
    """Load per-layer probes from probe_dir/{attr_name}/layer_*_multiclass.pt"""
    import re
    attr       = ALL_ATTRIBUTES[attr_name]
    n_classes  = len(attr["subcategories"])
    probe_subdir = probe_dir / attr_name
    probes = {}
    for ckpt in sorted(probe_subdir.glob("layer_*_multiclass.pt")):
        layer = int(re.search(r"layer_(\d+)_multiclass", ckpt.stem).group(1))
        data  = torch.load(ckpt, map_location="cpu")
        state = data["state_dict"]
        if "linear.weight" in state and "proj.0.weight" not in state:
            state = {"proj.0.weight": state["linear.weight"],
                     "proj.0.bias":   state["linear.bias"]}
        # Infer hidden dim from checkpoint rather than config constant
        hidden_dim = state["proj.0.weight"].shape[1]
        probe = LinearProbeClassification("cpu", n_classes, hidden_dim, logistic=True)
        probe.load_state_dict(state)
        probe.eval()
        probes[layer] = probe
    return probes


# ── helpers ───────────────────────────────────────────────────────────────────

def load_activations(attr_name: str):
    path = ACT_DIR / f"{attr_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Activations not found: {path}")
    data = torch.load(path, map_location="cpu")
    X = data["X"].float()       # [N, L, D]  L=41 (embedding + 40 layers)
    y = data["y"]
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    return X, y.long()


def probe_predict(probe: LinearProbeClassification, h: torch.Tensor) -> torch.Tensor:
    """Return predicted class index for each sample. h: [N, D]"""
    with torch.no_grad():
        scores = probe.proj(h)          # [N, C]
    return scores.argmax(dim=-1)       # [N]


def probe_confidence(probe: LinearProbeClassification, h: torch.Tensor, class_idx: int) -> torch.Tensor:
    """Return P(class_idx) for each sample. h: [N, D]"""
    with torch.no_grad():
        scores = probe.proj(h)          # [N, C]
    return scores[:, class_idx]        # [N]


def steer(h: torch.Tensor, direction: torch.Tensor, N: float) -> torch.Tensor:
    """Add N units in unit-norm direction to activations. h: [N_samples, D]"""
    d = direction / (direction.norm() + 1e-8)
    return h + d.unsqueeze(0) * N


# ── main ──────────────────────────────────────────────────────────────────────

def run_eval(attr_name: str, sub_a: str, sub_b: str, N: float, target_layer: int, test_frac: float):
    attr    = ALL_ATTRIBUTES[attr_name]
    subcats = attr["subcategories"]
    idx_a   = subcats.index(sub_a)
    idx_b   = subcats.index(sub_b)

    print(f"\n{'='*60}")
    print(f"Attribute : {attr_name}")
    print(f"Contrast  : {sub_a} (class {idx_a})  vs  {sub_b} (class {idx_b})")
    print(f"Layer     : {target_layer}   N = {N}")
    print(f"{'='*60}")

    # ── load activations ──────────────────────────────────────────────────────
    X, y = load_activations(attr_name)
    n_total = len(y)
    layer_slot = target_layer + 1          # slot 0 = embedding, slot i = layer i-1 output
    H = X[:, layer_slot, :]               # [N, D]

    # ── load probes ───────────────────────────────────────────────────────────
    probes = load_control_probes(attr_name, PROBE_DIR)
    if target_layer not in probes:
        raise ValueError(f"No probe found for layer {target_layer}. Available: {sorted(probes.keys())}")
    probe = probes[target_layer]
    probe.eval()

    # ── held-out test split (last test_frac of each class) ───────────────────
    mask_a = (y == idx_a)
    mask_b = (y == idx_b)
    n_test = max(10, int(n_total * test_frac))

    idx_a_all = torch.where(mask_a)[0]
    idx_b_all = torch.where(mask_b)[0]
    test_a = idx_a_all[-len(idx_a_all) // 5:]   # last 20% of class-a samples
    test_b = idx_b_all[-len(idx_b_all) // 5:]   # last 20% of class-b samples

    H_a = H[test_a]    # [Na, D] — "clear" activations
    H_b = H[test_b]    # [Nb, D] — "confused" activations

    print(f"\nTest samples: {sub_a}={len(H_a)}, {sub_b}={len(H_b)}")

    # ── steering directions ───────────────────────────────────────────────────
    W = probe.proj[0].weight.detach().float()   # [C, D]
    dir_a = W[idx_a] / (W[idx_a].norm() + 1e-8)
    dir_b = W[idx_b] / (W[idx_b].norm() + 1e-8)

    # Random control directions (different seeds for a and b)
    rng_a = torch.Generator(); rng_a.manual_seed(42 + idx_a)
    rng_b = torch.Generator(); rng_b.manual_seed(42 + idx_b)
    rand_a = torch.randn(W.shape[1], generator=rng_a)
    rand_b = torch.randn(W.shape[1], generator=rng_b)
    rand_a = rand_a / rand_a.norm(); rand_b = rand_b / rand_b.norm()

    results = {}

    for src_name, src_idx, H_src, target_name, target_idx, probe_dir, rand_dir in [
        (sub_a, idx_a, H_a, sub_b, idx_b, dir_b, rand_b),
        (sub_b, idx_b, H_b, sub_a, idx_a, dir_a, rand_a),
    ]:
        n = len(H_src)

        # Baseline classification
        pred_base   = probe_predict(probe, H_src)
        acc_base    = (pred_base == src_idx).float().mean().item()

        # Probe-direction steering
        H_steered   = steer(H_src, probe_dir, N)
        pred_steered = probe_predict(probe, H_steered)
        flip_probe  = (pred_steered == target_idx).float().mean().item()
        conf_before = probe_confidence(probe, H_src, target_idx).mean().item()
        conf_after  = probe_confidence(probe, H_steered, target_idx).mean().item()

        # Random-direction control
        H_rand      = steer(H_src, rand_dir, N)
        pred_rand   = probe_predict(probe, H_rand)
        flip_rand   = (pred_rand == target_idx).float().mean().item()

        print(f"\n  Steering {src_name} → {target_name} (n={n}):")
        print(f"    Baseline accuracy on {src_name}      : {acc_base:.1%}")
        print(f"    P({target_name}) before steering     : {conf_before:.3f}")
        print(f"    P({target_name}) after  steering     : {conf_after:.3f}   Δ={conf_after-conf_before:+.3f}")
        print(f"    Flip rate  (probe direction)          : {flip_probe:.1%}")
        print(f"    Flip rate  (random control)           : {flip_rand:.1%}")
        print(f"    Lift over control                     : {flip_probe - flip_rand:+.1%}")

        results[f"{src_name}_to_{target_name}"] = {
            "n": n,
            "baseline_acc": acc_base,
            "conf_before": conf_before,
            "conf_after": conf_after,
            "delta_conf": conf_after - conf_before,
            "flip_probe": flip_probe,
            "flip_random": flip_rand,
            "lift": flip_probe - flip_rand,
        }

    # ── save ─────────────────────────────────────────────────────────────────
    out_path = RESULT_DIR / f"{attr_name}_{sub_a}_vs_{sub_b}_probe_score.json"
    out = {
        "attribute":    attr_name,
        "contrast":     [sub_a, sub_b],
        "target_layer": target_layer,
        "N":            N,
        "results":      results,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute", default=None,
                        help="Single attribute (omit to run all)")
    parser.add_argument("--contrast", nargs=2, metavar=("SUB_A", "SUB_B"),
                        help="Two subcategories to contrast")
    parser.add_argument("--N",    type=float, default=20.0,
                        help="Steering magnitude (same as run_intervention.py)")
    parser.add_argument("--layer", type=int,  default=30,
                        help="Layer to evaluate probe scores at (default: 30)")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    # Default contrast pairs per attribute
    defaults = {
        "confusion":       ("clear",      "confused"),
        "knowledge_level": ("expert",     "novice"),
        "emotional_state": ("enthusiastic","frustrated"),
        "user_intent":     ("learn",      "accomplish"),
        "formality":       ("formal",     "casual"),
    }

    attrs = [args.attribute] if args.attribute else list(defaults.keys())

    all_results = {}
    for attr in attrs:
        sub_a, sub_b = args.contrast if args.contrast else defaults[attr]
        try:
            r = run_eval(attr, sub_a, sub_b, args.N, args.layer, args.test_frac)
            all_results[attr] = r
        except Exception as e:
            print(f"  [SKIP] {attr}: {e}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("SUMMARY — probe-direction lift over random control")
    print("="*70)
    print(f"{'Attribute':<20} {'Direction':<30} {'Flip%':>6} {'Ctrl%':>6} {'Lift':>8}")
    print("-"*70)
    for attr, res in all_results.items():
        for key, v in res.items():
            src, _, tgt = key.partition("_to_")
            print(f"{attr:<20} {src+' → '+tgt:<30} {v['flip_probe']:>5.1%} "
                  f"{v['flip_random']:>5.1%} {v['lift']:>+7.1%}")


if __name__ == "__main__":
    main()
