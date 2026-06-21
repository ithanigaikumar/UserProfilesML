"""
Train one-vs-rest linear logistic probes on residual-stream activations.

Matches the setup in Chen et al. §4.2:
  - Linear logistic probe:  p_θ(x) = σ(<x, θ>)
  - One-vs-rest per subcategory
  - L2 regularisation
  - 80/20 train-val split
  - Trained separately for each layer

Output layout:
    data/probes/{attribute}/layer_{l:02d}_vs_{subcategory}.pt
        {"state_dict": ..., "val_acc": float, "layer": int, "subcategory": str}

    data/results/{attribute}_accuracies.json
        { "layer_accuracies": {layer: {"multiclass_val_acc": float,
                                       "per_class": {sub: binary_val_acc}} },
          "best_layer": int }

Usage:
    python src/train_probes.py --attribute age
    python src/train_probes.py --all
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ACT_DIR,
    ALL_ATTRIBUTES,
    HIDDEN_DIM,
    L2_WEIGHT_DECAY,
    NUM_LAYERS,
    PROBE_BATCH,
    PROBE_DIR,
    PROBE_EPOCHS,
    PROBE_LR,
    RESULT_DIR,
    SEED,
    TRAIN_SPLIT,
)

torch.manual_seed(SEED)
random.seed(SEED)


# ── Minimal linear probe (matches paper: single linear layer + cross-entropy) ──

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_activations(attr_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return X [N, num_layers+1, hidden_dim] and y [N] from disk."""
    path = ACT_DIR / f"{attr_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run extract_activations.py first.")
    data = torch.load(path, map_location="cpu")
    return data["X"].float(), data["y"].long()


def train_val_split(
    X: torch.Tensor, y: torch.Tensor, split: float = TRAIN_SPLIT
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = X.shape[0]
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
    cut = int(N * split)
    tr, val = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[val], y[val]


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_probe(
    X_tr: torch.Tensor,   # [N_tr, hidden_dim]
    y_tr: torch.Tensor,   # [N_tr]
    X_val: torch.Tensor,  # [N_val, hidden_dim]
    y_val: torch.Tensor,  # [N_val]
    n_classes: int,
    device: torch.device,
) -> tuple[LinearProbe, float]:
    """Train a single multi-class linear probe. Returns (probe, val_accuracy)."""
    probe = LinearProbe(X_tr.shape[-1], n_classes).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=L2_WEIGHT_DECAY)

    ds     = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=PROBE_BATCH, shuffle=True)

    best_val_acc = 0.0
    best_state   = None

    for _ in range(PROBE_EPOCHS):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(probe(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            logits = probe(X_val.to(device))
            preds  = logits.argmax(dim=-1).cpu()
        val_acc = (preds == y_val).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    return probe, best_val_acc


# ── Per-subcategory binary probe (one-vs-rest) ─────────────────────────────────

def train_binary_probe(
    X_tr: torch.Tensor,
    y_tr_bin: torch.Tensor,   # 0 / 1
    X_val: torch.Tensor,
    y_val_bin: torch.Tensor,
    device: torch.device,
) -> tuple[LinearProbe, float]:
    return train_one_probe(X_tr, y_tr_bin, X_val, y_val_bin, n_classes=2, device=device)


# ── Main per-attribute training ────────────────────────────────────────────────

def train_for_attribute(
    attr_name: str,
    device: torch.device,
    overwrite: bool = False,
) -> None:
    attr         = ALL_ATTRIBUTES[attr_name]
    subcategories = attr["subcategories"]
    n_classes     = len(subcategories)
    probe_subdir  = PROBE_DIR / attr_name
    probe_subdir.mkdir(parents=True, exist_ok=True)
    result_path   = RESULT_DIR / f"{attr_name}_accuracies.json"

    if result_path.exists() and not overwrite:
        print(f"[{attr_name}] Results exist at {result_path}, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Training probes for attribute: {attr_name}")
    print(f"{'='*60}")

    X_all, y_all = load_activations(attr_name)
    # X_all: [N, num_layers+1, hidden_dim]
    X_tr_all, y_tr_all, X_val_all, y_val_all = train_val_split(X_all, y_all)
    n_layer_slots = X_all.shape[1]   # num_layers + 1 (embedding + transformer blocks)

    layer_accuracies: dict[int, dict] = {}

    for layer in range(n_layer_slots):
        X_tr  = X_tr_all[:, layer, :]    # [N_tr, hidden_dim]
        X_val = X_val_all[:, layer, :]   # [N_val, hidden_dim]

        # ── Multi-class probe ──────────────────────────────────────────────
        probe_mc, mc_val_acc = train_one_probe(
            X_tr, y_tr_all, X_val, y_val_all, n_classes, device
        )
        ckpt_path = probe_subdir / f"layer_{layer:02d}_multiclass.pt"
        torch.save(
            {"state_dict": probe_mc.state_dict(), "val_acc": mc_val_acc,
             "layer": layer, "n_classes": n_classes},
            ckpt_path,
        )

        # ── One-vs-rest binary probes ──────────────────────────────────────
        per_class: dict[str, float] = {}
        for cls_idx, sub in enumerate(subcategories):
            y_tr_bin  = (y_tr_all  == cls_idx).long()
            y_val_bin = (y_val_all == cls_idx).long()
            _, bin_acc = train_binary_probe(X_tr, y_tr_bin, X_val, y_val_bin, device)
            per_class[sub] = bin_acc
            ckpt_bin = probe_subdir / f"layer_{layer:02d}_vs_{sub}.pt"
            torch.save(
                {"state_dict": None,   # lightweight: save only the accuracy
                 "val_acc": bin_acc, "layer": layer, "subcategory": sub},
                ckpt_bin,
            )

        layer_accuracies[layer] = {
            "multiclass_val_acc": mc_val_acc,
            "per_class": per_class,
        }
        print(
            f"  Layer {layer:2d} | multiclass acc = {mc_val_acc:.3f} | "
            + " | ".join(f"{s}: {a:.3f}" for s, a in per_class.items())
        )

    best_layer = max(layer_accuracies, key=lambda l: layer_accuracies[l]["multiclass_val_acc"])
    result = {"layer_accuracies": layer_accuracies, "best_layer": best_layer}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{attr_name}] Best layer: {best_layer}  "
          f"(acc = {layer_accuracies[best_layer]['multiclass_val_acc']:.3f})")
    print(f"Results saved → {result_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute", choices=list(ALL_ATTRIBUTES.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    targets = list(ALL_ATTRIBUTES.keys()) if args.all else [args.attribute]
    if not targets or targets == [None]:
        parser.error("Specify --attribute or --all")

    device = torch.device(args.device)
    for attr_name in targets:
        train_for_attribute(attr_name, device, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
