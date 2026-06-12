"""
Evaluate trained probes and reproduce Figure 1 from Chen et al.

Reads data/results/{attribute}_accuracies.json produced by train_probes.py
and plots validation accuracy across layers for all four attributes.

Usage:
    python src/evaluate.py                    # plot all attributes
    python src/evaluate.py --attribute age    # single attribute
    python src/evaluate.py --save             # save figure to data/results/figure1.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch

from config import ALL_ATTRIBUTES, ATTRIBUTES, ACT_DIR, COT_DIR, RESULT_DIR

# Paper-style colours (one per attribute)
COLOURS = {
    "age":             "#1f77b4",
    "gender":          "#ff7f0e",
    "education":       "#2ca02c",
    "socioeconomic":   "#d62728",
    # Dynamic attributes
    "knowledge_level": "#9467bd",
    "confusion":       "#8c564b",
    "emotional_state": "#e377c2",
    "user_intent":     "#17becf",
    "formality":       "#bcbd22",
}

LABELS = {
    "age":             "Age",
    "gender":          "Gender",
    "education":       "Education",
    "socioeconomic":   "Socio-economic status",
    "knowledge_level": "Knowledge Level",
    "confusion":       "Confusion",
    "emotional_state": "Emotional State",
    "user_intent":     "User Intent",
    "formality":       "Formality",
}


def load_results(attr_name: str) -> dict:
    path = RESULT_DIR / f"{attr_name}_accuracies.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run train_probes.py first.")
    with open(path) as f:
        return json.load(f)


def print_summary(attr_name: str, results: dict) -> None:
    la = results["layer_accuracies"]
    best = results["best_layer"]
    print(f"\n── {LABELS[attr_name]} ──────────────────────────────")
    print(f"  Best layer : {best}  "
          f"(val acc = {la[str(best)]['multiclass_val_acc']:.3f})")
    print(f"  {'Layer':>6}  {'Multiclass acc':>15}")
    for layer_str, info in sorted(la.items(), key=lambda kv: int(kv[0])):
        print(f"  {int(layer_str):>6}  {info['multiclass_val_acc']:>15.3f}")


def plot_figure1(
    targets: list[str],
    save: bool = False,
    output_path: Path | None = None,
) -> None:
    """Reproduce Figure 1: probe accuracy across layers."""
    fig, axes = plt.subplots(
        1, len(targets),
        figsize=(4 * len(targets), 4),
        sharey=False,
    )
    if len(targets) == 1:
        axes = [axes]

    for ax, attr_name in zip(axes, targets):
        results = load_results(attr_name)
        la = results["layer_accuracies"]
        layers = sorted(int(k) for k in la.keys())
        mc_accs = [la[str(l)]["multiclass_val_acc"] for l in layers]

        # Per-class lines (lighter)
        sub_names = list(la[str(layers[0])]["per_class"].keys())
        for sub in sub_names:
            per_class_accs = [la[str(l)]["per_class"][sub] for l in layers]
            ax.plot(
                layers, per_class_accs,
                color=COLOURS[attr_name], alpha=0.25, linewidth=1.0,
                label=f"_{sub}",   # leading _ hides from legend
            )

        # Multi-class line (bold)
        ax.plot(
            layers, mc_accs,
            color=COLOURS[attr_name], linewidth=2.5,
            label=LABELS[attr_name],
        )

        # Best layer marker
        best = int(results["best_layer"])
        ax.axvline(x=best, color=COLOURS[attr_name], linestyle="--", alpha=0.5)
        ax.annotate(
            f"L{best}\n{la[str(best)]['multiclass_val_acc']:.2f}",
            xy=(best, la[str(best)]["multiclass_val_acc"]),
            xytext=(3, -10), textcoords="offset points",
            fontsize=7, color=COLOURS[attr_name],
        )

        ax.set_title(LABELS[attr_name], fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Validation accuracy")
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    STATIC = {"age", "gender", "education", "socioeconomic"}
    is_replication = all(t in STATIC for t in targets)
    subtitle = (
        "Replication of Chen et al. Figure 1"
        if is_replication
        else "Dynamic user-state attributes — LLaMA-2-Chat-13B"
    )
    fig.suptitle(
        f"Reading probe accuracy across layers\n{subtitle}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    if save:
        name = "_".join(targets) if len(targets) > 1 else targets[0]
        out = output_path or (RESULT_DIR / f"figure1_{name}.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"Figure saved → {out}")

    plt.show()


def plot_combined(
    targets: list[str],
    save: bool = False,
    output_path: Path | None = None,
) -> None:
    """All four attributes on one axes for a compact overview."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for attr_name in targets:
        results = load_results(attr_name)
        la = results["layer_accuracies"]
        layers  = sorted(int(k) for k in la.keys())
        mc_accs = [la[str(l)]["multiclass_val_acc"] for l in layers]
        ax.plot(layers, mc_accs, color=COLOURS[attr_name],
                linewidth=2, label=LABELS[attr_name])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Validation accuracy (multiclass)")
    ax.set_title("Reading probe accuracy across layers — all attributes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        out = output_path or (RESULT_DIR / "figure1_combined.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"Figure saved → {out}")

    plt.show()


def load_intervention_results(attr_name: str) -> dict:
    """
    Load intervention success-rate results produced by run_intervention.py.

    Expected file: data/results/{attribute}_intervention.json
    Schema:
        {
          "attribute": str,
          "subcategory_pairs": [ [sub_a, sub_b], ... ],
          "probe_types": {
              "control": {"pair_key": float, ...},   # success rate 0-1
              "reading": {"pair_key": float, ...}
          },
          "reading_accuracies": { sub: float, ... }   # best-layer val acc
        }
    """
    path = RESULT_DIR / f"{attr_name}_intervention.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run run_intervention.py first."
        )
    with open(path) as f:
        return json.load(f)


def plot_intervention_success(
    attr_name: str,
    save: bool = False,
    output_path: Path | None = None,
) -> None:
    """
    Bar chart of GPT-4 judged intervention success rate for control vs
    reading probes across all subcategory contrasts (Table 2 replica).

    If the full intervention JSON is unavailable, falls back to the
    hard-coded paper values so the figure can still be rendered.
    """
    # ── Paper values (Table 2) used as fallback ────────────────────────────
    PAPER_VALUES: dict[str, dict] = {
        "age": {
            "pairs":   ["older adult\nvs adolescent"],
            "control": [0.90],
            "reading": [0.80],
        },
        "gender": {
            "pairs":   ["female\nvs male"],
            "control": [0.87],
            "reading": [0.77],
        },
        "education": {
            "pairs":   ["college+\nvs some schooling"],
            "control": [0.83],
            "reading": [0.73],
        },
        "socioeconomic": {
            "pairs":   ["high SES\nvs low SES"],
            "control": [0.85],
            "reading": [0.72],
        },
    }

    try:
        data    = load_intervention_results(attr_name)
        pairs   = [" vs\n".join(p) for p in data["subcategory_pairs"]]
        control = [data["probe_types"]["control"]["/".join(p)] for p in data["subcategory_pairs"]]
        reading = [data["probe_types"]["reading"]["/".join(p)] for p in data["subcategory_pairs"]]
    except FileNotFoundError:
        print(f"[{attr_name}] No intervention results found — using paper values.")
        pv      = PAPER_VALUES.get(attr_name, {})
        pairs   = pv.get("pairs", [])
        control = pv.get("control", [])
        reading = pv.get("reading", [])

    if not pairs:
        print(f"[{attr_name}] No data to plot.")
        return

    x     = np.arange(len(pairs))
    width = 0.35
    color = COLOURS.get(attr_name, "#333333")

    fig, ax = plt.subplots(figsize=(max(5, 2.5 * len(pairs)), 4))
    bars_c = ax.bar(x - width / 2, control, width, label="Control probe",
                    color=color, alpha=0.9)
    bars_r = ax.bar(x + width / 2, reading, width, label="Reading probe",
                    color=color, alpha=0.45, hatch="//")

    # Value labels on bars
    for bar in list(bars_c) + list(bars_r):
        ax.annotate(
            f"{bar.get_height():.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=9)
    ax.set_ylabel("GPT-4 judged success rate")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="Chance (0.5)")
    ax.set_title(f"Intervention success rate — {LABELS.get(attr_name, attr_name)}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        out = output_path or (RESULT_DIR / f"{attr_name}_intervention_success.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"Figure saved → {out}")

    plt.show()


def plot_all_intervention_success(
    save: bool = False,
    output_path: Path | None = None,
) -> None:
    """
    Single grouped bar chart with all attributes side by side.
    Static attributes fall back to paper values; dynamic attributes are loaded
    from result files if available.
    """
    PAPER_VALUES = {
        "age":           {"label": "Age",   "control": 0.90, "reading": 0.80},
        "gender":        {"label": "Gender","control": 0.87, "reading": 0.77},
        "education":     {"label": "Edu",   "control": 0.83, "reading": 0.73},
        "socioeconomic": {"label": "SES",   "control": 0.85, "reading": 0.72},
    }
    # Dynamic attributes have no paper fallback — only include if file exists
    DYNAMIC_LABELS = {
        "knowledge_level": "Know.",
        "confusion":       "Confuse.",
        "emotional_state": "Emotion",
        "user_intent":     "Intent",
        "formality":       "Formal",
    }

    attrs, ctrl_vals, read_vals, colors = [], [], [], []
    # Static (paper) attributes
    for attr_name, pv in PAPER_VALUES.items():
        try:
            data = load_intervention_results(attr_name)
            ctrl = float(np.mean(list(data["probe_types"]["control"].values())))
            read = float(np.mean(list(data["probe_types"]["reading"].values())))
        except FileNotFoundError:
            ctrl = pv["control"]
            read = pv["reading"]
        attrs.append(pv["label"])
        ctrl_vals.append(ctrl)
        read_vals.append(read)
        colors.append(COLOURS.get(attr_name, "#333"))
    # Dynamic attributes — only include if results exist
    for attr_name, label in DYNAMIC_LABELS.items():
        try:
            data = load_intervention_results(attr_name)
            ctrl_items = data["probe_types"]["control"]
            read_items = data["probe_types"]["reading"]
            if not ctrl_items:
                continue
            ctrl = float(np.mean(list(ctrl_items.values())))
            read = float(np.mean(list(read_items.values()))) if read_items else None
            attrs.append(label)
            ctrl_vals.append(ctrl)
            read_vals.append(read)
            colors.append(COLOURS.get(attr_name, "#333"))
        except FileNotFoundError:
            pass

    x     = np.arange(len(attrs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(attrs) * 1.1), 4))
    for i, (c, r, col) in enumerate(zip(ctrl_vals, read_vals, colors)):
        ax.bar(x[i] - width / 2, c, width, color=col, alpha=0.9,
               label="Control" if i == 0 else "_")
        if r is not None:
            ax.bar(x[i] + width / 2, r, width, color=col, alpha=0.45, hatch="//",
                   label="Reading" if i == 0 else "_")

    ax.set_xticks(x)
    ax.set_xticklabels(attrs)
    ax.set_ylabel("GPT-4 judged success rate")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Causality intervention success — control vs reading probes (Table 2)")
    ax.legend(["Control probe", "Reading probe"], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        out = output_path or (RESULT_DIR / "table2_intervention_success.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"Figure saved → {out}")

    plt.show()


def plot_per_question_success(
    attr_name: str,
    responses_dict: dict[str, list[str]],
    questions: list[str],
    gpt4_results: list[int],
    where_correct: list[int],
    save: bool = False,
    output_path: Path | None = None,
) -> None:
    """
    Two-panel figure for a single attribute:
      Left  — per-question binary correct/incorrect heatmap
      Right — rolling accuracy across questions

    Parameters
    ----------
    responses_dict  : keys are subcategory names (e.g. "some_schooling", "college_more")
                      values are lists of model responses, one per question
    questions       : list of question strings
    gpt4_results    : GPT-4 predicted answer index (1 or 2) per question
    where_correct   : ground-truth correct answer index (1 or 2) per question
    """
    gpt4_results  = np.array(gpt4_results)
    where_correct = np.array(where_correct)
    correct       = (gpt4_results == where_correct).astype(int)
    rolling_acc   = np.cumsum(correct) / (np.arange(len(correct)) + 1)
    overall_acc   = correct.mean()

    fig, (ax_heat, ax_roll) = plt.subplots(1, 2, figsize=(14, max(3, len(questions) * 0.25 + 1)))

    # ── Left: per-question heatmap ─────────────────────────────────────────
    ax_heat.imshow(
        correct.reshape(-1, 1),
        aspect="auto",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax_heat.set_yticks(range(len(questions)))
    ax_heat.set_yticklabels(
        [f"Q{i+1}: {q[:45]}…" if len(q) > 45 else f"Q{i+1}: {q}" for i, q in enumerate(questions)],
        fontsize=7,
    )
    ax_heat.set_xticks([])
    ax_heat.set_title(f"Per-question result\n(green=correct, red=wrong)", fontsize=9)

    # ── Right: rolling accuracy ────────────────────────────────────────────
    ax_roll.plot(range(1, len(correct) + 1), rolling_acc,
                 color=COLOURS.get(attr_name, "#333"), linewidth=2)
    ax_roll.axhline(overall_acc, color="black", linewidth=1, linestyle="--",
                    label=f"Overall: {overall_acc:.2f}")
    ax_roll.axhline(0.5, color="grey", linewidth=0.8, linestyle=":",
                    label="Chance (0.5)")
    ax_roll.set_xlabel("Question index")
    ax_roll.set_ylabel("Cumulative accuracy")
    ax_roll.set_ylim(0, 1.05)
    ax_roll.set_title("Rolling GPT-4 judged success rate")
    ax_roll.legend(fontsize=9)
    ax_roll.grid(alpha=0.3)

    fig.suptitle(
        f"Intervention evaluation — {LABELS.get(attr_name, attr_name)}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    if save:
        out = output_path or (RESULT_DIR / f"{attr_name}_per_question.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"Figure saved → {out}")

    plt.show()


def plot_turn_level_accuracy(
    attr_name: str,
    layer: int | None = None,
    save: bool = False,
    output_path: Path | None = None,
) -> None:
    """
    Plot multiclass probe accuracy as a function of conversation turn.

    Requires data/activations/{attribute}_turn_level.pt (from extract_activations.py
    --turn-level) and data/results/{attribute}_accuracies.json (from train_probes.py).

    Answers: does the representation become more linearly separable as the
    conversation develops?
    """
    act_path = ACT_DIR / f"{attr_name}_turn_level.pt"
    if not act_path.exists():
        raise FileNotFoundError(
            f"{act_path} not found. Run: extract_activations.py --attribute {attr_name} --turn-level"
        )

    results   = load_results(attr_name)
    best_l    = layer if layer is not None else int(results["best_layer"])
    attr      = ALL_ATTRIBUTES[attr_name]
    subcats   = attr["subcategories"]

    data   = torch.load(act_path, map_location="cpu")
    X_all  = data["X"].float()   # [N, max_turns, num_layers+1, hidden_dim]
    y_all  = data["y"].long()    # [N]
    mask   = data["mask"]        # [N, max_turns]
    meta   = data["meta"]
    max_turns = X_all.shape[1]

    # For each turn, compute accuracy using the best-layer probe weights
    # We re-use the saved multiclass probe from train_probes.py
    from train_probes import LinearProbe, train_val_split
    import torch.nn.functional as F

    turn_accs: list[float] = []
    turn_ns:   list[int]   = []

    for t in range(max_turns):
        present = mask[:, t]          # [N] bool
        n_present = present.sum().item()
        if n_present < 10:
            break
        X_t = X_all[present, t, best_l, :]   # [n_present, hidden_dim]
        y_t = y_all[present]

        # Quick k-NN accuracy proxy (no retraining per turn)
        # Compute cosine similarities within the slice and do 1-NN classification
        X_norm = F.normalize(X_t, dim=-1)
        sim    = X_norm @ X_norm.T   # [n, n]
        sim.fill_diagonal_(-1.0)
        nn_idx  = sim.argmax(dim=-1)
        nn_pred = y_t[nn_idx]
        acc     = (nn_pred == y_t).float().mean().item()
        turn_accs.append(acc)
        turn_ns.append(n_present)

    fig, ax = plt.subplots(figsize=(7, 4))
    color = COLOURS.get(attr_name, "#333")
    ax.plot(range(1, len(turn_accs) + 1), turn_accs,
            marker="o", color=color, linewidth=2, markersize=6)
    for t, (acc, n) in enumerate(zip(turn_accs, turn_ns)):
        ax.annotate(f"n={n}", (t + 1, acc), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color="grey")

    ax.set_xlabel("Conversation turn")
    ax.set_ylabel("1-NN accuracy (layer {best_l})")
    ax.set_title(
        f"Does the {LABELS.get(attr_name, attr_name)} representation build turn-by-turn?\n"
        f"(1-NN accuracy @ layer {best_l})"
    )
    ax.set_xticks(range(1, len(turn_accs) + 1))
    ax.set_ylim(0, 1.05)
    ax.axhline(1 / len(subcats), color="grey", linestyle="--", linewidth=0.8,
               label=f"Chance ({1/len(subcats):.2f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        out = output_path or (RESULT_DIR / f"{attr_name}_turn_level.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"Figure saved \u2192 {out}")
    plt.show()


def plot_cot_complexity(
    attr_name: str,
    sub_a: str,
    sub_b: str,
    save: bool = False,
    output_path: Path | None = None,
) -> None:
    """
    Box-and-whisker plot comparing CoT complexity metrics between two
    intervention conditions.

    Requires data/cot_responses/{attribute}_{sub_a}_vs_{sub_b}_cot.json
    produced by run_intervention.py.

    This is the core causal figure: if {sub_a} and {sub_b} interventions
    produce systematically different complexity profiles, the model's internal
    epistemic state representation is *causally* shaping its reasoning.
    """
    cot_path = COT_DIR / f"{attr_name}_{sub_a}_vs_{sub_b}_cot.json"
    if not cot_path.exists():
        raise FileNotFoundError(
            f"{cot_path} not found. Run run_intervention.py first."
        )
    with open(cot_path) as f:
        data = json.load(f)

    attr    = ALL_ATTRIBUTES[attr_name]
    metrics = attr.get("cot_metrics",
                       ["explanation_depth", "technical_density", "hedging_frequency",
                        "flesch_kincaid_grade", "response_length"])
    metrics = [m for m in metrics if m in data["cot_metrics"].get(sub_a, [{}])[0]]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    color_a = COLOURS.get(attr_name, "#1f77b4")
    color_b = "#aec7e8"

    for ax, metric in zip(axes, metrics):
        vals_a   = [d[metric] for d in data["cot_metrics"][sub_a]   if metric in d]
        vals_b   = [d[metric] for d in data["cot_metrics"][sub_b]   if metric in d]
        vals_base = [d[metric] for d in data["cot_metrics"]["base"] if metric in d]

        bp = ax.boxplot(
            [vals_base, vals_a, vals_b],
            labels=["Base", sub_a.replace("_", "\n"), sub_b.replace("_", "\n")],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        colors = ["#dddddd", color_a, color_b]
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        ax.set_title(metric.replace("_", "\n"), fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    success = data.get("success_rate")
    title   = (
        f"CoT complexity: {LABELS.get(attr_name, attr_name)} — {sub_a} vs {sub_b}"
        + (f"  |  GPT-4 success: {success:.2f}" if success is not None else "")
    )
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save:
        out = output_path or (RESULT_DIR / f"{attr_name}_{sub_a}_vs_{sub_b}_cot_complexity.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"Figure saved \u2192 {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute", choices=list(ALL_ATTRIBUTES.keys()))
    parser.add_argument("--all", action="store_true", default=True,
                        help="Plot all available attributes (default)")
    parser.add_argument("--combined", action="store_true",
                        help="Overlay all attributes on one plot instead of subplots")
    parser.add_argument("--intervention", action="store_true",
                        help="Plot intervention success rate (Table 2)")
    parser.add_argument("--turn-level", action="store_true",
                        help="Plot probe accuracy across conversation turns")
    parser.add_argument("--cot", nargs=2, metavar=("SUB_A", "SUB_B"),
                        help="Plot CoT complexity comparison for this contrast")
    parser.add_argument("--save", action="store_true", help="Save figure to PDF")
    parser.add_argument("--summary", action="store_true", help="Print text summary")
    args = parser.parse_args()

    targets = (
        [args.attribute]
        if args.attribute
        else [a for a in ALL_ATTRIBUTES if (RESULT_DIR / f"{a}_accuracies.json").exists()]
    )
    if not targets and not args.intervention and not args.cot:
        print("No results found. Run train_probes.py first.")
        return

    if args.summary:
        for attr_name in targets:
            print_summary(attr_name, load_results(attr_name))

    if args.cot:
        # CoT complexity boxplots — requires run_intervention.py output
        if not args.attribute:
            parser.error("--cot requires --attribute")
        plot_cot_complexity(args.attribute, args.cot[0], args.cot[1], save=args.save)

    elif args.turn_level:
        # Turn-level probe accuracy — requires --turn-level extraction
        for attr_name in targets:
            plot_turn_level_accuracy(attr_name, save=args.save)

    elif args.intervention:
        plot_all_intervention_success(save=args.save)
        if args.attribute:
            plot_intervention_success(args.attribute, save=args.save)

    elif args.combined:
        plot_combined(targets, save=args.save)

    else:
        plot_figure1(targets, save=args.save)


if __name__ == "__main__":
    main()
