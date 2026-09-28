"""Generate fidelity-threshold methodology figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis.generate_tables import table_coupled, table_fidelity


def build_figure(kr6_tau_b, subset_tau_b):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, w, txt, fc="#f3f4f6", ec="#374151", fs=9, bold=False, y=2.6, h=1.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#111827")

    def arrow(x0, x1, y, label, color):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=15,
                                     lw=1.6, color=color))
        ax.text((x0 + x1) / 2, y + 0.35, label, ha="center", va="bottom", fontsize=8.5,
                color=color, fontweight="bold")

    box(0.3, 3.0, "Decoupled\nproxy", bold=True)
    box(4.5, 3.2, "+ fixed-form\ninter-joint coupling\n(coupled proxy)")
    box(8.6, 3.1, "Configuration-dependent\n$M(q),C(q),g(q)$\n(rigid-body)", fc="#fee2e2")

    arrow(3.3, 4.5, 3.4, f"KR6: $\\tau_b$ = {kr6_tau_b:.3f}\n(ranking preserved)", "#059669")
    arrow(3.3, 8.6, 1.3, f"URDF subset: $\\tau_b$ = {subset_tau_b:.3f}\n(ranking reorders)", "#dc2626")

    ax.text(6.0, 0.4,
            "Fidelity sufficiency is purpose-dependent: a decoupled proxy suffices for\n"
            "torque-magnitude and trajectory-feasibility checks, but not for controller ranking.",
            ha="center", va="center", fontsize=9, style="italic", color="#374151")

    fig.suptitle("The fidelity threshold for controller selection", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def main():
    coupled_computed, _ = table_coupled()
    fidelity_computed, _ = table_fidelity()

    kr6_tau_b = coupled_computed["tau_by_eps"][0.09]         # representative eps_c
    subset_tau_b = fidelity_computed["subset_tau_b"]

    fig = build_figure(kr6_tau_b, subset_tau_b)
    out = ROOT / "figures" / "methodology_threshold.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {out}")

    # this same figure under a different filename -- same content, redrawn
    # with a serif font family for the journal's typesetting.
    with plt.rc_context({"font.family": "serif"}):
        fig_serif = build_figure(kr6_tau_b, subset_tau_b)
        out_serif = ROOT / "figures" / "methodology_threshold_serif.pdf"
        fig_serif.savefig(out_serif, bbox_inches="tight")
        plt.close(fig_serif)
    print(f"SAVED {out_serif}")

    print(f"  KR6 coupled-proxy tau_b (eps_c=0.09): {kr6_tau_b:.4f}")
    print(f"  URDF cross-fidelity subset tau_b:     {subset_tau_b:.4f}")


if __name__ == "__main__":
    main()
