"""Nature-subjournal-style matplotlib configuration for the multi-agent paper.

Usage::

    from _style import apply_style, COLORS, COHORT_ORDER, METHOD_ORDER
    apply_style()

Design principles:
- Sans-serif Helvetica/Arial-equivalent font
- Thin axes, outward ticks, no gratuitous gridlines
- Colorblind-friendly palette (Bang Wong / Okabe-Ito derived)
- Consistent method/cohort ordering and colors across all figures
- Save both PDF (vector for final submission) and PNG (high-DPI for preview)
"""

from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# --- Color palette (Nature subjournal style, Okabe-Ito colorblind-safe) ---
# Method triad chosen for maximum separation within Okabe-Ito:
#   blue / vermillion / green → distinguishable in greyscale and for the
#   ~8% of readers with red-green color deficiency.
COLORS = {
    # Method colors
    "canonical":    "#0072B2",  # Okabe-Ito sky blue
    "single_agent": "#D55E00",  # Okabe-Ito vermillion
    "multi_agent":  "#009E73",  # Okabe-Ito bluish green (the hero)
    # Class / outcome colors
    "tp": "#0072B2",
    "tn": "#999999",
    "fp": "#D55E00",            # vermillion (matches single-agent family)
    "fn": "#CC79A7",            # Okabe-Ito reddish purple
    # FN tier colors (T1 = most concerning → T4 = least)
    "tier_T1": "#882255",       # Tol muted dark red
    "tier_T2": "#CC6677",       # Tol muted rose
    "tier_T3": "#DDCC77",       # Tol muted sand
    "tier_T4": "#88CCEE",       # Tol muted light blue
    # Cohort colors — Tol "light" qualitative subset (lower saturation than
    # Tol muted; specifically designed for soft, pastel-leaning Nature-style
    # figures). Distinct from the Okabe-Ito method triad to avoid hue collision.
    "cohort_D":      "#77AADD",  # light blue   — development (training anchor)
    "cohort_V1":   "#44BB99",  # mint green   — V1, small external
    "cohort_V2": "#EE8866",  # coral — headline external validation cohort
    # Neutral
    "ink":        "#333333",     # dark grey (not pure black — softer on print)
    "muted":      "#999999",     # mid grey for reference lines
    "light_grid": "#E5E5E5",     # very light grid
    "highlight":  "#D55E00",     # vermillion for callouts
    # SHAP-style continuous endpoints (also used by SHAP beeswarm / CP3 bars)
    "shap_low":   "#1E88E5",     # vivid blue
    "shap_high":  "#E91E63",     # pink-red
}


# --- Continuous colormaps ---
# SHAP-style "red_blue" continuous ramp: blue → magenta → red, no white midpoint.
# Matches the npj/Nature-style SHAP beeswarm convention. Saturated so each
# dot's value is legible against the white panel background.
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_rbm",
    [COLORS["shap_low"], "#A41AB0", COLORS["shap_high"]],
)


# --- Canonical orderings for all figures/tables ---
COHORT_ORDER = ["cohort_D", "cohort_V1", "cohort_V2"]

COHORT_LABELS = {
    "cohort_D":      "Cohort D\n(Development, n=1,010)",
    "cohort_V1":   "Cohort V1\n(External, n=173)",
    "cohort_V2": "Cohort V2\n(External, n=15,109)",
}

# Short display name (used on most figure axes / legends).
COHORT_LABELS_SHORT = {
    "cohort_D":      "D",
    "cohort_V1":   "V1",
    "cohort_V2": "V2",
}

# Display name with n inline; used in tables and longer captions.
COHORT_LABELS_WITH_N = {
    "cohort_D":      "Cohort D (n=1,010)",
    "cohort_V1":   "Cohort V1 (n=173)",
    "cohort_V2": "Cohort V2 (n=15,109)",
}

# Full names for cohort-overview table and figure captions.
COHORT_FULL_NAMES = {
    "cohort_D":      "Cohort D — Development cohort",
    "cohort_V1":   "Cohort V1 — External validation 1",
    "cohort_V2": "Cohort V2 — External validation 2",
}

METHOD_ORDER = ["canonical", "single_agent", "multi_agent"]

METHOD_LABELS = {
    "canonical":    "Canonical",
    "single_agent": "Single-agent",
    "multi_agent":  "Multi-agent",
}

TIER_ORDER = [
    "T1_acute_typical",
    "T2_atypical_or_chronic",
    "T3_prior_stent_evar_tevar",
    "T4_post_op_or_followup",
]

TIER_LABELS = {
    "T1_acute_typical":          "T1 · Acute-typical",
    "T2_atypical_or_chronic":    "T2 · Atypical / chronic",
    "T3_prior_stent_evar_tevar": "T3 · Prior stent / EVAR / TEVAR",
    "T4_post_op_or_followup":    "T4 · Post-op / follow-up",
}

ACTION_ORDER = [
    "observe_or_reassess",
    "call_lab_agent",
    "call_ecg_agent",
    "call_echo_agent",
    "direct_cta",
    "urgent_transfer",
]

ACTION_LABELS = {
    "observe_or_reassess": "Observe /\nreassess",
    "call_lab_agent":      "Continue:\nlaboratory",
    "call_ecg_agent":      "Continue:\nECG",
    "call_echo_agent":     "Continue:\nechocardio.",
    "direct_cta":          "Direct\nCTA",
    "urgent_transfer":     "Urgent pathway\nescalation",
}


# --- rcParams ---
def apply_style() -> None:
    """Apply unified Nature-subjournal figure style.

    Conventions:
    - Regular-weight, centered titles (not bold)
    - Frameless legend, method-only labels (metrics belong in tables)
    - Light grid, dark-grey spines, Okabe-Ito triad for methods
    """
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,

        # Axes — keep four spines (boxed look) but make them thin/dark-grey
        "axes.linewidth": 0.7,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.titlecolor": COLORS["ink"],
        "axes.titlepad": 8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titleweight": "regular",
        "axes.labelweight": "regular",

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],

        # Grid — ON, very light, behind data
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["light_grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "grid.alpha": 0.7,

        # Legend — Nature convention: no frame
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.45,

        # Lines
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.0,

        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # PDF — embed TrueType (Type 42), required by most journals
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, base: Path | str, *, formats=("pdf", "png")) -> dict[str, Path]:
    """Save a figure to multiple formats and return the resulting paths."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in formats:
        p = base.with_suffix(f".{fmt}")
        fig.savefig(p, format=fmt)
        paths[fmt] = p
    return paths


# --- Common helpers ---
def panel_label(ax, label: str, *, x: float = -0.18, y: float = 1.04, weight: str = "bold") -> None:
    """Add a Nature-style panel label (a, b, c, ...) to an axis."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=10, fontweight=weight,
        ha="left", va="bottom",
    )


# ---- single-panel figure conventions ----
PANEL_WIDTH_IN  = 4.4   # slightly wider for gridded chart aesthetic
PANEL_HEIGHT_IN = 3.6   # default aspect; overridable per figure
PANEL_DPI       = 300


def new_panel_fig(width: float = PANEL_WIDTH_IN, height: float = PANEL_HEIGHT_IN):
    """Create a single-panel figure with consistent sizing + axes."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def set_grid(ax, *, axis: str = "both") -> None:
    """Apply the unified gridline style to an axis."""
    ax.grid(True, which="major", axis=axis,
            color=COLORS["light_grid"], linewidth=0.6, linestyle="-",
            alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def bold_title(ax, text: str, *, fontsize: float = 11, pad: float = 8) -> None:
    """Centered bold title — matches dissertation-style figure heading."""
    ax.set_title(text, fontsize=fontsize, fontweight="bold", pad=pad, loc="center")


def lighten(color_hex: str, amount: float = 0.5) -> str:
    """Return a lighter version of a hex color, amount in [0, 1] (1 = white)."""
    h = color_hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02X}{g:02X}{b:02X}"
