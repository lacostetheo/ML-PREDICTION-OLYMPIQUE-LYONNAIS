"""Shared plotting style. Colours are the first three slots of a CVD-validated
categorical palette (light mode) plus neutrals; colour follows the entity, so a
model keeps its colour in every figure."""
import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8d8c86"      # naive / reference lines
BLUE = "#2a78d6"         # series 1
ORANGE = "#eb6834"       # series 2
AQUA = "#1baf7a"         # series 3

MODEL_COLORS = {
    "Elo (LogReg)": BLUE,
    "LogReg: all features": ORANGE,
    "Bookmaker Bet365 (reference)": AQUA,
}


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2.0, "legend.frameon": False, "figure.dpi": 150,
    })
