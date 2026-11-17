"""
Generează graficele din README (analiza exploratorie) din datasetul final.

    python src/make_figures.py

Rezultat, în docs/figures/:
  03_ploaie_pe_orase.png    cât de des plouă mâine, pe orașe
  07_semnale_ploaie.png     presiune, umiditate, nebulozitate azi, după cum plouă sau nu mâine
  09_sezonalitate_zone.png  ciclul anual pe zone climatice
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "weather_dataset.csv"
FIG_DIR = ROOT / "docs" / "figures"

ZONES = ["atlantic", "mediterranean", "continental", "nordic", "mountain"]
ZONE_COLORS = dict(zip(ZONES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MONTHS = ["ian", "feb", "mar", "apr", "mai", "iun", "iul", "aug", "sep", "oct", "noi", "dec"]

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984", "axes.titleweight": "bold", "axes.titlesize": 12,
    "font.size": 10, "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
})


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"salvat: docs/figures/{name}.png")


def rain_by_city(df):
    """Procentul zilelor urmate de ploaie, pe fiecare oraș, colorat după zonă."""
    share = df["ploua_maine"].mean()
    by_city = (df.groupby(["city", "zone"])["ploua_maine"].mean()
                 .reset_index().sort_values("ploua_maine"))

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(by_city["city"], by_city["ploua_maine"] * 100, height=0.75,
            color=by_city["zone"].map(ZONE_COLORS), edgecolor="#fcfcfb", linewidth=1)
    ax.axvline(share * 100, color="#52514e", linewidth=1, linestyle="--")
    ax.text(share * 100 + 0.5, len(by_city) - 0.4, f"media {share:.0%}", color="#52514e", fontsize=9)
    ax.set_xlabel("% din zile cu ploaie a doua zi (≥ 1 mm)")
    ax.set_title("Cât de des plouă mâine, pe orașe")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ZONE_COLORS[z]) for z in ZONES]
    ax.legend(handles, ZONES, title="zonă", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    save(fig, "03_ploaie_pe_orase")


def rain_signals(df):
    """Vremea de azi pentru zilele urmate / neurmate de ploaie (boxplot)."""
    groups = [(0, "nu plouă mâine", "#eda100"), (1, "plouă mâine", "#2a78d6")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, col in zip(axes, ["pressure_change_1d", "humidity", "cloud_cover"]):
        data = [df.loc[df["ploua_maine"] == g, col].dropna() for g, _, _ in groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        flierprops={"markersize": 1, "alpha": 0.3},
                        medianprops={"color": "#0b0b0b"})
        for patch, (_, _, color) in zip(bp["boxes"], groups):
            patch.set_facecolor(color)
        ax.set_xticks([1, 2], [label for _, label, _ in groups])
        ax.set_title(col)
    fig.suptitle("Vremea de azi, după cum plouă sau nu a doua zi", fontweight="bold")
    fig.tight_layout()
    save(fig, "07_semnale_ploaie")


def seasonality(df):
    """Ciclul anual pe zone: % zile cu ploaie, precipitații medii, temperatura maximă medie."""
    zone_month = df.groupby(["zone", "month"]).agg(
        rain_share=("rain_today", "mean"), precip=("precip", "mean"),
        temp_max=("temp_max", "mean")).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    panels = [("rain_share", "% zile cu ploaie"),
              ("precip", "precipitații medii (mm/zi)"),
              ("temp_max", "temperatura maximă medie (°C)")]
    for ax, (col, title) in zip(axes, panels):
        for z in ZONES:
            s = zone_month[zone_month["zone"] == z]
            y = s[col] * 100 if col == "rain_share" else s[col]
            ax.plot(s["month"], y, color=ZONE_COLORS[z], linewidth=2, marker="o", markersize=4, label=z)
        ax.set_title(title)
        ax.set_xticks(range(1, 13), MONTHS, fontsize=8)
    axes[0].legend(frameon=False)
    fig.suptitle("Ciclul anual pe zone climatice", fontweight="bold")
    fig.tight_layout()
    save(fig, "09_sezonalitate_zone")


def main():
    df = pd.read_csv(DATA)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rain_by_city(df)
    rain_signals(df)
    seasonality(df)


if __name__ == "__main__":
    main()
