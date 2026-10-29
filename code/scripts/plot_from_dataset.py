# -*- coding: utf-8 -*-
"""Plot the point-to-route distance ECDF straight from the released residuals.

    python code/scripts/plot_from_dataset.py --airport KDCA --out ecdf_KDCA.png
    python code/scripts/plot_from_dataset.py --airport both --out ecdf.png

No dependency beyond numpy + matplotlib; no dataset download needed.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.environ.get("ROUTE_ID_ROOT") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(ROOT, "data")
COLOR = {"KDCA": "#1F77B4", "KLAX": "#D95F02"}


def load_airport(ap):
    tids = [x.strip() for x in open(os.path.join(DATA, ap, "tids.txt"), encoding="utf-8")
            if x.strip()]
    d = []
    for tid in tids:
        with open(os.path.join(DATA, ap, "residuals", tid + ".json"), encoding="utf-8") as f:
            d.append(np.asarray(json.load(f)["d"], float))
    return np.concatenate(d)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--airport", default="both", choices=["KDCA", "KLAX", "both"])
    ap_.add_argument("--out", default="ecdf.png")
    a = ap_.parse_args()

    aps = ["KDCA", "KLAX"] if a.airport == "both" else [a.airport]
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(3.45, 2.75))

    for ap in aps:
        d = load_airport(ap)
        x = np.sort(d)
        y = np.arange(1, x.size + 1) / float(x.size) * 100.0
        ax.plot(x, y, color=COLOR[ap], lw=1.1,
                label="%s (%d flights / %s pts)" % (ap, len(os.listdir(
                    os.path.join(DATA, ap, "residuals"))), format(x.size, ",")))
        ax.plot([np.median(d)], [50], "o", ms=3.2, mfc="w", mec=COLOR[ap], mew=0.9)
        print("%s: n=%d  P50 %.2f  P90 %.2f  P95 %.2f  <=3 m %.1f%%"
              % (ap, d.size, np.median(d), np.percentile(d, 90), np.percentile(d, 95),
                 (d <= 3).mean() * 100))

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Point $\\rightarrow$ identified route distance (m)", fontsize=8)
    ax.set_ylabel("Cumulative share of points (%)", fontsize=8)
    ax.tick_params(labelsize=7, width=0.7)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out, dpi=600, facecolor="white")
    print("saved", a.out)


if __name__ == "__main__":
    main()
