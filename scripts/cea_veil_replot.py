"""Re-plot the veil/structure mechanism figure on a shared, honest y-scale (no retraining)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path("results/cea/veil_mechanism")
s = json.loads((base / "summary.json").read_text(encoding="utf-8"))
fog = s["fog_sweep"]
struct = s["structure_sweep"]

fog_levels = [0.0, 0.15, 0.30, 0.45, 0.60]
ymin = min(min(a["bicubic_miou"] for a in fog + struct), min(a["inr_miou"] for a in fog + struct)) - 0.02
ymax = max(a["clean_miou"] for a in fog + struct) + 0.06

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
ax1.plot(fog_levels, [a["clean_miou"] for a in fog], "o-", label="clean", color="#444")
ax1.plot(fog_levels, [a["bicubic_miou"] for a in fog], "s-", label="fogged (bicubic)", color="#d95f02")
ax1.plot(fog_levels, [a["inr_miou"] for a in fog], "^-", label="restored (INR)", color="#1b9e77")
ax1.set_ylim(ymin, ymax)
ax1.set_xlabel("fog strength"); ax1.set_ylabel("frozen segmenter mIoU")
ax1.set_title("Veil: segmenter robust; little task damage to recover")
ax1.legend(fontsize=8, loc="lower left")
axb = ax1.twinx()
axb.bar(fog_levels, [a["psnr_gain"] for a in fog], width=0.03, alpha=0.22, color="#1b9e77")
axb.set_ylabel("INR PSNR gain (dB)", color="#1b9e77")

conds = [a["condition"] for a in struct]
x = np.arange(len(conds))
ax2.plot(x, [a["clean_miou"] for a in struct], "o-", label="clean", color="#444")
ax2.plot(x, [a["bicubic_miou"] for a in struct], "s-", label="degraded (bicubic)", color="#d95f02")
ax2.plot(x, [a["inr_miou"] for a in struct], "^-", label="restored (INR)", color="#1b9e77")
ax2.set_xticks(x); ax2.set_xticklabels(conds, rotation=20)
ax2.set_title("Structure: large task damage, substantially recovered")
ax2.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(base / "veil_vs_structure.png", dpi=130)
plt.close(fig)
print("replotted with shared y-range", round(ymin, 3), round(ymax, 3))
