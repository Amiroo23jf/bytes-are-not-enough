"""Plot cumulative_loss_bits vs bytes_processed for bytes and token runs."""

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

RUNS = {
    "Bytes (V=256)":    "outputs/text_transformer/enwik8_bytes",
    "Tokens (V=16000)": "outputs/text_transformer/enwik8_v16000",
}
COLORS = {
    "Bytes (V=256)":    "#378ADD",
    "Tokens (V=16000)": "#D85A30",
}

fig, ax = plt.subplots(figsize=(8, 5))

for label, run_rel in RUNS.items():
    run_dir = os.path.join(ROOT, run_rel)

    cfg_path = os.path.join(run_dir, "resolved_config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    batch_size = int(cfg["train"]["batch_size"])

    d = np.load(os.path.join(run_dir, "seeds", "seed_0.npz"))
    cumulative_bits = d["cumulative_loss_bits"] * batch_size

    ax.plot(
        d["bytes_processed"],
        cumulative_bits,
        color=COLORS[label],
        linewidth=1.5,
        label=f"{label}  (B={batch_size})",
    )

ax.set_xlabel("source bytes processed", fontsize=12)
ax.set_ylabel("cumulative loss (bits)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
fig.tight_layout()

out = os.path.join(ROOT, "outputs/text_transformer/figures/cumulative_loss.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
fig.savefig(out.replace(".png", ".pdf"))
print("Saved:", out)
plt.show()
