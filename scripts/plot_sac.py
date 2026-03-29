import re
import matplotlib.pyplot as plt

LOG_PATH = "logs/sac_2m.log"

pattern = re.compile(r"Step\s+(\d+).+Avg100 R=\s*([-+]?\d*\.?\d+)")

steps, returns = [], []
with open(LOG_PATH, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            steps.append(int(m.group(1)) / 1e6)   # in millions
            returns.append(float(m.group(2)))

plt.figure(figsize=(8, 4.5))
plt.plot(steps, returns, color="royalblue", linewidth=2)
plt.xlabel("Training environment steps (millions)")
plt.ylabel("Mean return (Avg100 R)")
plt.title("SAC Learning Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig("results/plots/sac_curve.png", dpi=200)
