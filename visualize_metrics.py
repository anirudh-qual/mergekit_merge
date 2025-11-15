import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("metrics.csv")

# Create 3 separate plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: TTFT (Time To First Token)
axes[0].plot(df["batch_size"], df["avg_ttft"], marker='o', color='blue', linewidth=2)
axes[0].set_xlabel('Batch Size', fontsize=12)
axes[0].set_ylabel('Time (seconds)', fontsize=12)
axes[0].set_title('Time To First Token (TTFT)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: E2E (End-to-End Latency)
axes[1].plot(df["batch_size"], df["avg_e2e"], marker='o', color='green', linewidth=2)
axes[1].set_xlabel('Batch Size', fontsize=12)
axes[1].set_ylabel('Time (seconds)', fontsize=12)
axes[1].set_title('End-to-End Latency (E2E)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot 3: TBT (Time Between Tokens)
axes[2].plot(df["batch_size"], df["avg_tbt"], marker='o', color='red', linewidth=2)
axes[2].set_xlabel('Batch Size', fontsize=12)
axes[2].set_ylabel('Time (seconds)', fontsize=12)
axes[2].set_title('Time Between Tokens (TBT)', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("metrics_plot.png", dpi=300, bbox_inches='tight')
plt.show()