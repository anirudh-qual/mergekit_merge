import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import re

#Read json files from directory path
directory_path = "vllm_results_10merge_new"
json_files = sorted([f for f in os.listdir(directory_path) if f.endswith(".json")])

# Read each JSON file as a single record (not a list)
dfs = []
for f in json_files:
    # Extract qps number from filename using regex
    qps_match = re.search(r'(\d+\.?\d*)qps', f)
    qps = float(qps_match.group(1)) if qps_match else None
    
    with open(os.path.join(directory_path, f), 'r') as file:
        data = json.load(file)
        if qps:
            data['batch_size'] = qps
        dfs.append(pd.DataFrame([data]))

if not dfs:
    raise ValueError(f"No JSON files found in {directory_path}")

df = pd.concat(dfs, ignore_index=True) 
df = df.sort_values('batch_size').reset_index(drop=True)

#Create plots for columns p99_ttft_ms, mean_ttft_ms, p99_tpot_ms, mean_tpot_ms
plot_columns = ["p99_ttft_ms", "mean_ttft_ms", "p99_tpot_ms", "mean_tpot_ms","p95_ttft_ms","p95_tpot_ms","mean_itl_ms","p99_itl_ms","p95_itl_ms"]

for col in plot_columns:
    plt.figure(figsize=(8, 5))
    plt.plot(df["batch_size"], df[col], marker='o', linewidth=2)
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title(f'{col} vs Batch Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{directory_path}/{col}_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

# # Create 3 separate plots
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # Plot 1: TTFT (Time To First Token)
# axes[0].plot(df["batch_size"], df["avg_ttft"], marker='o', color='blue', linewidth=2)
# axes[0].set_xlabel('Batch Size', fontsize=12)
# axes[0].set_ylabel('Time (seconds)', fontsize=12)
# axes[0].set_title('Time To First Token (TTFT)', fontsize=14, fontweight='bold')
# axes[0].grid(True, alpha=0.3)

# # Plot 2: E2E (End-to-End Latency)
# axes[1].plot(df["batch_size"], df["avg_e2e"], marker='o', color='green', linewidth=2)
# axes[1].set_xlabel('Batch Size', fontsize=12)
# axes[1].set_ylabel('Time (seconds)', fontsize=12)
# axes[1].set_title('End-to-End Latency (E2E)', fontsize=14, fontweight='bold')
# axes[1].grid(True, alpha=0.3)

# # Plot 3: TBT (Time Between Tokens)
# axes[2].plot(df["batch_size"], df["avg_tbt"], marker='o', color='red', linewidth=2)
# axes[2].set_xlabel('Batch Size', fontsize=12)
# axes[2].set_ylabel('Time (seconds)', fontsize=12)
# axes[2].set_title('Time Between Tokens (TBT)', fontsize=14, fontweight='bold')
# axes[2].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig("metrics_plot.png", dpi=300, bbox_inches='tight')
# plt.show()
