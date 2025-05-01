import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from genetic_runner import run_all_configs

def plot_heatmaps(result, name):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    victim_map = np.zeros((15, 15))
    for key, val in result["victim_positions"].items():
        x, y = eval(key) if isinstance(key, str) else key
        victim_map[y, x] = val

    sns.heatmap(victim_map, cmap="Reds", annot=True, fmt=".0f", ax=axs[0])
    axs[0].set_title("Victim Distribution")

    sns.heatmap(result["visited"], cmap="Blues", annot=True, fmt=".0f", ax=axs[1])

    axs[1].set_title("Cell Visit Frequency")
    plt.suptitle(f"Policy Result: {name}")
    plt.tight_layout()
    plt.savefig(f"Experiment/Results/{name}_heatmap.png")
    plt.close()

# def generate_summary_chart(results):
#     print("DEBUG RESULT SAMPLE:", results[0])  # Add this temporarily
#     df = pd.DataFrame([{
#         "Configuration": r["name"],
#         "Victims Found": r["victims_found"],
#         "Cells Covered": r["cells_covered"],
#         "Time (s)": r["time_taken"]
#     } for r in results])
    
#     df.to_csv("Experiment/Results/config_results_summary.csv", index=False)
    
#     df.set_index("Configuration").plot(kind="bar", figsize=(10,6))
#     plt.title("Comparison of Genetic Algorithm Configurations")
#     plt.ylabel("Values")
#     plt.tight_layout()
#     plt.savefig("config_comparison_chart.png")
#     plt.show()
def generate_summary_chart(results):
    df = pd.DataFrame([{
        "Configuration": str(r["name"]),
        "Victims Found": int(r["victims_found"]),
        "Cells Covered": int(r["cells_covered"]),
        "Time (ms)": round(float(r["time_taken"]) * 1000, 2)
    } for r in results])

    print("DEBUG DataFrame:\n", df)  # optional: for inspection

    df.to_csv("Experiment\Results\config_results_summary.csv", index=False)

    df[["Victims Found", "Cells Covered", "Time (ms)"]] = df[["Victims Found", "Cells Covered", "Time (ms)"]].apply(pd.to_numeric)
    df.set_index("Configuration")[["Victims Found", "Cells Covered", "Time (ms)"]].plot(kind="bar", figsize=(10, 6))

    plt.title("Comparison of Genetic Algorithm Configurations")
    plt.ylabel("Values")
    plt.tight_layout()
    plt.savefig("Experiment\Results\config_comparison_chart.png")
    print("✅ Chart saved as config_comparison_chart.png")
    plt.show()



if __name__ == "__main__":
    results = run_all_configs()
    for r in results:
        plot_heatmaps(r, r["name"])
    generate_summary_chart(results)
