import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_hpl_results.py <input.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    pivot = df.pivot_table(index="N", columns="NB", values="Gflops")

    plt.figure(figsize=(10, 10), dpi=300)

    sns.heatmap(
    pivot,
    annot=True,
    fmt=".00f",  # full integer Gflops
    cmap="viridis",
    linewidths=0.5,
    cbar_kws={"label": "Gflops"},
    )

    plt.title("HPL Performance Heatmap (Gflops vs N and NB)", fontsize=14)
    plt.xlabel("NB (Block Size)")
    plt.ylabel("N (Problem Size)")
    
    # Make tick labels dense and readable
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # Add major and minor grid lines
    ax = plt.gca()
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
    ax.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.3)
    
    # Enable minor ticks for dense grids
    ax.minorticks_on()
    
    plt.tight_layout()
    plt.savefig("hpl_heatmap.png", dpi=300)
    plt.close()

    print("Plots saved as 'hpl_heatmap.png'.")

if __name__ == "__main__":
    main()   
