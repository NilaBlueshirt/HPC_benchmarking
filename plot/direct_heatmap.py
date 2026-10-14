import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
if len(sys.argv) != 2:
    print("Usage: python plot_hpl_text_heatmap.py <input.txt>")
    sys.exit(1)
txt_file = sys.argv[1]
# Read whitespace-separated file, no header
df = pd.read_csv(txt_file, delim_whitespace=True, header=None, comment='#')
# Extract only the required columns: 2nd (N), 3rd (NB), 7th (Gflops)
df = df.iloc[:, [1, 2, 6]]
df.columns = ["N", "NB", "Gflops"]
# Convert scientific notation to float
df["Gflops"] = df["Gflops"].astype(float)
# Pivot to create N vs NB grid, taking MAXIMUM Gflops for duplicates
pivot = df.pivot_table(index="N", columns="NB", values="Gflops", aggfunc='max')
# Create high-resolution figure
plt.figure(figsize=(10, 10), dpi=300)
# Draw the heatmap with full Gflops values (no scientific notation)
sns.heatmap(
    pivot,
    annot=True,
    fmt=".00f",
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
ax.minorticks_on()
plt.tight_layout()
plt.savefig("hpl_heatmap.png", dpi=300)
