import pandas as pd
import matplotlib.pyplot as plt

# ----- Data -----
data = [
    {"Subject": "PAT013",   "Total": 494},
    {"Subject": "PAT015",   "Total": 1126},
    {"Subject": "PAT021_A", "Total": 1236},
    {"Subject": "PATID15",  "Total": 1960},
    {"Subject": "PATID16",  "Total": 434},
    {"Subject": "PATID26",  "Total": 1464},
]
df = pd.DataFrame(data)

labels = df["Subject"].tolist()
sizes = df["Total"].tolist()
total_trials = sum(sizes)

# show percent + count
def autopct_format(values):
    total = sum(values)
    def fmt(pct):
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count})"
    return fmt

# ----- Plot -----
fig, ax = plt.subplots(figsize=(9, 8))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    startangle=90,
    autopct=autopct_format(sizes),
    pctdistance=0.65,         # position of % text
    labeldistance=1.10,       # position of subject labels
    wedgeprops=dict(edgecolor="white", linewidth=2),  # clean borders
)

# Styling
plt.setp(texts, fontsize=12)
plt.setp(autotexts, fontsize=11, fontweight="bold", color="black")

ax.set_title("Distribution of Total Trials per Subject", fontsize=18, pad=20)
ax.axis("equal")

# Legend (subject + count)
legend_labels = [f"{subj}: {n}" for subj, n in zip(labels, sizes)]
ax.legend(
    wedges,
    legend_labels,
    title="Subjects",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=11,
    title_fontsize=12,
)

# Total number under the plot
fig.text(
    0.5, 0.04,
    f"Total number of trials: {total_trials}",
    ha="center",
    fontsize=13,
    fontweight="bold",
)

plt.tight_layout(rect=[0, 0.06, 0.85, 1])
plt.show()
