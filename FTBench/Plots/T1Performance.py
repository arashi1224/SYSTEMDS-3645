import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#Mean execution times (seconds)
data = np.array([
    [0.13],   #Pandas
    [9.51],   #PySpark MLlib
    [0.1128], #UPLIFT
    [0.22]    #PyTorch
])

labels = [
    "Pandas",
    "PySpark MLlib",
    "UPLIFT",
    "PyTorch",
]

plot_colors = [
    "red",
    "orange",
    "cornflowerblue",
    "brown"
]

#Plot
n_bars = len(data)
x = np.arange(1)
width = 0.3

fig, ax = plt.subplots(figsize=(8, 5))

#Center bars around the single category
for i in range(n_bars):
    ax.bar(
        x + i * width - width * (n_bars - 1) / 2,
        data[i],
        width=width,
        color=plot_colors[i],
        label=labels[i]
    )
ax.set_yscale("log")
ax.set_ylim(0.01, 13.5)

yticks = [0.01, 0.05, 0.10, 0.50, 1.00, 5.00, 10.00]
ax.set_yticks(yticks)

ax.yaxis.set_major_formatter(
    FuncFormatter(lambda y, _: f"{y:.2f}")
)

ax.set_xticks(x)
ax.set_xticklabels(["T1"])
ax.set_ylabel("Execution Time [s]")

ax.legend(
    ncol=2,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=9
)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("SYSTEMDS-3645\FTBench\Plots\T1_Performance.pdf")
plt.close()
