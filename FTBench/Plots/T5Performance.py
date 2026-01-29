import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#Mean execution times (seconds)
data = np.array([
    [5.07],      #Pandas
    # [5],      #UPLIFT
    [10.82]    #PyTorch
])

labels = [
    "Pandas",
    # "UPLIFT",
    "PyTorch"
]

plot_colors = [
    "red",
    "orange"
    # "cornflowerblue"
]

#Plot
n_bars = len(data)
x = np.arange(1)
width = 0.2

fig, ax = plt.subplots(figsize=(7, 5))

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
ax.set_ylim(0, 15)

yticks = [1.00, 2.00, 3.00, 4.00, 5.00, 10.00]
ax.set_yticks(yticks)

ax.yaxis.set_major_formatter(
    FuncFormatter(lambda y, _: f"{y:.2f}")
)

ax.set_xticks(x)
ax.set_xticklabels(["T5"])
ax.set_ylabel("Execution Time [s]")

ax.legend(
    ncol=2,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=9
)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("SYSTEMDS-3645\FTBench\Plots\T5_Performance.pdf")
plt.close()
