import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import MultipleLocator, FuncFormatter

with open("../iterations.txt") as iters_f:
    text = iters_f.read()
    parts = text.split("\n\n")
    loss_vals = []
    for part in parts:
        second_line = part.split('\n')[1]
        loss_val = float(second_line.split(' = ')[1].strip())
        loss_vals.append(loss_val)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.grid(alpha = 0.3)
ax1.set_title("Fitting the AutoInsrSweden.txt data", fontsize=20, pad=20)
ax1.set_ylabel("Loss", size=17)
ax1.set_xlabel("no. of iteration", size=17)
ax1.set_xlim(-2E4, 5E6 + 2E4)
ax1.set_ylim(-100, max(loss_vals) + 100)
ax1.tick_params(axis='y', labelsize=13)
ax1.tick_params(axis='x', labelsize=13)
ax1.plot(range(0, 5_000_000, 20_000), loss_vals, linewidth=3, marker='.', markersize=7, mfc='red', mew=0,
                    label="Loss on train data through iterations of Batch GD")
ax1.yaxis.set_major_locator(MultipleLocator(1000))
ax1.xaxis.set_major_locator(MultipleLocator(1E6 / 2))
ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, y: f"{x / 10**6:.1f}" + " mln"))
ax1.xaxis.set_minor_locator(MultipleLocator(1E6 / 4))
ax1.text(2.3E6, 8300, r"""Model: 2-hiddle layer MLP with sigmoid activation after each
layer except the output one. Size of input = 1 + bias, 
Size of 2 hidden layers = 128 + bias, size of output layer = 1

$\alpha = 2\mathrm{e}{-5}$
Final loss on train data $L_{train} = 944.646$
Final loss on test data  $L_{test} = 1237.22$
Loss Type = MSE
""", fontsize=16, linespacing=1.5)
ax1.legend(fontsize=15)


plt.show()