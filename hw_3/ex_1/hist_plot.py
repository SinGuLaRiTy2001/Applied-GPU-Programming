import numpy as np
import matplotlib.pyplot as plt

histData = []
n_bins = 4096
bar_width = 1.0

with open('output.txt') as f:
    lines = f.readlines()
    for line in lines:
        histData.append(int(line))

hist, bin_edges = np.histogram(histData, bins=n_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.bar(bin_centers, hist, align='center', width=bar_width)
plt.show()
