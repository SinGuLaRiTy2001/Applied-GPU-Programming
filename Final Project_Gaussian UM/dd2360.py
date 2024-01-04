from matplotlib import pyplot as plt
import numpy as np

categories = ["total time", "kernel time"]
x_label = ["Original", "UM", "UM_Advised", "UM_Prefetched"]
x_label1 = ["UM",  "UM_Advised", "UM_Prefetched"]
bar_width = 0.2
index = np.arange(len(x_label))
index1 = np.arange(len(x_label1))
total_kernel = np.array([[0.21, 0.006], [0.011, 0.011],  [0.0096, 0.0096], [0.007, 0.007]])
total_kernel_no_ori = np.array([[0.011, 0.011],  [0.009599, 0.009598], [0.007, 0.007]])

plt.subplot(1, 2, 1)
for i, category in enumerate(categories):
    plt.bar(index + i * bar_width, total_kernel[:, i], bar_width, label=category)

plt.xlabel('Version')
plt.ylabel('Time(s)')
plt.title('Time of Different Version')
plt.xticks(index + bar_width, x_label)
plt.legend()

plt.subplot(1, 2, 2)
for i, category in enumerate(categories):
    plt.bar(index1 + i * bar_width, total_kernel_no_ori[:, i], bar_width, label=category)

plt.xlabel('Version')
plt.ylabel('Time(s)')
plt.title('Time of Different Version without Original')
plt.xticks(index1 + bar_width, x_label1)
plt.legend()
plt.show()

