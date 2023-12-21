import matplotlib.pyplot as plt

segment_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

async_times = [33.59, 24.83, 19.16, 16.02, 14.04, 12.24, 11.87]

plt.figure(figsize=(10, 6))
plt.plot(segment_sizes, async_times, label='Streamed Execution Time', marker='o')

plt.xlabel('Segment Size (S_seg)')
plt.ylabel('Execution Time (ms)')
plt.title('CUDA Streamed Execution Time for Different Segment Sizes')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.show()

