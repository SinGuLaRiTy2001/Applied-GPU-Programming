import matplotlib.pyplot as plt

input_lengths = [4096, 16384, 65536, 262144, 1048576]

sync_times = [0.72, 0.91, 1.41, 3.57, 12.53]
async_times = [0.65, 0.98, 2.75, 9.21, 33.19]

plt.figure(figsize=(10, 6))
plt.plot(input_lengths, sync_times, label='Non-streamed', marker='o')
plt.plot(input_lengths, async_times, label='Streamed', marker='o')

plt.xlabel('Input Length')
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time Comparison')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.show()
