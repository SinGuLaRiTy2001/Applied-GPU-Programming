import numpy as np
from matplotlib import pyplot as plt

'''
EX3-1/3
'''
x = [128, 256, 512, 1024, 2048, 4096, 8192, 12800]
y = [5.17, 9.12, 17.95, 36.31, 72.24, 141.98, 286.04, 441.90]
plt.plot(x, y, '-o',  label=' FLOPS ')
plt.xlabel("dimX")
plt.ylabel("FLOPS")
plt.title("FLOPS with dimX")
plt.legend()
plt.show()

'''
EX3-2
'''
x = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
y = [3.31, 2, 1.49, 0.96, 0.66, 0.47, 0.34, 0.25, 0.19, 0.14, 0.11, 0.08]
plt.plot(x, y, 'o-', label='relative error')
plt.xlabel("nsteps")
plt.ylabel("relative error of the approximation")
plt.title("relative error of the approximation with nsteps")
plt.legend()
plt.show()

'''
EX3-3
'''
x = [128, 256, 512, 1024, 2048, 4096, 8192, 12800]
y = [5.17, 9.12, 17.95, 36.31, 72.24, 141.98, 286.04, 441.90]
plt.plot(x, y,  label=' prefetching ')

x1 = [128, 256, 512, 1024, 2048, 4096, 8192, 12800]
y1 = [4.29, 6.54, 15.35, 35.26, 62.07, 124.64, 268.78, 409.55]
plt.plot(x1, y1,  label=' no prefetching')
plt.xlabel("dimX")
plt.ylabel("FLOPS")
plt.title("prefetching/no prefetching")
plt.legend()
plt.show()