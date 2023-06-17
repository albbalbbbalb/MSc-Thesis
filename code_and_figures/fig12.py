import numpy as np
import matplotlib.pyplot as plt
import magdynsys as mag


n = 1000
dep = 128
b = np.linspace(0.1, 10, n)
R = 1/3

Xinit = np.array([ 5/6, 1/2, 1, 0])

LZ = np.zeros(n)

for i, bs in enumerate(b):
    _, _, xydir, _ = mag.orbit(Xinit, R, bs, dep, maxIt=10000)
    shifts = np.array([ np.sum(xy,axis=0) for xy in xydir])
    LZ[i] = mag.LZ76(shifts)

fig, ax = plt.subplots(1, 1, figsize=(10,10))

fig.tight_layout()
plt.rc('xtick', labelsize=30) 
plt.rc('ytick', labelsize=30)

ax.semilogy(b, LZ)

plt.savefig("fig12.png")

plt.show()
