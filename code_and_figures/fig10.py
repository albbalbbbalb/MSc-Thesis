import numpy as np
import matplotlib.pyplot as plt
import magdynsys as mag


n = 10000
dep = 128
b = 3
R = 1/3

Y =  np.linspace(-1, 1, n)*0.32+1/2

X = 1/2 + np.sqrt(R**2-(Y-1/2)**2)
Xinits = np.vstack([ X  , Y, np.ones(n), np.zeros(n)]).T
print(Xinits.shape)

LZ = np.zeros(n)

for i, Xinit in enumerate(Xinits):
    _, _, xydir, _ = mag.orbit(Xinit, R, b, dep, maxIt=10000)
    shifts = np.array([ np.sum(xy,axis=0) for xy in xydir])
    LZ[i] = mag.LZ76(shifts)

fig, ax = plt.subplots(1, 1, figsize=(10,10))

fig.tight_layout()
plt.rc('xtick', labelsize=30) 
plt.rc('ytick', labelsize=30)

ax.semilogy(Y, LZ)

plt.savefig("fig10.png")

plt.show()
