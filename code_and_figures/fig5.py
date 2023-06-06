import magdynsys as mag
import matplotlib.pyplot as plt
import numpy as np


suptitle = {"fontsize":23, "usetex":True}
titles = {"fontsize":20, "usetex":True} 
ticks = {"fontsize":15, "usetex":True} 

n = 100
R = 1/3
b = 3
Y = -1/b + 1/2
X = 1/2 - np.sqrt(R**2-(Y-1/2)**2)
Xinit = np.array([ X  , Y, 0, -1])
print(Xinit)

XOut, XIn, xydir, _ = mag.orbit(Xinit, R, b, n, maxIt=1000)

fig, ax = plt.subplots(1, 2, figsize=(10,10))


mag.plotCircles(ax[0], R, xydir, **ticks)
mag.plotTrajectory(ax[0], XOut, XIn, xydir)

mag.plotPoincareOut(ax[1], XOut, **ticks)

ax[0].set_aspect(1)
ax[1].set_aspect(1)



ax[0].set_title("Trajectory", **titles)
ax[1].set_title("Poincar\\'e section", **titles)

fig.suptitle("Quasiperiodic orbit, $(X,V)\\approx(0.5, 0.166, 0, -1), R=1/3, b=3$",
        **suptitle, y=0.77)
fig.tight_layout()

plt.savefig("fig5.png")
plt.show()
