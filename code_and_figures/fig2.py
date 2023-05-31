import numpy as np
import matplotlib.pyplot as plt
import itertools
import magdynsys as mag


np.set_printoptions(precision=5)
np.random.seed(1)

num = 5


R = 1/3
b = 10.0**np.arange(-1,-1-num,-1,dtype=int)
n = np.array([3000,3000,13000,20000,15000])
linestyles = ['solid','dotted','dashed','dashdot',(0, (5,10))]

X = np.random.random(2)-1/2
X = R*X/np.linalg.norm(X)
V = np.random.random(2)
V = V/np.linalg.norm(V)
V = V if np.dot(V,X) > 0 else -V
X = np.block([X+1/2,V])
print(X)



fig, ax = plt.subplots(1,1,figsize=(10,10),dpi=80)


for idx, (i,j) in enumerate(zip(b, n)):
    XOut, XIn, xydir, dirs, entered = mag.orbit(X, R, i, int(j), maxIt=10000)
    print(entered)

    shifts = np.array([ np.sum([mag.from_where[i] for i in xy],axis=0) for xy in xydir])

    shifts = np.add.accumulate(shifts,0)

    XIn[:,:2] += shifts
    XOut[:,:2] += np.vstack([np.zeros(2),shifts])

    Xs = np.zeros((2*j-1,4))
    Xs[::2] = XOut
    Xs[1::2] = XIn

    ax.plot(Xs[:,0], Xs[:,1], "b", linestyle=linestyles[idx], label='b='+str(i), linewidth=2)

ax.plot(Xs[0,0],Xs[0,1],"ks", label="Init. point")

ax.tick_params(axis='both', labelsize=20)
ax.set_aspect(1)
ax.legend(fontsize=20, loc="lower right")
ax.set_title("With $R=1/3$", fontsize=25)
fig.tight_layout()

plt.savefig('fig2.png')

plt.show()

