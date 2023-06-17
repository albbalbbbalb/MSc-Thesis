import numpy as np
import matplotlib.pyplot as plt
import magdynsys as mag


np.random.seed(1)

its = 500
num = 50
tries = 20

Rmin, Rmax = 0.25, 0.45
bmin, bmax = 1e-10, 1e-6
R = np.random.uniform(Rmin, Rmax, num)
b = np.random.uniform(bmin, bmax, num)

L = np.zeros(num)

for idx, (R1, b1) in enumerate(zip(R,b)):
    
    cumul = np.zeros(tries)
    for i in range(tries):
        # generate random point on circle
        # and random velocity pointing outward
        th, ph = np.random.random(2)
        X = np.array([np.cos(th), np.sin(th)])
        V = np.array([np.cos(ph), np.sin(ph)])
        V = -V if np.dot(V, X) < 0 else V

        X = np.block([R1*X + 1/2,V])

        XOut, XIn, xydir, entered = mag.orbit(X, R1, b1, its, maxIt=10000)

        shifts = np.array([ np.sum(xy ,axis=0) for xy in xydir])
        shifts = np.add.accumulate(shifts,0)

        XIn[:,:2] += shifts
        XOut[:,:2] += np.vstack([np.zeros(2),shifts])
        Xs = np.zeros((2*its-1,4))
        Xs[::2] = XOut
        Xs[1::2] = XIn

        _, cumul[i] = mag.circleFit(Xs[:,:2].T)
    L[idx] = np.average(cumul)
    
    
fig, ax = plt.subplots(1,1, figsize=(10,10), subplot_kw=dict(projection="3d"), layout="tight")

A = np.array([R*0+1, R, b, R**2, R*b, b**2, R**3, R**2*b, R*b**2, b**3]).T
B = 1/L

coef, res, _, _ = np.linalg.lstsq(A, B, rcond=None)
print(coef)
print(res)
ax.scatter(R, b, 1/L, c="black")

x = np.linspace(Rmin, Rmax, 20) 
y = np.linspace(bmin, bmax,  20)
X, Y = np.meshgrid(x, y, copy=False)
Z = coef[7]*X**2*Y
ax.plot_surface(X, Y, Z, alpha = 0.5)

ax.set_title("Fitting $\hat b = CR^2b$ where $C$ expected to be $\pi$", usetex=True, y=0.9)
ax.set_xlabel("$R$ values", usetex=True)
ax.set_ylabel("$b$ values", usetex=True)
ax.set_zlabel("(avg.) $\hat b$ values", usetex=True)
ax.view_init(elev =5)

plt.savefig('fig8.png')

plt.show()
