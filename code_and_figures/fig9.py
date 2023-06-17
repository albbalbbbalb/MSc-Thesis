import numpy as np
import matplotlib.pyplot as plt
import magdynsys as mag


np.random.seed(1)

its = 500
num = 50
tries = 20

R = 1/3
bmin, bmax = 1e-10, 1e-6
b = np.random.uniform(bmin, bmax, num)

L = np.zeros(num)

for idx, b1 in enumerate(b):
    
    cumul = np.zeros(tries)
    for i in range(tries):
        # generate random point on circle
        # and random velocity pointing outward
        th, ph = np.random.random(2)
        X = np.array([np.cos(th), np.sin(th)])
        V = np.array([np.cos(ph), np.sin(ph)])
        V = -V if np.dot(V, X) < 0 else V

        X = np.block([R*X + 1/2,V])

        XOut, XIn, xydir, entered = mag.orbit(X, R, b1, its, maxIt=10000)

        shifts = np.array([ np.sum(xy ,axis=0) for xy in xydir])
        shifts = np.add.accumulate(shifts,0)

        XIn[:,:2] += shifts
        XOut[:,:2] += np.vstack([np.zeros(2),shifts])
        Xs = np.zeros((2*its-1,4))
        Xs[::2] = XOut
        Xs[1::2] = XIn

        _, cumul[i] = mag.circleFit(Xs[:,:2].T)
    L[idx] = np.average(cumul)
    
    
fig, ax = plt.subplots(1,1, figsize=(10,10), layout="tight")

A = np.array([b*0+1, R**2*b]).T
B = 1/L

coef, res, _, _ = np.linalg.lstsq(A, B, rcond=None)
print(coef)
print(res)
ax.scatter(b, B, c="black")

x = np.linspace(bmin, bmax,  20)
y = coef[1]*R**2*x
ax.plot(x, y)

ax.set_title("Fitting $\hat b = CR^2b$ where $R=1/3$ is fixed", usetex=True)
ax.set_xlabel("$b$ values", usetex=True)
ax.set_ylabel("(avg.) $\hat b$ values", usetex=True)

plt.savefig('fig9.png')

plt.show()
