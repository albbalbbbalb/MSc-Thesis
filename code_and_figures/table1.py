import numpy as np
from re import sub
from tabulate import tabulate
import magdynsys as mag

np.random.seed(1)

its = 500
num = 15
avg = 20

R = np.random.uniform(0.05, 0.49, num)
b = np.random.uniform(1e-5, 1e-2, num)
avgL = np.zeros_like(R)
avgdiff = np.zeros_like(R)

for idx, (R1, b1) in enumerate(zip(R,b)):
    
    cumul = np.zeros(avg)
    
    for i in range(avg):
        
        # generate random point on circle
        # and random velocity on outward
        th, ph = np.random.random(2)
        X = np.array([np.cos(th), np.sin(th)])
        V = np.array([np.cos(ph), np.sin(ph)])
        V = -V if np.dot(V, X) < 0 else V

        X = np.block([R1*X + 1/2,V])

        XOut, XIn, xydir, dirs, entered = mag.orbit(X, R1, b1, its, maxIt=10000)

        shifts = np.array([ np.sum([mag.from_where[i] for i in xy],axis=0) for xy in xydir])
        shifts = np.add.accumulate(shifts,0)

        XIn[:,:2] += shifts
        XOut[:,:2] += np.vstack([np.zeros(2),shifts])
        Xs = np.zeros((2*its-1,4))
        Xs[::2] = XOut
        Xs[1::2] = XIn

        _, cumul[i] = mag.circleFit(Xs[:,:2].T)
    avgL[idx] = np.average(cumul)
    
avgdiff = np.abs(np.pi*avgL*R**2*b - 1)
    
table = np.vstack([R, b, avgL, avgdiff]).T
tab = tabulate(table, headers=["$R$", "$b$", "$\hat L$ (avg. over "+str(avg)+")", "$\Delta = \pi\hat LR^{2}b-1$"],
                        tablefmt="latex_raw", floatfmt=".3E", showindex=range(1,num+1))
tab = sub(r'([0-9].[0-9]+)E((-)?|\+)0?([0-9]+)',r'$\1\\cdot10^{\3\4}$', tab)

with open("table1.tex", "w") as text_file:
    text_file.write(tab)

print(tab)
