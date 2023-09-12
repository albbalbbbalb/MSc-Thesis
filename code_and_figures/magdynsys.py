import numpy as np

def Larmor_center(X, b):
    x1, x2, v1, v2 = X
    return np.array([x1+v2/b,x2-v1/b])

def reflect_mat(C):
    C1, C2 = C
    A = C1**2 - C2**2
    B = 2*C1*C2
    return np.array([[A,B],[B,-A]]) / np.sum(C**2)

def SInToSOut(X,b):
    pos, vel = X[:2], X[2:]
    shiftX = pos - 1/2
    A = reflect_mat(Larmor_center(np.block([shiftX,vel]),b))
    new_pos = A @ shiftX + 1/2
    new_vel = - A @ vel
    return np.block([new_pos,new_vel])

def circle_and_line(X, R):

    A = np.sum(X[2:]**2)
    B = np.dot(X[:2]-1/2,X[2:])
    C = np.sum((X[:2]-1/2)**2)-R**2

    d = B**2-C*A
    if d>=0:
        t = (-B - np.sqrt(d))/A
        return np.block([X[2:]*t+X[:2],X[2:]])
    return np.block([np.ones(2)*np.Inf,X[2:]])


def line_and_4lines(X,tol=1e-15):

    x1, x2, v1, v2 = X
    t1, t2 = np.Inf, np.Inf
    d1, d2 = int(v1 > 0), int(v2 > 0)

    if np.abs(v2) > tol:
        t2 = (d2-x2)/v2

    if np.abs(v1) > tol:
        t1 = (d1-x1)/v1

    if t1 < t2:
        return np.array([d1, v2*t1+x2, v1, v2])
    else:
        return np.array([v1*t2+x1, d2, v1, v2])




to_where = {(-1,-1):0, (-1, 0):1, (-1, 1):2, ( 0, 1):3,
            ( 1, 1):4, ( 1, 0):5, ( 1,-1):6, ( 0,-1):7, ( 0, 0):8}

from_where = {
        0:np.array([-1,-1]),
        1:np.array([-1, 0]),
        2:np.array([-1, 1]),
        3:np.array([ 0, 1]),
        4:np.array([ 1, 1]),
        5:np.array([ 1, 0]),
        6:np.array([ 1,-1]),
        7:np.array([ 0,-1]),
        8:np.array([ 0, 0])
        }


def cellToCell(X, tol=1e-15):
    '''
    takes a point and maps it to a point on the boundary

    x=0 or x=1 for y in [0,1]
    y=0 or y=1 for x in [0,1]

    while also translating the point to the right side of the square
    depending on the direction of the velocity
    '''

    x1, x2, v1, v2 = X
    t1, t2 = np.Inf, np.Inf
    d1, d2 = int(v1 > 0), int(v2 > 0)
    s1, s2 = 0, 0

    if np.abs(v2) > tol:
        s2 = np.sign(v2)
        t2 = (d2 - x2)/v2

    if np.abs(v1) > tol:
        s1 = np.sign(v1)
        t1 = (d1 - x1)/v1


    if np.abs(t1 - t2) < tol:
        return np.array([1 - d1, 1 - d2, v1, v2]),     np.array([s1, s2])
    elif t1 < t2:
        return np.array([1 - d1, v2*t1 + x2, v1, v2]), np.array([s1,  0])
    else:
        return np.array([v1*t2 + x1, 1 - d2, v1, v2]), np.array([ 0, s2])


def next_sqr(X,tol=1e-15):

    x,y = X[:2]

    dx = int(np.abs(x - 1) < tol) - int(np.abs(x) < tol)
    dy = int(np.abs(y - 1) < tol) - int(np.abs(y) < tol)

    return np.array(direction), np.array([dx, dy])




def SOutToSIn(X, R, maxIt=1000, tol=1e-15):
    '''
    Assumes that the input is on SOut
    
    Xn          is the point at which we enter SIn
    xydir       gives an array of directions for the LZ76 complexity
    entered     a flag announcing the entry into SIn
    '''
    
    xydir = np.zeros((maxIt, 2))
    Xn, xydir[0,:] = cellToCell(X, tol)
    entered = 0

    for m in range(1, maxIt):
        circ = circle_and_line(Xn, R)
        if circ[0] != np.Inf:
            Xn = circ
            entered = 1
            break
        Xn, xydir[m,:] = cellToCell(Xn, tol)
    
    return Xn, xydir[:m,:], entered


def orbit(X, R, b, n=1000, maxIt=1000, tol=1e-15):
    '''
    Assumes the input is on SOut    
    '''

    xydir = np.zeros(n, dtype=object)
    XOut = np.zeros((n+1,4))
    XIn = np.zeros((n,4))
    XOut[0] = X
    entered = 1

    for m in range(n):
        XIn[m], xydir[m], entered = SOutToSIn(XOut[m], R, maxIt,tol)
        if 1 - entered:
            break
        XOut[m+1] = SInToSOut(XIn[m], b)

    return XOut[:m+1], XIn[:m], xydir[:m], entered




def plotCircles(ax, R, xydir, **kwargs):

    from matplotlib.patches import Circle

    shifts = np.array([ np.sum(xy,axis=0) for xy in xydir])
    shifts = np.add.accumulate(shifts,0)
    shifts = np.unique(shifts, axis=0)
 
    for a,b in shifts:
        ax.add_patch(Circle((a+1/2,b+1/2), R, color="black", linestyle=":", fill=False))

    out = ax.add_patch(Circle((1/2,1/2), R, color="black", linestyle=":", fill=False))

    return out




def plotTrajectory(ax, XOut, XIn, xydir, **kwargs):

    XO = XOut.copy()
    XI = XIn.copy()

    shifts = np.array([ np.sum(xy,axis=0) for xy in xydir])
    shifts = np.add.accumulate(shifts,0)
    
    XI[:,:2] += shifts
    XO[:,:2] += np.vstack([np.zeros(2), shifts])

    n = XI.shape[0] + XO.shape[0]
    Xs = np.zeros((n,4))
    Xs[::2] = XO
    Xs[1::2] = XI

    out = ax.plot(Xs[:,0], Xs[:,1], "b", **kwargs)

    
    return out




def prettyPlotTrajectory(ax, XOut, XIn, xydir):

    XO = XOut.copy()
    XI = XIn.copy()

    shifts = np.array([ np.sum(xy,axis=0) for xy in xydir])
    shifts = np.add.accumulate(shifts,0)
    
    XI[:,:2] += shifts
    XO[:,:2] += np.vstack([np.zeros(2), shifts])

    n = XI.shape[0] + XO.shape[0]
    Xs = np.zeros((n,4))
    Xs[::2] = XO
    Xs[1::2] = XI

    for k in range(Xs.shape[0]//2):
        ax.plot(Xs[2*k:2*k+2,0], Xs[2*k:2*k+2,1], "b")
        ax.plot(Xs[2*k+1:2*k+3,0], Xs[2*k+1:2*k+3,1], "r")
    
    out = ax.plot(Xs[0,0],Xs[0,1],"ks")
    
    return out





def preparePoincare(ax, S='out', **kwargs):

    from matplotlib.patches import Rectangle

    H = np.pi/2 * np.sqrt(2)

    if S == 'out':
        artists = [
            Rectangle((-np.pi,   np.pi/2),   H, H, angle=45, alpha=0.5),
            Rectangle((-np.pi,-3*np.pi/2), 5*H, H, angle=45, alpha=0.5),
            Rectangle(( np.pi,-3*np.pi/2),   H, H, angle=45, alpha=0.5)
            ]
    elif S == 'in':
        artists = [
            Rectangle((-np.pi,  -np.pi/2), 3*H, H, angle=45, alpha=0.5),
            Rectangle((     0,-3*np.pi/2), 3*H, H, angle=45, alpha=0.5)
            ]

    for i in artists:
        ax.add_artist(i)

    ticks = np.linspace(-1,1,5)*np.pi
    labels = ["$-\pi$", "$-\\frac{\pi}{2}$" , "$0$", "$\\frac{\pi}{2}$" , "$\pi$"]
    ax.set_xticks(ticks, labels, **kwargs)
    ax.set_yticks(ticks, labels, **kwargs)
    ax.grid(color='k', linestyle=':')

    ax.set_xlim([-np.pi,np.pi])
    out = ax.set_ylim([-np.pi,np.pi])

    return out




def plotPoincareOut(ax, XOut, **kwargs):

    preparePoincare(ax, S='out', **kwargs)

    thetas = np.arctan2(XOut[:,1]-1/2,XOut[:,0]-1/2)
    phis = np.arctan2(XOut[:,3],XOut[:,2])
    out = ax.plot(thetas, phis,"rs")

    return out




def circleFit(X):
    '''
    X - (n,m) array, n is the dimension of the circle, and m is the number of data points

    fits an n circle to the data points via the method of I. Coope (1993)

    returns

    c - the center of the circle
    r - the radius
    '''

    B = np.pad(X, ((0, 1), (0, 0)), constant_values=(1,)).T
    d = np.linalg.norm(X, axis=0)**2

    Y = np.linalg.lstsq(B, d, rcond=-1)[0]

    c = Y[:-1]/2
    r = np.sqrt(Y[-1] + np.linalg.norm(c)**2)

    return c, r




def LZ76(ss):
    """
    Albert: we removed .flatten() for ss because it was easier to work with that way

    Simple script implementing Kaspar & Schuster's algorithm for
    Lempel-Ziv complexity (1976 version).
    
    If you use this script, please cite the following paper containing a sample
    use case and further description of the use of LZ in neuroscience:
    
    Dolan D. et al (2018). The Improvisational State of Mind: A Multidisciplinary
    Study of an Improvisatory Approach to Classical Music Repertoire Performance.
    Front. Psychol. 9:1341. doi: 10.3389/fpsyg.2018.01341
    
    Pedro Mediano and Fernando Rosas, 2019

    Calculate Lempel-Ziv's algorithmic complexity using the LZ76 algorithm
    and the sliding-window implementation.

    Reference:

    F. Kaspar, H. G. Schuster, "Easily-calculable measure for the
    complexity of spatiotemporal patterns", Physical Review A, Volume 36,
    Number 2 (1987).

    Input:
      ss -- array of integers

    Output:
      c  -- integer
    """

    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    n = len(ss)
    while True:
        if ss[i + k - 1] == ss[l + k - 1]:
            k = k + 1
            if l + k > n:
                c = c + 1
                break
        else:
            if k > k_max:
               k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
    return c

