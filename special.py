import numpy as np

def grad_finite_diff(func, x, eps=1e-8):

    g=np.zeros(x.shape)
    n=x.size

    for i in range(n):
        e = np.zeros(x.shape)
        e[i,0] = eps
        g[i] = (func(x + e) - func(x))/(eps)

    return g

def hess_finite_diff(func, x, eps=1e-5):

    n = x.size
    h = np.zeros((n,n))

    for i in range(n):
        e1 = np.zeros(x.shape)
        e1[i, 0] = eps
        for j in range(n):
            e2 = np.zeros(x.shape)
            e2[j, 0] = eps

            h[i,j] = (func(x + e1 + e2) - func(x + e1) - func(x + e2) + func(x))/(eps*eps)

    return h