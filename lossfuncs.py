import numpy as np
import scipy as sp

logaddexp = np.logaddexp
sigm = sp.special.expit
dsigm = lambda x: sp.special.expit(x)*(1 - sp.special.expit(x))

def logistic(w, X, y, reg_coef, hess=False):
    n = X.shape[0]
    arg = X.dot(w)*(-y)
    f = 1/n*np.sum(logaddexp(0, arg)) + reg_coef/2*w.T.dot(w)

    G = -y * sigm(arg)
    g = 1/n*X.T.dot(G) + reg_coef * w

    if(not hess):
        return f,g
    else:
        diag = y*y*dsigm(arg)
        D = diag * np.eye(diag.size)
        H = 1/n*X.T.dot(D.dot(X)) + reg_coef*np.eye(w.size)
        return f,g,H