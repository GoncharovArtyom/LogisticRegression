from numpy import *
import scipy as sp
import time as time
from scipy.optimize.linesearch import line_search_armijo, line_search_wolfe2

def cg(matvec, b, x0, tol=1e-5, max_iter=None, disp=False, trace=False):
    time_start = time.clock()

    if(not(max_iter)):
        max_iter=x0.size

    g = matvec(x0) - b
    d = -g
    u = matvec(d)
    x = x0

    if (trace):
        hist = dict(norm_g=[], n_evals=[], elaps_t=[])

    for i in range(0, max_iter):
        alpha = dot(g.T, g)/dot(d.T,u)
        x = x + alpha*d
        g_next = g + alpha*u
        norm = linalg.norm(g_next, inf)

        if (trace):
            hist['norm_g'].append(norm)
            hist['n_evals'].append(i)
            hist['elaps_t'].append(time.clock() - time_start)

        if (disp):
            print(i, ') ', norm)
        if (norm < tol):
            if(not trace):
                return (x, 0)
            return (x, 0, hist)
        else:
            betta = dot(g_next.T, g_next)/dot(g.T, g)
            d = -g_next + betta*d
            u = matvec(d)
            g = g_next

    if (trace):
        return (x, 1, hist)
    return (x, 1)

def gd(func, x0, tol=1e-4, max_iter=500, max_n_evals=1000, c1=1e-4, c2=0.9, disp=False, trace=False):

    time_start = time.clock()

    f = lambda x: func(x)[0]
    myfprime = lambda x: func(x)[1].T

    iter=0
    n_evals=1
    loss, grad = func(x0)
    norm = linalg.norm(grad,inf)
    x = x0

    if(trace):
        hist = dict(f=[loss], norm_g=[norm], n_evals=[0], elaps_t=[0])

    while(iter<max_iter and n_evals<max_n_evals):

        res = line_search_wolfe2(f, myfprime, x, -grad, grad.T, loss, c1=c1, c2=c2)
        alpha = res[0]
        fc = res[1]
        gc = res[2]
        x = x - alpha*grad
        loss, grad = func(x)
        norm = linalg.norm(grad, inf)
        n_evals = n_evals + fc + gc + 1
        iter = iter + 1

        if (trace):
            hist['f'].append(loss)
            hist['norm_g'].append(norm)
            hist['n_evals'].append(n_evals)
            hist['elaps_t'].append(time.clock() - time_start)

        if (disp):
            print(iter,') ',loss[0,0],' ',n_evals,' ',norm)

        if (norm < tol):
            result = [x, loss[0, 0], 0]
            if (trace):
                result.append(hist)
            return result

    result = [x, loss[0, 0], 1]
    if (trace):
        result.append(hist)
    return result

def newton(func, x0, tol=1e-4, max_iter=500, max_n_evals=1000, c1=1e-4, c2=0.9, disp=False, trace=False):
    time_start = time.clock()
    f = lambda x: func(x)[0]
    myfprime = lambda x: func(x)[1].T

    iter = 0
    n_evals = 1
    loss, grad, hess = func(x0)
    choDec = sp.linalg.cho_factor(hess,True)
    gk = sp.linalg.cho_solve(choDec, -grad)

    norm = linalg.norm(grad, inf)
    x = x0

    if (trace):
        hist = dict(f=[loss], norm_g=[norm], n_evals=[0], elaps_t=[0])

    while (iter < max_iter and n_evals < max_n_evals):

        res = line_search_wolfe2(f, myfprime, x, gk, grad.T, loss, c1=c1, c2=c2)
        alpha = res[0]
        fc = res[1]
        gc = res[2]
        x = x + alpha * gk
        loss, grad, hess = func(x)
        choDec = sp.linalg.cho_factor(hess, True)
        gk = sp.linalg.cho_solve(choDec, -grad)
        norm = linalg.norm(grad, inf)
        n_evals = n_evals + fc + gc + 1
        iter = iter + 1

        if (trace):
            hist['f'].append(loss)
            hist['norm_g'].append(norm)
            hist['n_evals'].append(n_evals)
            hist['elaps_t'].append(time.clock() - time_start)

        if (disp):
            print(iter, ') ', loss[0, 0], ' ', n_evals, ' ', norm)

        if (norm < tol):
            result = [x, loss[0, 0], 0]
            if (trace):
                result.append(hist)
            return result

    result = [x, loss[0, 0], 1]
    if (trace):
        result.append(hist)
    return result

def hfn(func, x0, hess_vec, tol=1e-5, max_iter=500, c1=1e-4, c2=0.9, disp=False, trace=False):

    if (trace):
        hist = {}
        hist['f'] = []
        hist['norm_g'] = []
        hist['elaps_t'] = []
        start_time = time.clock()

    f = lambda x: func(x)[0];
    df = lambda x: func(x)[1];

    x = x0
    [loss, grad, extra] = func(x)
    grad_norm = linalg.norm(grad, inf)
    eps = min(1 / 2, sqrt(grad_norm)) * grad_norm

    for i in range(0, max_iter):

        #Start cg
        z = zeros(shape(x))
        g =  grad
        d = -g
        u = hess_vec(x, d, extra)

        for j in range(0,1000):
            gamma = g.transpose().dot(g)/(d.transpose().dot(u))
            z = z + gamma*d
            g1 = g + gamma*u
            b = True
            if linalg.norm(g1,inf)<eps:
                b = False
                break
            else:
                betta = g1.transpose().dot(g1)/(g.transpose().dot(g))
                d = -g1+betta*d
                u = hess_vec(x,d,extra)
                g = g1
        if b:
            print('CG не сошелся')

        #Одномерный линейный поиск
        alpha = line_search_wolfe2(f = f,myfprime = df, xk = x, pk = z, gfk = grad, old_fval = loss, c1 = c1, c2 = c2)
        if (alpha[0] == None):
            alpha = line_search_armijo(f = f,myfprime = df, xk = x, pk = z, gfk = grad, old_fval = loss, c1 = c1, alpha0 = 1)
        x = x + alpha[0]*z

        [loss, grad, extra] = func(x)
        grad_norm = linalg.norm(grad, inf)
        eps = min(1 / 2, sqrt(grad_norm)) * grad_norm

        if (disp):
            print(str(1+i) + ')', loss, grad_norm);
        if (trace):
            hist['f'].append(loss)
            hist['norm_g'].append(grad_norm)
            current_time = time.clock() - start_time
            hist['elaps_t'].append(current_time)

        if grad_norm<tol:
            return x, loss, 0

    return x, loss, 1

def ncg(func, x0, tol=1e-4, max_iter=500, max_n_evals=1000, c1=1e-4, c2=0.1, disp=False, trace=False):
    time_start = time.clock()
    f = lambda x: func(x)[0]
    myfprime = lambda x: func(x)[1].T

    iter = 0
    n_evals = 1
    loss, g = func(x0)
    norm = linalg.norm(g, inf)
    x = x0
    d = -g

    if (trace):
        hist = dict(f=[loss], norm_g=[norm], n_evals=[0], elaps_t=[0])

    while (iter < max_iter and n_evals < max_n_evals):

        res = line_search_wolfe2(f, myfprime, x, d, g.T, loss, c1=c1, c2=c2)
        alpha = res[0]
        fc = res[1]
        gc = res[2]
        x = x + alpha * d
        loss, g_next = func(x)

        norm = linalg.norm(g, inf)
        n_evals = n_evals + fc + gc + 1
        iter = iter + 1

        if (trace):
            hist['f'].append(loss)
            hist['norm_g'].append(norm)
            hist['n_evals'].append(n_evals)
            hist['elaps_t'].append(time.clock() - time_start)

        if (disp):
            print(iter, ') ', loss[0, 0], ' ', n_evals, ' ', norm)

        if (norm < tol):
            result = [x, loss[0, 0], 0]
            if (trace):
                result.append(hist)
            return result
        else:
            betta = g_next.T.dot(g_next - g)/(g.T.dot(g))
            d = -g_next + betta*d
            g = g_next

    result = [x, loss[0, 0], 1]
    if (trace):
        result.append(hist)
    return result
