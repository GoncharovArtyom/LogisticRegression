import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from optim import *
from lossfuncs import *
from special import *

'''Решение СЛАУ методом сопряженных градиентов'''
# n=100
# s = np.random.randn(n,n)
# S = s * np.eye(n)
# Q = sp.linalg.orth(np.random.randn(n, n))
# A = Q.dot(S.dot(Q.T))
#
# b = np.random.rand(n,1)
# x0 = np.zeros((n,1))
#
# matvec = lambda x: A.dot(x)
#
# x, code = cg(matvec, b, x0, disp=True, max_iter=200)
#
# print(code)

'''Проверка градиента и гессиана'''
# X = np.random.randn(5,5)
# y = np.ones((5,1))
# y[1] = -1
# y[2] = -1
#
# w = np.random.randn(5,1)
# reg_coeff = 1/5
#
# f, g, h = logistic(w, X, y, reg_coeff, True)
#
# func = lambda w: logistic(w, X, y, reg_coeff)[0]
# h1 = hess_finite_diff(func, w)
# print(h1)

'''Проверка gd'''
# X = np.random.randn(5,5)
# y = np.ones((5,1))
# y[1] = -1
# y[2] = -1
#
# w = np.random.randn(5,1)
# reg_coeff = 1/5
# func = lambda x: logistic(x, X, y, reg_coeff)
#
# gd(func, w, disp = True)

'''Начальные установки'''
elemsnum = 3000
print(2)
data = load_svmlight_file('gisette_scale.svmlight')
print(1)
X = data[0].toarray()
y = data[1].reshape(X.shape[0], 1)
y = y[0:elemsnum,:]
X = X[0:elemsnum,:]

d=X.shape[1]
n=X.shape[0]
w_start = np.zeros((d,1))
reg_coeff = 1/n

print(d)

'''Отчет (cg)'''
# k = [5, 25, 125];
# color = ['r', 'g', 'b'];
# n = 100
# for j in range(3):
#     for i in range(3):
#         s = np.linspace(1, k[j], n).T;
#         S = s * np.eye(n)
#         Q = sp.linalg.orth(np.random.randn(n, n))
#         A = Q.dot(S.dot(Q.T))
#
#         b = np.random.rand(n,1)
#         x0 = np.zeros((n,1))
#
#         matvec = lambda x: A.dot(x)
#
#         x, code, hist = cg(matvec, b, x0, disp=True, max_iter=200, trace=True)
#
#         plt.plot(hist['n_evals'], hist['norm_g'], color[j], alpha=0.8)
#         plt.plot(hist['n_evals'][len(hist['n_evals'])-1], hist['norm_g'][len(hist['norm_g'])-1], color[j]+'o', alpha=0.8, markersize = 10)
# plt.xlabel('iterations');
# plt.ylabel('norm_g');
#
# plt.show()

'''Обучение с помощью gd'''
func = lambda x: logistic(x, X, y, reg_coeff,disp = True, hess=False)
x, loss, code, hist1 = gd(func, w_start, max_iter = 10000,max_n_evals=10000, trace=True)
if(code == 0):
    print("gd сошелся")
else:
    print("gd не сошелся")

'''Обучение с помощью ncg'''
func = lambda x: logistic(x, X, y, reg_coeff,disp = True, hess=False)
x, loss, code, hist2 = ncg(func, w_start, max_iter = 10000,max_n_evals=10000, trace=True)
if(code == 0):
    print("ncg сошелся")
else:
    print("ncg не сошелся")

'''Обучение с помощью newton'''
func = lambda x: logistic(x, X, y, reg_coeff, hess=True)
x, loss, code, hist3 = newton(func, w_start, max_iter = 10000,max_n_evals=10000,disp = True, trace=True)
if(code == 0):
    print("newton сошелся")
else:
    print("newton не сошелся")

'''Вывод графиков'''
plt.plot(hist1['n_evals'], np.log(hist1['norm_g']),'r-')
plt.plot(hist2['n_evals'], np.log(hist2['norm_g']),'b-')
plt.plot(hist3['n_evals'], np.log(hist3['norm_g']),'g-')

plt.plot(hist1['n_evals'][len(hist1['n_evals'])-1], np.log(hist1['norm_g'][len(hist1['norm_g'])-1]),'ro', markersize = 10)
plt.plot(hist2['n_evals'][len(hist2['n_evals'])-1], np.log(hist2['norm_g'][len(hist2['norm_g'])-1]),'bo', markersize = 10)
plt.plot(hist3['n_evals'][len(hist3['n_evals'])-1], np.log(hist3['norm_g'][len(hist3['norm_g'])-1]),'go', markersize = 10)
plt.legend(('gradient descent', 'conjugate gradient', 'newton'))
plt.xlabel("number of oracle's calls");
plt.ylabel("log(grad_norm)");

plt.figure()
plt.plot(hist1['elaps_t'], np.log(hist1['norm_g']),'r-')
plt.plot(hist2['elaps_t'], np.log(hist2['norm_g']), 'b-')
plt.plot(hist3['elaps_t'], np.log(hist3['norm_g']),'g-')

plt.plot(hist1['elaps_t'][len(hist1['elaps_t'])-1], np.log(hist1['norm_g'][len(hist1['norm_g'])-1]),'ro', markersize = 10)
plt.plot(hist2['elaps_t'][len(hist2['elaps_t'])-1], np.log(hist2['norm_g'][len(hist2['norm_g'])-1]),'bo', markersize = 10)
plt.plot(hist3['elaps_t'][len(hist3['elaps_t'])-1], np.log(hist3['norm_g'][len(hist3['norm_g'])-1]),'go', markersize = 10)
plt.legend(('gradient descent', 'conjugate gradient', 'newton'))
plt.xlabel("elapsed time");
plt.ylabel("log(grad_norm)");

plt.show()
