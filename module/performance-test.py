import numpy as np
from timeit import timeit
import scipy
from scipy import linalg
import quantum_linear_systems as qls
import module.qhll as qhll

b = np.random.random(8)
T = 1000

def func1():
    phi_0 = np.zeros(T, dtype="complex")
    for i in range(T):
        phi_0[i] = np.sqrt(2 / T) * np.sin(np.pi * (i + 1 / 2) / T)
    return phi_0

#faster than the first one
def func2():
    phi_0 = [np.sqrt(2 / T) * np.sin(np.pi * (i + 1 / 2) / T) for i in range(T)]
    return phi_0


phi_0 = np.random.random(T)
m = len(b)

def func3():
    registers = np.zeros(T * len(b), dtype='complex')
    for i in range(T):
        registers[i * m:(i + 1) * m] = phi_0[i] * b
    return registers

#faster than the first one
def func4():
    registers = np.kron(phi_0, b)
    return registers

A = np.random.random((10,10))
b = np.random.random(10)

def func5():
    prof = qls.qls(A, b, 100, 0.01)
    return prof

def func6():
    me = qhll.hhl(A, b, 0.01, 100)
    return me

for func in (func5, func6):
   print(func.__name__ + ':', timeit(func, number=100))