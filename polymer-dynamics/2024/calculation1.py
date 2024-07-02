import numpy as np
import sympy as sp

t_w = 2
def t_n(n) :
    return n * t_w
t_0 = 3
c_0 = 1

def Q_n(n, t) :
    return (t - t_n(n))**2 * np.heaviside(t-t_n(n), 1) - (t - t_n(n) - t_0)**2 * np.heaviside(t - t_n(n) - t_0, 1)

def q_n(n, t) :
    return c_0 * (Q_n(n-1, t) - 2*Q_n(n, t) + Q_n(n+1, t))/(2*t_w)
