import numpy as np
import sympy as sp
import matplotlib.pylab as plt

def input_function(t, t_0, c_0) : 
    # t_0 means width of Input function
    # c_0 means height of peak 
    return c_0 * (np.heaviside(t, 1) - np.heaviside(t-t_0, 1))






