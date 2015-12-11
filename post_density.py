import numpy as np
from kernel import *

def Find_Density(x, m, h, R, theta):
    """
    Finds the density at some radius R

    Input: particles position x = (x,y)
           mass of particle m
           smoothing kernel h
           distance from center R
           angle theta

    Output: Density of particle at position x

    """

    N = len(x)

    f_rho = 0

    pos = np.array([R*np.cos(theta), R*np.sin(theta)])
    # print(pos)
    # print(x)

    for i in range(N):
        # print(x[i,:])
        x_ij = pos - x[i,:]
        f_rho_ij = m * Kernel(x_ij,h)
        f_rho = f_rho + f_rho_ij

    return f_rho
