import numpy as np
from kernel import *

def Calc_Density(x,m,h):
    """
    Calculates density of each particle using the smoothing kernel in kenrel.py

    Input: position x, mass m, of each particle, smoothing length h
    Output: density of each particle
    """

    # Number of particles
    N = len(x)

    # Initialize any variables needed
    rho = np.ones((N,1))*m*Kernel([0,0],h)

    for i in range(N):
        for j in range(i+1,N):
            if i != j:
                # calculate the distance between particles
                x_ij = x[i,:] - x[j,:]

                # calculate contributions due to neighbours
                rho_ij = m*Kernel(x_ij, h)

                # add contributions from neighbours 
                rho[i] = rho[i] + rho_ij
                rho[j] = rho[j] + rho_ij

    return rho