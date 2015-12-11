import numpy as np
from d_kernel import *

def Calc_Accel(x, v, m, rho, P, g, nu, h):
    """
    Calculates acceleration of each particle based on inputted parameters

    Input: position x, velocity v, mass m, density rho, 
           pressure P of particle and smoothing length h
    Outputs: acceleration of particle
    """

    # Number of particles
    N = len(x)

    # Initialize any variables needed
    a = np.zeros((N,2))

    for i in range(N):
        a[i,:] = a[i,:] - v[i,:]*nu - g*x[i,:]
        for j in range(i+1,N):
            if j != i:
                # calculate the distance between particles
                x_ij = x[i,:] - x[j,:]

                # calculate effect of pressure
                dW = d_Kernel(x_ij, h)
                pressure = -m*((P[i]/(rho[i]*rho[i])) + (P[j]/(rho[j]*rho[j])))*dW

                # calculate acceleration
                a[i,:] = a[i,:] + pressure
                a[j,:] = a[j,:] - pressure

    return a