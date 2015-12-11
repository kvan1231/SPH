import numpy as np

def d_Kernel(r,h):
    """
    Returns the derivative of the kernel function, which is the gradient in 1D

    Input: position vector r, scaling length h
    Output: derivative of weight dW
    """
    try:
        x = r[0,:]
        y = r[1,:]
    except:
        x = r[0]
        y = r[1]

    # q = np.sqrt(x*x+y*y)/h
    # # print(q)

    # C = -5.*r/(14.*(q*np.pi*h*h*h*h))
    # # print(C)

    # if q < 1:
    #     dW_p = ((3.*(2.-q)**2) - 12.*(1.-q)**2)
    # elif (q >= 1) and (q < 2):
    #     dW_p = (3.*(2.-q)**2)
    # else:
    #     dW_p = 0

    # dW = C*dW_p

    exp_term = np.exp(-(x*x+y*y)/(h*h))
    dWx = - 2.0 * x *exp_term / ((h*h*h*h) * (np.pi))
    dWy = - 2.0 * y *exp_term / ((h*h*h*h) * (np.pi))

    dW = np.array([dWx, dWy])

    return dW