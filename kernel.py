import numpy as np

def Kernel(r,h):
    """
    Return the value for the Kernal at various r position
    
    Input: position vector r, scaling length h
    Output: weight W
    """
    try:
        x = r[0,:]
        y = r[1,:]
    except:
        x = r[0]
        y = r[1]

    q = np.sqrt(x*x+y*y)/h

    C = 5. / (14.*(np.pi*h*h))

    if q < 1:
        W_p = (2.-q)**3 - 4.*(1.-q)**3
    elif (q >= 1) and (q < 2):
        W_p = (2.-q)**3
    else:
        W_p = 0


    W = C * W_p

    exp_term = np.exp(-(x*x+y*y)/(h*h))
    W = exp_term / (h*h*(np.pi))
    
    return W