import numpy as np
import matplotlib.pyplot as plt
from accel import *
from density import *
from post_density import *

# def test_diffuse():
M           = 2                         # total mass
R           = 0.75                      # radius
N           = 100.                      # number of particles
dt          = 0.04                      # timestep
n_tstep     = 200                       # number of timesteps
nu          = 0                         # damping
k           = 0.1                       # pressure constant
npoly       = 1.0                       # polytropic index
m           = M/N                       # mass of particles
h           = 0.04/np.sqrt(N/1000.)     # smoothing index
# acceleration due to gravity parameter
g           = 2*k*np.pi**(-1/npoly)*(M*(1+npoly)/(R**2))**(1+1/npoly)/M

# Initialize
# attempt to randomly place particles in a circle
np.random.seed(42)
theta = 2*np.pi*np.random.random((N,1))
r = R*np.sqrt(np.random.random((N,1)))
pre_x = np.array([r*np.cos(theta), r*np.sin(theta)])
x_T = np.array([pre_x[0].flatten(), pre_x[1].flatten()])
x = x_T.transpose()

# initialize zero velocity
v = np.zeros((N,2))

# calculate initial values
rho = Calc_Density(x, m, h)     
P = k*rho**(1+1/npoly)
a = Calc_Accel(x, v, m, rho, P, g, nu, h)

# calculate v at t=-0.5dt for leap frog integration
v_nhalf = v - 0.5*dt*a

# Evolve
plt.figure(1)
for i in range(n_tstep):
    # leapfrog integration
    v_phalf = v_nhalf + a*dt
    x = x + v_phalf*dt
    v = 0.5*(v_nhalf + v_phalf)
    v_nhalf = v_phalf

    rho = Calc_Density(x, m, h)
    P = k*rho**(1+1/npoly)
    a = Calc_Accel(x, v, m, rho, P, g, nu, h)

    plt.title("System at N="+str(i)+" Timesteps")    
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.scatter(x[:,0], x[:,1], c=rho, marker='o', edgecolors='face', vmin = 0, vmax = 3)
    cb = plt.colorbar()
    cb.set_label('Density')
    plt.grid()
    plt.draw()
    plt.savefig("Fig6/"+"%03d"%i)
    plt.clf()

plt.title("System at N="+str(i)+" Timesteps")    
plt.xlim(-1.2,1.2)
plt.ylim(-1.2,1.2)
plt.scatter(x[:,0], x[:,1], c=rho, marker='o', edgecolors='face', vmin = 0, vmax = 3)
cb = plt.colorbar()
cb.set_label('Density')
plt.grid()
plt.draw()

# radii = np.linspace(0,R,N)
# slices = 4
# rho_final= np.zeros((slices,len(radii)))
# for i in range(slices):
#     for j in range(len(radii)):
#         rho_final[i,j] = Find_Density(x, m, h, radii[j], i*2*np.pi/slices)
   
# plt.figure(2)
# for i in range(slices):
#     plt.plot(radii, rho_final[i,:],'b +', label = 'SPH');

# plt.plot(radii,((g*(R*R-radii*radii))/(4*k)),'r--', label='Theory')
# plt.title('Density profile of polytrope after N='+str(n_tstep)+' timesteps')
# plt.xlabel('r');
# plt.ylabel('rho(r)');
# plt.show()
# plt.savefig("rho_profile_2")
#    