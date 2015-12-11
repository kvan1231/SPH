import numpy as np
import matplotlib.pyplot as plt
from accel import *
from density import *
from post_density import *

# def test_diffuse():
M           = 2                         # total mass
R           = 0.25                      # radius
N           = 400.                      # number of particles
dt          = 0.04                      # timestep
n_tstep     = 600                       # number of timesteps
nu          = 1                         # damping
k           = 0.1                       # pressure constant
npoly       = 1.0                       # polytropic index
m           = M/N                       # mass of particles
h           = 0.04/np.sqrt(N/1000.)     # smoothing index
# acceleration due to gravity parameter
g           = 2*k*np.pi**(-1/npoly)*(M*(1+npoly)/(R**2))**(1+1/npoly)/M

# Initialize
# attempt to randomly place particles in a circle
np.random.seed(42)
theta1 = 2*np.pi*np.random.random((N,1))
r1 = R*np.sqrt(np.random.random((N,1)))
pre_x1 = np.array([r1*np.cos(theta1), r1*np.sin(theta1)])
x_T1 = np.array([pre_x1[0].flatten(), pre_x1[1].flatten()])
x1 = x_T1.transpose()
x1[:,0] = x1[:,0] + 0.4

np.random.seed(50)
theta2 = 2*np.pi*np.random.random((N,1))
r2 = R*np.sqrt(np.random.random((N,1)))
pre_x2 = np.array([r2*np.cos(theta2), r2*np.sin(theta2)])
x_T2 = np.array([pre_x2[0].flatten(), pre_x2[1].flatten()])
x2 = x_T2.transpose()
x2[:,0] = x2[:,0] - 0.4

x = np.concatenate((x1,x2))
# x = x[:int(6*len(x)/10)]

# initialize zero velocity
v2 = np.ones((N,2)) * 0.5
v1 = np.zeros((N,2)) * -0.5
v = np.concatenate((v1,v2))

# v = v[:int(6*len(v)/10)]

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
    plt.xlim(-1.4,1.4)
    plt.ylim(-1.4,1.4)
    # plt.plot(x[:N][:,0],x[:N][:,1],'or')
    # plt.plot(x[N:][:,0],x[N:][:,1],'ob')
    plt.scatter(x[:,0], x[:,1], c=rho, marker='o', edgecolors='face', vmin = 0, vmax = 20)
    cb = plt.colorbar()
    cb.set_label('Density')
    plt.grid()
    plt.draw()
    plt.savefig("Fig3/"+"%03d"%i)
    plt.clf()

plt.title("System at N="+str(i)+" Timesteps")    
plt.xlim(-1.4,1.4)
plt.ylim(-1.4,1.4)
# plt.plot(x[:N][:,0],x[:N][:,1],'or')
# plt.plot(x[N:][:,0],x[N:][:,1],'ob')
plt.scatter(x[:,0], x[:,1], c=rho, marker='o', edgecolors='face', vmin = 0, vmax = 20)
cb = plt.colorbar()
cb.set_label('Density')
plt.grid()
plt.draw()

# radii = np.linspace(0,R,100)
# slices = 4
# rho_final= np.zeros((slices,len(radii)))
# for i in range(slices):
#     for j in range(len(radii)):
#         rho_final[i,j] = Find_Density(x, m, h, radii[j], i*2*np.pi/slices)
   
# plt.figure(2)
# for i in range(slices):
#     plt.plot(radii, rho_final[i,:],'b +', label = 'SPH');

# plt.plot(radii,((g/(2*k*(1+npoly)))*(R*R-radii*radii))**npoly,'r--', label='Theory')
# plt.title('Density profile of polytrope after N='+str(n_tstep)+' timesteps')
# plt.xlabel('r');
# plt.ylabel('rho(r)');
# plt.show()
