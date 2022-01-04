#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


# In[2]:


kB = 1


# In[3]:


def pot_energy(x):
    if x < -1.25:
        return (4 * (np.pi**2)) * (x + 1.25)**2
    
    if x >= -1.25 and x <= -0.25:
        return 2 * (1 + np.sin(2 * np.pi * x))
        
    if x >= -0.25 and x <= 0.75:
        return 3 * (1 + np.sin(2 * np.pi * x))
                  
    if x >= 0.75 and x <= 1.75:
        return 4 * (1 + np.sin(2 * np.pi * x))
                  
    # if x >= 1.75:
    return 64 * ((x - 7 / 4) ** 2)

def force(x):
    if x <= -1.25:
        return (-8 * (np.pi)**2) * (x + 1.25)

    if x > -1.25 and x <= -0.25:
        return -4 * np.pi * np.cos(2 * np.pi * x)

    if x >= -0.25 and x <= 0.75:
        return -6 * np.pi * np.cos(2 * np.pi * x)

    if x >= 0.75 and x <= 1.75:
        return -8 * np.pi * np.cos(2 * np.pi * x)

    if x >= 1.75:
        return (-16 * (np.pi)**2) * (x - 1.75)


# In[85]:


def U(x):
    u = np.zeros_like(x)
    u[x < -1.25] = 4 * np.pi**2 * (x[x < -1.25] + 1.25)**2
    
    ind = np.logical_and(x >= -1.25, x < -0.25)
    u[ind] = 2 * (1 + np.sin(2*np.pi*x[ind]))
    
    ind = np.logical_and(x >= -0.25, x < 0.75)
    u[ind] = 3 * (1 + np.sin(2*np.pi*x[ind]))
    
    ind = np.logical_and(x >= 0.75, x < 1.75)
    u[ind] = 4 * (1 + np.sin(2*np.pi*x[ind]))
    
    ind = (x >= 1.75)
    u[ind] = 64 * (x[ind] - 1.75)**2
    
    return u.sum()


# In[4]:


x = np.linspace(-2, 2.25, 1000)
a = np.array([pot_energy(i) for i in x])
b = np.array([force(i) for i in x])

fig = plt.figure(figsize = (15, 5))
fig.add_subplot(1, 2, 1)
plt.plot(x, a)
plt.xlabel(r'$x$', fontsize = 15)
ay = plt.ylabel(r'$U(x)$', fontsize = 15)
ay.set_rotation(0)

fig.add_subplot(1, 2, 2)
plt.plot(x, b)
plt.xlabel(r'$x$', fontsize = 15)
ay = plt.ylabel(r'$F(x)$', fontsize = 15)
ay.set_rotation(0)


# In[58]:


Nsim = 100


def get_y0s(nsim, N, beta):
    """
    Arguments:
    nsim : number of simulations to be performed
    
    Returns :
    y0 : numpy array of shape (nsim * 2 * N)
    This is because scipy integrate only accepts one dimensional vector
    First half of y0 denotes x, other half p
    y0[i] denotes the ith phase space vector 
    """
    sigma_p = 1 / np.sqrt(beta)
    linsp = np.linspace(-2, 2.25, 10000)
    u = np.array([pot_energy(i) for i in linsp])
    prob = np.exp(- beta * u)
    prob /= prob.sum()

    x = np.random.choice(linsp, size = (Nsim * N), p = prob)
    p = np.random.normal(size = (Nsim * N), scale=sigma_p)
    y0 = np.hstack((x, p))
    return y0


# In[59]:


N = 10
y0s = get_y0s(Nsim, N, 1 / (kB * 0.3))


# In[60]:


assert(y0s.shape == (Nsim*2*N,))


# In[65]:


print(y0s.shape[0])


# In[67]:


x = y0s[:y0s.shape[0]//2]
p = y0s[y0s.size//2:]

fig = plt.figure(figsize = (15, 5))
fig.add_subplot(1, 2, 2)
probs, be = np.histogram(p, bins = 20, density = True)
coords = (be[1:] + be[:-1])/2
plt.plot(coords, probs)

fig.add_subplot(1, 2, 1)
probs, be = np.histogram(x, bins = 50, density = True)
coords = (be[1:] + be[:-1])/2
plt.plot(coords, probs)


# In[48]:


def F(x):
    f = np.zeros_like(x)
    f[x < -1.25] = -8 * np.pi**2 * (x[x < -1.25] + 1.25)
    
    ind = np.logical_and(x >= -1.25, x < -0.25)
    f[ind] = -4 * np.pi * np.cos(2 * np.pi * x[ind])
    
    ind = np.logical_and(x >= -0.25, x < 0.75)
    f[ind] = -6 * np.pi * np.cos(2 * np.pi * x[ind])
    
    ind = np.logical_and(x >= 0.75, x < 1.75)
    f[ind] = -8 * np.pi * np.cos(2 * np.pi * x[ind])
    
    ind = (x >= 1.75)
    f[ind] = -16 * np.pi**2 * (x[ind] - 1.75)
    
    return f


# In[124]:


def T_lambda(t):
    return T_A + (T_B - T_A) * (t / t_s)

def dy_dt(y, t):
    x = y[:y0s.shape[0]//2]
    p = y[y0s.shape[0]//2:]
    dx_dt = p 
    T_lamda = T_lambda(t)
    dT_dt = (T_B - T_A) * (1 / t_s)
    dp_dt = F(x) + dT_dt * p / (2 * T_lamda)
    return np.hstack((dx_dt,dp_dt))

def simulate(initial):
    N_t = 10000
    t = np.linspace(0, t_s, N_t)
    traj = scipy.integrate.odeint(dy_dt,initial,t)
    return t, traj

def calc_W(traj, N, nsim, TA, TB):
    """
    Arguments : 
    traj : Trajectory of simulations of size (time, nsim * N * 2)
    N : number of particles
    nsim : number of simulations performed
    
    Returns :
    Work array of size (nsim) containing the work done over nsim simulations
    """
    assert(traj.shape[1] == (nsim * N * 2))
    initial = traj[0].reshape(nsim, 2 * N)
    final = traj[-1].reshape(nsim, 2 * N)
    
    pot_I = np.array([U(i) for i in initial[:, :N]])
    kin_I = np.sum(initial[:, N:]**2, axis = 1)
    H_I = pot_I + kin_I
    h_i = H_I / (kB * TA)
    
    pot_F = np.array([U(i) for i in final[:, :N]])
    kin_F = np.sum(final[:, N:]**2, axis = 1)
    H_F = pot_F + kin_F
    h_f = H_F / (kB * TB)
    
    w = h_f - h_i
    return w


# In[143]:


Nsim = 10
works_forward = np.zeros(shape = (ts_values.size, Nsim))
ts_values = np.linspace(0, 100, 25)
for i, ts in enumerate(ts_values):
    T_A = 0.3
    T_B = 2.0
    t_s = ts

    y0s = get_y0s(Nsim, N, 1 / (kB * T_A))
    t, forward_traj = simulate(y0s)
    w_forward = calc_W(forward_traj, N, Nsim, T_A, T_B)
    works_forward[i] = w_forward
    print("ts = {} forward done".format(ts))
np.savetxt('forward_works.dat', works_forward, fmt = '%10.5f')


# In[146]:


works_backward = np.zeros(shape = (ts_values.size, Nsim))
for i, ts in enumerate(ts_values):
    T_A = 2.0
    T_B = 0.3
    t_s = ts

    y0s = get_y0s(Nsim, N, 1 / (kB * T_A))
    t, backward_traj = simulate(y0s)
    w_backward = calc_W(backward_traj, N, Nsim, T_A, T_B)
    works_backward[i] = w_backward
    print("ts = {} backward done".format(ts))
np.savetxt('backward_works.dat', works_backward, fmt = '%10.5f')


# ## Analyzing done simulations

# In[11]:


works_forward = np.loadtxt('forward_works.dat')
works_backward = np.loadtxt('backward_works.dat')


# In[149]:


w = works_forward.mean(axis = 1) + works_backward.mean(axis = 1)
plt.plot(ts_values, w)


# In[12]:


average_w = works_forward.mean(axis = 1) + works_backward.mean(axis = 1)


# In[13]:


plt.plot(ts_values, average_w)
plt.xlabel(r'$\tau$ (OR $t_s$)', fontsize = 20)
ay = plt.ylabel(r'$<w>$', labelpad = 40, fontsize = 20)
ay.set_rotation(0)
plt.title(r'Average work vs $\tau$')
plt.savefig('Work_dist.png', bbox_inches='tight')

