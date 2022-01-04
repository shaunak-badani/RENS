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


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


N = 10
y0s = get_y0s(Nsim, N, 1 / (kB * 0.3))


# In[8]:


assert(y0s.shape == (Nsim*2*N,))


# In[9]:


print(y0s.shape[0])


# In[10]:


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


# In[11]:


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


# In[12]:


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
#     p = np.vstack((traj[0], traj[-1]))
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


# In[13]:


def simulate_custom(initial):
    N_t = 10000
    a = initial.shape[0] // 2
    x = initial[:a][:]
    p = initial[a:][:]
    
    t = 0
    dt = 1e-3
    
    traj = [initial]
    while t < t_s:
        T_lamda = T_lambda(t)
        z = (1 / t_s) * (1 / (2 * T_lamda)) * (T_B - T_A)
        factor = np.exp(z * dt / 2)
        p = p * factor + F(x) * (factor - 1) / z
        x = x + p * dt
        p = p * factor + F(x) * (factor - 1) / z
        phase = np.hstack([x, p])
        
        traj.append(phase)
        
        t += dt
    
    traj = np.array(traj)
    return traj


# In[14]:


T_A = 0.3
T_B = 2.0
t_s = 10
Nsim = 1
y0s = get_y0s(Nsim, N, 1 / (kB * T_A))
_, scipy_traj = simulate(y0s)

custom_traj = simulate_custom(y0s)


# In[15]:


print(scipy_traj.shape)
print(custom_traj.shape)


# In[16]:


print(calc_W(scipy_traj, N, Nsim, T_A, T_B))
print(calc_W(custom_traj, N, Nsim, T_A, T_B))


# In[17]:



fig = plt.figure(figsize = (15, 5))


fig.add_subplot(1, 2, 1)
plt.plot(scipy_traj[:, 0], label = 'q')
plt.plot(scipy_traj[:, 10], label = 'p')
plt.title("Scipy Trajectory")
plt.legend(loc='best')
plt.xlabel('Time (t)')

fig.add_subplot(1, 2, 2)
plt.plot(custom_traj[:, 0], label = 'q')
plt.plot(custom_traj[:, 10], label = 'p')
plt.legend(loc='best')
plt.title("Custom integrator traectory")
plt.xlabel('Time (t)')


# In[18]:


def simulate_custom(initial):
    """
    Redefining simulate custom but now with only two phase space vectors being returned
    """
    N_t = 10000
    a = initial.shape[0] // 2
    x = initial[:a][:]
    p = initial[a:][:]
    
    t = 0
    dt = 1e-3
    
    traj = [initial]
    while t < t_s:
        T_lamda = T_lambda(t)
        z = (1 / t_s) * (1 / (2 * T_lamda)) * (T_B - T_A)
        factor = np.exp(z * dt / 2)
        p = p * factor + F(x) * (factor - 1) / z
        x = x + p * dt
        p = p * factor + F(x) * (factor - 1) / z
        t += dt
    phase = np.hstack([x, p])    
    traj.append(phase)
    traj = np.array(traj)
    return traj


# In[19]:


# %%time
# Nsim = 10
# ts_values = np.linspace(0, 100, 25)
# works_forward = np.zeros(shape = (ts_values.size, Nsim))
# for i, ts in enumerate(ts_values):
#     T_A = 0.3
#     T_B = 2.0
#     t_s = ts

#     y0s = get_y0s(Nsim, N, 1 / (kB * T_A))
#     t, forward_traj = simulate(y0s)
#     w_forward = calc_W(forward_traj, N, Nsim, T_A, T_B)
#     works_forward[i] = w_forward
#     print("ts = {} forward done".format(ts))
# # np.savetxt('forward_works_1.dat', works_forward, fmt = '%10.5f')


# In[20]:


# %%time
Nsim = 10
ts_values = np.linspace(0, 100, 25)
works_forward_1 = np.zeros(shape = (ts_values.size, Nsim))
for i, ts in enumerate(ts_values):
    T_A = 0.3
    T_B = 2.0
    t_s = ts

    y0s = get_y0s(Nsim, N, 1 / (kB * T_A))
    forward_traj = simulate_custom(y0s)
    w_forward = calc_W(forward_traj, N, Nsim, T_A, T_B)
    works_forward_1[i] = w_forward
    print("ts = {} forward done".format(ts))
np.savetxt('forward_works_1.dat', works_forward_1, fmt = '%10.5f')


# In[21]:


works_backward_1 = np.zeros(shape = (ts_values.size, Nsim))
for i, ts in enumerate(ts_values):
    T_A = 2.0
    T_B = 0.3
    t_s = ts

    y0s = get_y0s(Nsim, N, 1 / (kB * T_A))
    backward_traj = simulate_custom(y0s)
    w_backward = calc_W(backward_traj, N, Nsim, T_A, T_B)
    works_backward_1[i] = w_backward
    print("ts = {} backward done".format(ts))
np.savetxt('backward_works_1.dat', works_backward_1, fmt = '%10.5f')

