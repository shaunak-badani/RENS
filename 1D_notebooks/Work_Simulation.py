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


# In[11]:


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


# In[23]:


class RENS_Simulation:
    
    def __init__(self, ts, T_A = 0.3, T_B = 2.0):
        self.T_A = T_A
        self.T_B = T_B
        self.beta = 1 / (kB * T_A)
        self.t_s = ts
        
        self.linsp = np.linspace(-2, 2.25, 10000)
        self.u = np.array([pot_energy(i) for i in self.linsp])
        
    def get_y0(self):
        sigma_p = 1/np.sqrt(self.beta)
        prob = np.exp(- self.beta * self.u)
        prob /= prob.sum()
        
        x = np.random.choice(self.linsp, 1, p = prob)[0]
        p = np.random.normal(scale=sigma_p)
        self.y0 = np.array([x, p])
        
    def T_lambda(self, t):
        return self.T_A + (self.T_B - self.T_A)*(t/self.t_s)
        
    def dy_dt(self, y, t):
        x, p = y
        # assume m = 1
        dx_dt = p 
        T_lambda = self.T_lambda(t)
        dT_dt = (self.T_B - self.T_A) * (1 / self.t_s)
        dp_dt = force(x) + dT_dt * p / (2 * T_lambda)
        return np.array([dx_dt,dp_dt])
    
    def calc_h(self, y, t, T):
        H = 0.5*y[1]**2 + pot_energy(y[0])
        h = H / (kB * T)
        return h
    
    def calc_W(self):
        y0 = self.traj[0]
        y1 = self.traj[-1]
        t = self.t
        W = self.calc_h(y1,t[-1], self.T_B) - self.calc_h(y0, t[0], self.T_A)
        return W
        
        
    def simulate(self):
        N_t = 10000
        t = np.linspace(0,self.t_s,N_t)
        traj = scipy.integrate.odeint(self.dy_dt,self.y0,t)
        self.traj = traj
        self.t = t
        
        self.w = self.calc_W()


# In[24]:


a = RENS_Simulation(ts = 10)
a.get_y0()
a.simulate()


# In[25]:


fig = plt.figure(figsize = (15, 5))

plt.plot(a.t, a.traj[:, 0], label = 'x')
plt.plot(a.t, a.traj[:, 1], label = 'p')
plt.legend(loc = 'best')


# In[30]:


ts_values = np.linspace(0, 100, 25)


# In[38]:


Nsim = 100000


works_forward = np.zeros(shape = (ts_values.size, Nsim))
for i, ts in enumerate(ts_values):
    forward = RENS_Simulation(T_A = 0.3, T_B = 2.0, ts = ts)
    for j in range(Nsim):
        forward.get_y0()
        forward.simulate()
        works_forward[i, j] = forward.w


# In[39]:


np.savetxt('forward_works.dat', works_forward, fmt = '%10.5f')


# In[40]:


works_backward = np.zeros(shape = (ts_values.size, Nsim))
for i, ts in enumerate(ts_values):
    backward = RENS_Simulation(T_A = 2.0, T_B = 0.3, ts = ts)
    for j in range(Nsim):
        backward.get_y0()
        backward.simulate()
        works_backward[i, j] = backward.w


# In[27]:


np.savetxt('backward_works.dat', works_backward, fmt = '%10.5f')


# ## Analyzing done simulations

# In[ ]:




