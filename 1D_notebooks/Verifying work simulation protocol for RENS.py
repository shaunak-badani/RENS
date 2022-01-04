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


class RENS_Simulation:
    
    def __init__(self, T_A = 0.3, T_B = 2.0, ts = 10):
        self.T_A = T_A
        self.T_B = T_B
        self.beta = 1 / (kB * T_A)
        self.t_s = ts
        
    def get_y0(self):
        sigma_p = 1/np.sqrt(self.beta)
        prob = np.exp(- self.beta * u)
        prob /= prob.sum()
        x = np.random.choice(x_vals, 1, p = prob)[0]
        p = np.random.normal(scale=sigma_p)
        self.y0 = np.array([x, p])
        
    def T_lambda(self, t):
        return self.T_A + (self.T_B - self.T_A)*(t/self.t_s)
        
    def dy_dt(self, y, t):
        dx_dt = y[1]
        T_lambda = self.T_lambda(t)
        dT_dt = (self.T_B - self.T_A) * (1 / self.t_s)
        dp_dt = force(y[0]) + dT_dt * y[1] / (2 * T_lambda)
        return np.array([dx_dt,dp_dt])
    
    def calc_h(self, y, t, T):
        H = 0.5*y[1]**2 + pot_energy(y[0])
        h = H / (kB * T)
        return h
    
    def calc_W(self):
        y0 = self.res[0]
        y1 = self.res[-1]
        t = self.t
        W = self.calc_h(y1,t[-1], self.T_B) - self.calc_h(y0, t[0], self.T_A)
        return W
        
        
    def simulate_scipy(self):
        N_t = 10000
        t = np.linspace(0,self.t_s,N_t)
        res = scipy.integrate.odeint(self.dy_dt,self.y0,t)
        self.res = res
        self.t = t
        
        self.w = self.calc_W()
        
    def simulate_vv(self):
        dt = 1e-3
        traj = []
        t = 0
        x, p = self.y0
        times = []
        while t < self.t_s:
            dT_dt = (self.T_B - self.T_A) * (1 / self.t_s)
            T_lambda = self.T_lambda(t)
            z = dT_dt / (2 * T_lambda)
            p = p * np.exp(z * dt / 2) + force(x) * (np.exp(z * dt / 2) - 1) / (z)
            x = x + p * dt
            p = p * np.exp(z * dt / 2) + force(x) * (np.exp(z * dt / 2) - 1) / (z)
            t += dt
            times.append(t)
            traj.append(np.array([x, p]))
        self.t = np.array(times)
        self.traj = np.array(traj)


# In[5]:


ini_phase = np.array([-0.25, 1.0])
a = RENS_Simulation()
a.y0 = ini_phase
a.simulate_scipy()


# In[6]:


ini_phase = np.array([-0.25, 1.0])
b = RENS_Simulation()
b.y0 = ini_phase
b.simulate_vv()


# In[7]:


fig = plt.figure(figsize = (15, 5))

fig.add_subplot(1, 2, 1)
plt.plot(a.t, a.res[:, 0], label = 'x')
plt.plot(a.t, a.res[:, 1], label = 'p')
plt.legend(loc = 'best')

fig.add_subplot(1, 2, 2)
plt.plot(b.t, b.traj[:, 0], label = 'x')
plt.plot(b.t, b.traj[:, 1], label = 'p')
plt.legend(loc = 'best')


# ## Simulating the step like protocol

# In[5]:


class NoseHoover():

    def __init__(self, dt, freq = 50, M = 2, n_c = 1, T = 0.3):
        self.n_c = n_c
        self.M = M
        self.vxi = np.zeros(M)
        self.xi = np.zeros(M)
        self.dt = dt
        self.num_particles = 1
        self.T = T
        self.Q = np.full(M, kB * T / freq**2)
        self.Q[0] *= self.num_particles
        self.w = np.array([0.2967324292201065,  0.2967324292201065, -0.1869297168804260, 0.2967324292201065, 0.2967324292201065])


    def universe_energy(self, KE, PE):
        total_universe_energy = (self.num_particles * self.xi[0] + self.xi[1:].sum()) * kB * self.T
        total_universe_energy += 0.5 * np.sum(self.Q * self.vxi**2)
        total_universe_energy += KE + PE
        return total_universe_energy

    def step(self, m, v):
        
        N_f = self.num_particles
        T = self.T
        M = self.M
        n_c = self.n_c
        n_ys = self.w.shape[0]
        SCALE = 1.0
        KE2 = np.sum(m * v**2)        
        
        
        for i in range(n_c):
            for w_j in self.w:
                delta = (w_j * self.dt / n_c)

                
                G_M = (self.Q[M - 2] * self.vxi[M - 2]**2 - kB * T) / self.Q[M - 1]
                self.vxi[M - 1] += (delta / 4) * G_M

                for j in range(M - 2, 0, -1):
                    self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])
                    G_j = (self.Q[j - 1] * self.vxi[j-1]**2 - kB * T) / self.Q[j]
                    self.vxi[j] += G_j * delta / 4
                    self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])

                self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1]) 
                G_1 = (KE2 - N_f * kB * T) / self.Q[0]
                self.vxi[0] += (delta / 4) * G_1 
                self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1])

                # UPDATE xi and v_new
                self.xi += (delta/2) * self.vxi
                SCALE_FACTOR = np.exp(-delta / 2 * self.vxi[0])
                SCALE *= SCALE_FACTOR
                KE2 *= SCALE_FACTOR * SCALE_FACTOR

                # REVERSE
                self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1]) 
                G_1 = (KE2 - N_f * kB * T) / self.Q[0]
                self.vxi[0] += (delta / 4) * G_1 
                self.vxi[0] *= np.exp(-delta / 8 * self.vxi[1])

                for j in range(1, M - 1):
                    self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])
                    G_j = (self.Q[j - 1] * self.vxi[j-1]**2 - kB * T) / self.Q[j]
                    self.vxi[j] += G_j * delta / 4
                    self.vxi[j] *= np.exp(-delta / 8 * self.vxi[j + 1])

                G_M = (self.Q[M - 2] * self.vxi[M - 2]**2 - kB * T) / self.Q[M - 1]
                self.vxi[M - 1] += (delta / 4) * G_M
        
        v_new = v*SCALE
        return v_new
        


# In[6]:


dt = 1e-3
a = NoseHoover(dt = dt)


# In[7]:


class Step_Rens(RENS_Simulation):
    
    def setup_steps(self, delta_t = 0.1):
        self.delta_t = delta_t
        self.n = self.t_s / self.delta_t
        self.nsteps = self.t_s / (dt)
        self.nht = NoseHoover(dt = 1e-3, T = self.T_A)
        self.k = self.nsteps / self.n
        self.delta_lambda = 1 / (self.n + 1)
        self.w = 0
        
        
    def lamda(self, t):
        import math
        l = (t / dt)
#         print(l, self.k)
        return self.delta_lambda * (math.ceil(l / self.k))

    def T_lamda(self, t):
        l = self.lamda(t)
        return self.T_A + l * (self.T_B - self.T_A)
    
    def calc_h(self, x, p, T):
        H = pot_energy(x) + p**2/2
        return H / (kB * T)
        
    def simulate_vv(self):
        dt = 1e-3
        traj = []
        t = 0
        x, p = self.y0
        times = []
        i = 0
        while t <= self.t_s:
            if i % (self.k) == 0:
                T_next = self.T_lamda(t + dt)
                T_prev = self.T_lamda(t)
                p *= np.sqrt(T_next / T_prev)

                self.w += self.calc_h(x, p, T_next) - self.calc_h(x, p, T_prev)
    
                i += 1
                t += dt
                self.nht.T = T_next
                continue
            p = self.nht.step(1, p)
            p = p + force(x) * dt 
            x = x + p * dt
            p = p + force(x) * dt
            p = self.nht.step(1, p)
            
            t += dt
            i += 1
            times.append(t)
            traj.append(np.array([x, p]))
        self.t = np.array(times)
        self.traj = np.array(traj)
    
    
    


# In[9]:


k = Step_Rens(ts = 10)
k.setup_steps(delta_t = 2)

t = np.linspace(0, k.t_s + dt, int(k.t_s / (dt)))

lambd = np.array([k.lamda(i) for i in t])


# In[10]:


plt.plot(t, lambd)


# In[11]:


k = Step_Rens(ts = 10)
k.setup_steps(delta_t = 2)
k.y0 = ini_phase
k.simulate_vv()


# In[15]:


x_vals = np.linspace(-2, 2.25, 1000)
u = np.array([pot_energy(i) for i in x_vals])
print(ts_values)


# In[17]:


Nsim = 10
ts_values = np.logspace(0, 2, 5)[:3]

works_a = np.zeros(shape = (ts_values.size, Nsim))
for i, ts in enumerate(ts_values):
    a = Step_Rens(T_A = 0.3, T_B = 2.0, ts = ts)
    for j in range(Nsim):
        a.get_y0()
        a.setup_steps()
        a.simulate_vv()
        works_a[i, j] = a.w


# In[22]:


np.savetxt('works_a.dat', works_a, fmt = '%10.5f')


# In[26]:


works_b = np.zeros(shape = (ts_values.size, Nsim))
for i, ts in enumerate(ts_values):
    b = Step_Rens(T_A = 2.0, T_B = 0.3, ts = ts)
    for j in range(Nsim):
        b.get_y0()
        b.setup_steps()
        b.simulate_vv()
        works_b[i, j] = b.w


# In[27]:


np.savetxt('works_b.dat', works_b, fmt = '%10.5f')


# In[29]:


prob, be = np.histogram(works_b, bins = 25, density = True)
coords = (be[1:] + be[:-1]) / 2
plt.plot(coords, prob)

prob, be = np.histogram(works_a, bins = 25, density = True)
coords = (be[1:] + be[:-1]) / 2
plt.plot(-coords, prob)


# In[22]:


print(Z_A, Z_B)


# In[21]:


Z_A = scipy.integrate.quad(lambda x : np.exp(-a.beta * pot_energy(x)), -np.inf, np.inf)[0]
Z_B = scipy.integrate.quad(lambda x : np.exp(-pot_energy(x) / (kB * a.T_B)), -np.inf, np.inf)[0]
F_A =  -(kB * a.T_A) * np.log(Z_A)
F_B =  -(kB * a.T_B) * np.log(Z_B)


# In[23]:


delta_f = F_B - F_A


# In[28]:


prob, be = np.histogram(works_b, bins = 50, density = True)
coords = (be[1:] + be[:-1]) / 2
plt.plot(coords, prob)

prob, be = np.histogram(works_a, bins = 50, density = True)
coords = (be[1:] + be[:-1]) / 2
plt.plot(-coords, prob)
plt.axvline(x = -delta_f)
plt.title('WTF', fontsize = 20)


# In[33]:


W_x = -1 / (a.beta) * np.log(np.mean(np.exp(- a.beta * works_a)))
W_a = np.mean(works_a)


# In[34]:


print(W_x, W_a, delta_f)


# In[ ]:




