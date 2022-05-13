import numpy as np

def Q_orig(d, alpha, r, r0):
    exponent = -2 * alpha * (r - r0)
    if exponent > cap:
        return (d / 2) * (1.5 * np.exp(cap) - np.exp(cap // 2))
    return (d / 2) * (1.5 * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0)))
    
def J_orig(d, alpha, r, r0):
    exponent = -2 * alpha * (r - r0)
    cap_local = 350
    if exponent > cap_local:
        return (d / 4) * (1.5 * np.exp(cap_local) - 6 * np.exp(cap_local // 2))
    return (d / 4) * (np.exp(-2 * alpha * (r - r0)) - 6 * np.exp(-alpha * (r - r0)))

class Q:
    def __init__(self, d, alpha, r0):
        self.d = d
        self.alpha = alpha
        self.r0 = r0
        self.cap = 700
    
    def value(self, r):
        d = self.d
        alpha = self.alpha
        r0 = self.r0
        cap = self.cap
        exponent = -2 * alpha * (r - r0)
        if exponent > cap:
            return (d / 2) * (1.5 * np.exp(cap) - np.exp(cap // 2))
        return (d / 2) * (1.5 * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0)))
    
    def der(self, r):
        d = self.d
        alpha = self.alpha
        r0 = self.r0
        return (-d * alpha / 2) * (3 * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0)))

class J:
    def __init__(self, d, alpha, r0):
        self.d = d
        self.alpha = alpha
        self.r0 = r0
        self.cap = 350
        
    def value(self, r):
        d = self.d
        alpha = self.alpha
        r0 = self.r0
        exponent = -2 * alpha * (r - r0)
        cap_local = self.cap
        
        if exponent > cap_local:
            return (d / 4) * (1.5 * np.exp(cap_local) - 6 * np.exp(cap_local // 2))
        return (d / 4) * (np.exp(-2 * alpha * (r - r0)) - 6 * np.exp(-alpha * (r - r0)))

    
    def der(self, r):
        d = self.d
        alpha = self.alpha
        r0 = self.r0
        return (-d * alpha / 2) * (np.exp(-2 * alpha * (r - r0))  - 3 * np.exp(-alpha * (r - r0)))
    
class LEPS_I:
    a = 0.05
    b = 0.30
    c = 0.05
    dAB = dBC = 4.746
    dAC = 3.445
    r0 = 0.742
    alpha = 1.942
    
    def __init__(self):
        pass
    
    def V(self, rAB, rBC):
        
        QAB = Q(self.dAB, self.alpha, self.r0).value(rAB)
        QBC = Q(self.dBC, self.alpha, self.r0).value(rBC)
        rAC = rAB + rBC
        QAC = Q(self.dAC, self.alpha, self.r0).value(rAC)
        
        JAB = J(self.dAB, self.alpha, self.r0).value(rAB)
        JBC = J(self.dBC, self.alpha, self.r0).value(rBC)
        JAC = J(self.dAC, self.alpha, self.r0).value(rAC)
        
        a = self.a
        b = self.b
        c = self.c
        Q_values = (QAB / (1 + a)) + (QBC / (1 + b)) + (QAC / (1 + c)) 
        J_values = (JAB / (1 + a))**2 + (JBC / (1 + b))**2 + (JAC / (1 + c))**2
        J_values = J_values - ((JAB*JBC/((1+a)*(1+b))) + (JBC*JAC/((1+b)*(1+c))) + (JAB*JAC/((1+a)*(1+c))))
        return Q_values - np.sqrt(J_values)
    
    def F(self, rAB, rBC):
        a = self.a
        b = self.b
        c = self.c
        rAC = rAB + rBC
        J_AB = J(self.dAB, self.alpha, self.r0)
        J_BC = J(self.dBC, self.alpha, self.r0)
        J_AC = J(self.dAC, self.alpha, self.r0)
        
        # Computing F_x
        F_x = Q(self.dAB, self.alpha, self.r0).der(rAB) / (1 + a)
        F_x += Q(self.dAC, self.alpha, self.r0).der(rAC) / (1 + c)
        
        comp_x = (2 * J_AB.value(rAB) * J_AB.der(rAB) / ((1 + a)**2) + 2 * J_AC.value(rAC) * J_AC.der(rAC) / ((1 + c)**2))
        comp_x -= (J_AB.der(rAB) * J_BC.value(rBC) / ((1 + a)*(1 + b)) + J_BC.value(rBC) * J_AC.der(rAC) / ((1 + b)*(1 + c)))
        comp_x -= ((J_AB.der(rAB) * J_AC.value(rAC) + J_AC.der(rAC) * J_AB.value(rAB)) / ((1 + a) * (1 + c)))
        
        jAB = J_AB.value(rAB)
        jBC = J_BC.value(rBC)
        jAC = J_AC.value(rAC)
        
        J_values = (jAB / (1 + a))**2 + (jBC / (1 + b))**2 + (jAC / (1 + c))**2
        J_values = J_values - ((jAB*jBC/((1+a)*(1+b))) + (jBC*jAC/((1+b)*(1+c))) + (jAB*jAC/((1+a)*(1+c))))
        comp_x *= 1 / (2 * np.sqrt(J_values))
        F_x -= comp_x
        
        # Computing F_y
        F_y = Q(self.dBC, self.alpha, self.r0).der(rBC) / (1 + b)
        F_y += Q(self.dAC, self.alpha, self.r0).der(rAC) / (1 + c)
        
        comp_y = (2 * J_BC.value(rBC) * J_BC.der(rBC) / ((1 + b)**2) + 2 * J_AC.value(rAC) * J_AC.der(rAC) / ((1 + c)**2))
        comp_y -= (J_AB.value(rAB) * J_BC.der(rBC) / ((1 + a)*(1 + b)) + J_AB.value(rAB) * J_AC.der(rAC) / ((1 + a)*(1 + c)))
        comp_y -= ((J_BC.der(rBC) * J_AC.value(rAC) + J_BC.value(rBC) * J_AC.der(rAC)) / ((1 + b) * (1 + c)))
        
        comp_y *= 1 / (2 * np.sqrt(J_values))
        F_y -= comp_y
        return np.array([-F_x, -F_y])
    

class LEPS_II(LEPS_I):
    rAC = 3.742
    kC = 0.2025
    c = 1.154
    
    def __init__(self):
        super().__init__()
        pass
    
    def V(self, rAB, x):
        
        V_normal = super().V(rAB, self.rAC - rAB)
        return V_normal + 2 * self.kC * (rAB - (self.rAC / 2 - x / self.c))**2
    
    def F(self, rAB, x):
        
        F_I = super().F(rAB, self.rAC - rAB)
        F_x = F_I[0] - F_I[1] - 4 * self.kC * (rAB - (self.rAC / 2 - x / self.c))
        
        F_y = -4 * (self.kC / self.c) * (rAB - (self.rAC / 2 - x / self.c)) 
        return np.array([F_x, F_y])
    
class LEPS_II_Mod(LEPS_II):
    b = 0.03
    
def Z_LEPS(N, T):
    from scipy.integrate import dblquad
    U = LEPS_II().V
    kB = 1
    beta = 1 / (kB * T)
    f = lambda x,y : np.exp(-beta * U(x, y))
    rv = dblquad(f, -np.inf, np.inf, lambda a:-np.inf, lambda a:np.inf)[0]
    return rv
   
def Free_energy(N, T):
    
    kB = 1
    beta = 1 / (kB * T)
    Z = Z_LEPS(N, T)
    f = -beta * (N / 2) * np.log(Z)
    m = 1
    f -= beta * N * np.log(4 * np.pi * beta / m)
    return f
    