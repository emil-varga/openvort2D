import numpy as np
from numpy.random import rand

import taichi as ti

kappa = 9.96e-4


@ti.kernel
def update_velocity_ti(xs: ti.types.ndarray(), ys: ti.types.ndarray(), signs: ti.types.ndarray(),
                       vx: ti.types.ndarray(), vy: ti.types.ndarray(), shifts: ti.types.ndarray()):
    N = xs.shape[0]
    S = shifts.shape[0]
    for j in range(N):
        if signs[j] == 0:
            continue
        vx[j] = 0
        vy[j] = 0
        for k in range(N):
            for xshift in range(S):
                for yshift in range(S):
                    if k == j and shifts[xshift] == 0 and shifts[yshift] == 0:
                        continue
                    x_jk = xs[j] - xs[k] + shifts[xshift]
                    y_jk = ys[j] - ys[k] + shifts[yshift]
                    r2_jk = x_jk**2 + y_jk**2
                    vx[j] += -kappa/2/3.14159/r2_jk*y_jk*signs[k]
                    vy[j] += kappa/2/3.14159/r2_jk*x_jk*signs[k]

@ti.kernel
def annihilate_ti(xs: ti.types.ndarray(), ys: ti.types.ndarray(), 
                  signs: ti.types.ndarray(dtype=ti.int64), shifts: ti.types.ndarray(),
                  a0: float, to_annihilate: ti.types.ndarray()) -> int:
    N = xs.shape[0]
    S = shifts.shape[0]

    for j in range(N):
        to_annihilate[j] = 0

    # ti.loop_config(serialize=True)
    for j in range(N):
        if signs[j] == 0:
            continue
        move_on = False
        for k in range(j+1, N):
            if signs[k] == 0 or signs[j] == signs[k]:
                continue
            for xshift in range(S):
                for yshift in range(S):
                    x_jk = xs[k] - xs[j] + shifts[xshift]
                    y_jk = ys[k] - ys[j] + shifts[yshift]
                    r_jk = ti.sqrt(x_jk**2 + y_jk**2)
                    if r_jk < a0:
                        to_annihilate[j] += 1
                        to_annihilate[k] += 1
                        move_on = True
                        break
                if move_on:
                    break
            if move_on:
                break
    
    OK = True
    for j in range(N):
        if to_annihilate[j] > 1:
            OK = False

    if OK:
        for j in range(N):
            if to_annihilate[j] > 0:
                signs[j] = 0
    else:
        ti.loop_config(serialize=True)
        for j in range(N):
            if signs[j] == 0:
                continue
            move_on = False
            for k in range(N):
                if k == j:
                    continue
                if signs[k] == 0 or signs[j] == signs[k]:
                    continue
                for xshift in range(S):
                    for yshift in range(S):
                        x_jk = xs[k] - xs[j] + shifts[xshift]
                        y_jk = ys[k] - ys[j] + shifts[yshift]
                        r_jk = ti.sqrt(x_jk**2 + y_jk**2)
                        if r_jk < a0:
                            signs[j] = 0
                            signs[k] = 0
                            move_on = True
                            break
                    if move_on:
                        break
                if move_on:
                    break
    return OK
        

class VortexPoints:
    def __init__(self, N, D=1, a0=1e-5, polarization=0, polarization_type='none'):
        self.a0 = a0 # annihilation distance, in cm
        self.N = N
        self.D = D
        self.xs = np.zeros(N)
        self.ys = np.zeros(N)
        self.xs += rand(N)*D
        self.ys += rand(N)*D
        self.vx = np.zeros_like(self.xs)
        self.vy = np.zeros_like(self.ys)
        self.signs = np.ones(N, dtype=int)
        self.signs[N//2:] = -1
        match polarization_type:
            case 'none':
                pass
            case 'jet':
                print(f"initializing polarized {polarization_type}, {polarization}")
                npos = int(0.5*polarization*N)
                nneg = npos
                nrest = N - npos - nneg
                self.xs[:npos] = (rand(npos) + 1)*D/2
                self.signs[:npos] = +1
                self.xs[npos:(npos+nneg)] = rand(nneg)*D/2
                self.signs[npos:(npos+nneg)] = -1
                self.signs[(npos+nneg):(npos+nneg+int(nrest/2))] = +1
                self.signs[(npos+nneg+int(nrest/2)):] = -1
        self.shifts = np.array([-D, 0, D])
        self.to_annihilate = np.zeros(N)
        self.t = 0
        self.omega = 0
        self.A = 0

    def plot(self, ax):
        ixp = self.signs > 0
        ixn = self.signs < 0
        ax.scatter(self.xs[ixp], self.ys[ixp], color='r')
        ax.scatter(self.xs[ixn], self.ys[ixn], color='b')
    
    def update_velocity(self):
        update_velocity_ti(self.xs, self.ys, self.signs, self.vx, self.vy, self.shifts)
    
    def dissipation(self, alpha=0.1, alphap=0):
        for j in range(self.N):
            vx0 = self.vx[j]
            vy0 = self.vy[j]

            self.vx[j] = vx0 + alpha*vy0*self.signs[j] - alphap*vx0
            self.vy[j] = vy0 - alpha*vx0*self.signs[j] - alphap*vy0
    
    def annihilate(self):
        annihilate_ti(self.xs, self.ys, self.signs, self.shifts, self.a0, self.to_annihilate)
    
    def inject(self, npairs):
        stepping = self.D/(2*npairs)
        posy = np.linspace(0, self.D-stepping, npairs) + np.random.randn(npairs)*self.D/100
        negy = np.linspace(stepping, self.D, npairs) + np.random.randn(npairs)*self.D/100

        free = np.sum(self.signs == 0)
        if free > 2*npairs:
            ixfree = np.where(self.signs == 0)[0]
            self.ys[ixfree[:npairs]] = posy
            self.ys[ixfree[npairs:(2*npairs)]] = negy
            self.xs[ixfree[:(2*npairs)]] = self.D/2
            self.signs[ixfree[:npairs]] = 1
            self.signs[ixfree[npairs:(2*npairs)]] = -1
            return
        
        #otherwise we need to expand the arrays
        self.xs = np.append(self.xs, np.zeros(len(posy) + len(negy)) + self.D/2)
        self.ys = np.append(self.ys, posy)
        self.ys = np.append(self.ys, negy)
        
        self.vx = np.append(self.vx, np.zeros(len(posy) + len(negy)))
        self.vy = np.append(self.vy, np.zeros(len(posy) + len(negy)))

        self.signs = np.append(self.signs, np.ones_like(posy, dtype=int))
        self.signs = np.append(self.signs, -np.ones_like(negy, dtype=int))

        self.N += 2*npairs
        self.to_annihilate = np.zeros(self.N)

    def step(self, dt):
        for j in range(self.N):
            self.xs[j] += self.vx[j]*dt
            self.ys[j] += self.vy[j]*dt
        self.t += dt
    
    def coerce(self):
        for j in range(self.N):
            if self.xs[j] > self.D:
                self.xs[j] -= self.D
            if self.ys[j] > self.D:
                self.ys[j] -= self.D
            if self.xs[j] < 0:
                self.xs[j] += self.D
            if self.ys[j] < 0:
                self.ys[j] += self.D
    
    def check(self):
        v = sum(self.signs)
        if abs(v) > 0:
            raise RuntimeError("nonzero vorticity")
