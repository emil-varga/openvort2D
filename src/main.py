import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

import taichi as ti

ti.init(ti.gpu)

N = 50

kappa = 9.96e-4
a0 = 1e-4

class VortexPoints:
    def __init__(self, N, D=1):
        self.N = N
        self.D = D
        self.xs = np.zeros(N)
        self.ys = np.zeros(N)
        self.xs += randn(N)*D
        self.ys += randn(N)*D
        self.vx = np.zeros_like(self.xs)
        self.vy = np.zeros_like(self.ys)
        self.signs = np.ones(N)
        self.signs[N//2:] = -1

    def plot(self, ax):
        ixp = self.signs > 0
        ixn = self.signs < 0
        ax.scatter(self.xs[ixp], self.ys[ixp], color='r')
        ax.scatter(self.xs[ixn], self.ys[ixn], color='b')
    
    def update_velocity(self):
        for j in range(self.N):
            if self.signs[j] == 0:
                continue
            self.vx[j] = 0
            self.vy[j] = 0
            for k in range(self.N):
                for xshift in [-self.D, 0, self.D]:
                    for yshift in [-self.D, 0, self.D]:
                        if k == j and xshift == 0 and yshift == 0:
                            continue
                        x_jk = self.xs[k] - self.xs[j] + xshift
                        y_jk = self.ys[k] - self.ys[j] + yshift
                        r_jk = np.sqrt(x_jk**2 + y_jk**2)
                        self.vx[j] += -kappa/4/np.pi/r_jk**2*y_jk*self.signs[k]
                        self.vy[j] += kappa/4/np.pi/r_jk**2*x_jk*self.signs[k]
    
    def update_velocity_vector(self):
        for j in range(self.N):
            if self.signs[j] == 0:
                continue
            self.vx[j] = 0
            self.vy[j] = 0
            # vxk = 0
            # vyk = 0
            for xshift in [-self.D, 0, self.D]:
                for yshift in [-self.D, 0, self.D]:
                    x_jk = self.xs[j] - self.xs + xshift
                    y_jk = self.ys[j] - self.ys + yshift
                    r_jk = np.sqrt(x_jk**2 + y_jk**2)
                    vxk = -kappa/4/np.pi/r_jk**2*y_jk*self.signs
                    vyk = kappa/4/np.pi/r_jk**2*x_jk*self.signs
                    if xshift == 0 and yshift == 0:
                        vxk[j] = 0
                        vyk[j] = 0
                    self.vx[j] += np.sum(vxk)
                    self.vy[j] += np.sum(vyk)

                    if any(np.isnan(self.vx)):
                        raise ValueError("nan")
    
    def dissipation(self, alpha=0.1):
        for j in range(self.N):
            vx0 = self.vx[j]
            vy0 = self.vy[j]

            self.vx[j] += alpha*vy0*self.signs[j]
            self.vy[j] -= alpha*vx0*self.signs[j]
    
    def annihilate(self, a0=a0):
        for j in range(self.N):
            if self.signs[j] == 0:
                continue
            for k in range(self.N):
                if k == j:
                    continue
                if self.signs[j]*self.signs[k] >= 0:
                    continue
                for xshift in [-self.D, 0, self.D]:
                    for yshift in [-self.D, 0, self.D]:
                        x_jk = self.xs[k] - self.xs[j] + xshift
                        y_jk = self.ys[k] - self.ys[j] + yshift
                        r_jk = np.sqrt(x_jk**2 + y_jk**2)
                        if r_jk < a0:
                            self.signs[j] = 0
                            self.signs[k] = 0
    
    def inject(self, npairs):
        stepping = self.D/(2*npairs)
        posy = np.linspace(0, self.D-stepping, npairs) + np.random.randn(npairs)*self.D/100
        negy = np.linspace(stepping, self.D, npairs) + np.random.randn(npairs)*self.D/100

        free = np.sum(self.signs == 0)
        if free > 2*npairs:
            ixfree = np.where(self.signs == 0)[0]
            self.ys[ixfree[:len(posy)]] = posy
            self.ys[ixfree[len(posy):(len(posy) + len(negy))]] = negy
            self.xs[ixfree] = self.D/2
            self.signs[ixfree[:len(posy)]] = 1
            self.signs[ixfree[len(posy):(len(posy) + len(negy))]] = -1
            return
        
        #otherwise we need to expand the arrays
        self.xs = np.append(self.xs, np.zeros(len(posy) + len(negy)) + self.D/100)
        self.ys = np.append(self.ys, posy)
        self.ys = np.append(self.ys, negy)
        
        self.vx = np.append(self.vx, np.zeros(len(posy) + len(negy)))
        self.vy = np.append(self.vy, np.zeros(len(posy) + len(negy)))

        self.signs = np.append(self.signs, np.ones_like(posy))
        self.signs = np.append(self.signs, -np.ones_like(negy))

        self.N += len(posy) + len(negy)


    def step(self, dt):
        for j in range(self.N):
            self.xs[j] += self.vx[j]*dt
            self.ys[j] += self.vy[j]*dt
    
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
            
   
if __name__ == '__main__':
    D = 1e-2
    vp = VortexPoints(N, D)
    fig, ax = plt.subplots()
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    ax.set_aspect('equal')
    pos, = ax.plot(vp.xs[vp.signs > 0], vp.ys[vp.signs > 0], 'o', color='r')
    neg, = ax.plot(vp.xs[vp.signs < 0], vp.ys[vp.signs < 0], 'o', color='b')
    t = 0
    dt = 0.0001
    last_inject = t
    while True:
        if t - last_inject > 0.001:
            vp.inject(3)
            vp.annihilate()
            last_inject = t
            print("injecting")
        vp.update_velocity_vector()
        vp.dissipation(0.01)
        vp.step(dt)
        vp.annihilate()
        vp.coerce()
        t += dt
        pos.set_xdata(vp.xs[vp.signs > 0])
        pos.set_ydata(vp.ys[vp.signs > 0])
        neg.set_xdata(vp.xs[vp.signs < 0])
        neg.set_ydata(vp.ys[vp.signs < 0])
        plt.pause(0.01)
        N = abs(vp.signs).sum()
        print(t, N, vp.N)
        if N == 0:
            break