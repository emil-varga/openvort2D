import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

ti.init('ti.gpu')

N = 200

kappa = 9.96e-4
a0 = 1e-4

class VortexPoints:
    def __init__(self, N, D=1):
        self.N = N
        self.D = D
        self.pos = ti.field(dtype=float, shape=(2, self.N))
        self.vel = ti.field(dtype=float, shape=(2, self.N))
        self.xs = np.random.rand(N)*D
        self.ys = np.random.rand(N)*D
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
                if self.signs[j]*self.signs[k] >= 0:
                    continue
                for xshift in [-self.D, 0, self.D]:
                    for yshift in [-self.D, 0, self.D]:
                        if k == j and xshift == 0 and yshift == 0:
                            continue
                        x_jk = self.xs[k] - self.xs[j] + xshift
                        y_jk = self.ys[k] - self.ys[j] + yshift
                        r_jk = np.sqrt(x_jk**2 + y_jk**2)
                        if r_jk < a0:
                            self.signs[j] = 0
                            self.signs[k] = 0

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
    vp = VortexPoints(N, 1e-2)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    pos, = ax.plot(vp.xs[vp.signs > 0], vp.ys[vp.signs > 0], 'o', color='r')
    neg, = ax.plot(vp.xs[vp.signs < 0], vp.ys[vp.signs < 0], 'o', color='b')
    while True:
        vp.update_velocity_vector()
        vp.dissipation(0.01)
        vp.step(0.0001)
        vp.annihilate()
        vp.coerce()
        pos.set_xdata(vp.xs[vp.signs > 0])
        pos.set_ydata(vp.ys[vp.signs > 0])
        neg.set_xdata(vp.xs[vp.signs < 0])
        neg.set_ydata(vp.ys[vp.signs < 0])
        plt.pause(0.01)
        N = abs(vp.signs).sum()
        print(N)
        if N == 0:
            break