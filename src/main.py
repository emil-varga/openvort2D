import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import time
import argparse

import taichi as ti
import os

ti.init(ti.gpu)

N = 50

kappa = 9.96e-4
a0 = 1e-4


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
                  a0: float, to_annihilate: ti.types.ndarray()):
    N = xs.shape[0]
    S = shifts.shape[0]

    for j in range(N):
        to_annihilate[j] = 0

    # ti.loop_config(serialize=True)
    for j in range(N):
        if signs[j] == 0:
            continue
        for k in range(j+1, N):
            if signs[k] == 0 or signs[j] == signs[k]:
                continue
            for xshift in range(S):
                for yshift in range(S):
                    x_jk = xs[k] - xs[j] + shifts[xshift]
                    y_jk = ys[k] - ys[j] + shifts[yshift]
                    r_jk = ti.sqrt(x_jk**2 + y_jk**2)
                    if r_jk < a0:
                        # print("Add ", j, k)
                        to_annihilate[j] += 1
                        to_annihilate[k] += 1
    
    OK = True
    # total = 0
    # ti.loop_config(serialize=True)
    for j in range(N):
        # if to_annihilate[j] > 0:
        #     total += 1
        if to_annihilate[j] > 1:
            # print("NotOK")
            OK = False
            # break
    
    # print(OK, total)
    if OK:
        # s = 0
        # for j in range(N):
        #     s += signs[j]
        # print('before', s)
        for j in range(N):
            if to_annihilate[j] > 0:
                signs[j] = 0
        # s = 0
        # for j in range(N):
        #     s += signs[j]
        # print('after', s)
    else:
        ti.loop_config(serialize=True)
        for j in range(N):
            if signs[j] == 0:
                continue
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
        self.signs = np.ones(N, dtype=int)
        self.signs[N//2:] = -1

    def plot(self, ax):
        ixp = self.signs > 0
        ixn = self.signs < 0
        ax.scatter(self.xs[ixp], self.ys[ixp], color='r')
        ax.scatter(self.xs[ixn], self.ys[ixn], color='b')
    
    def update_velocity(self):
        shifts = np.array([-self.D, 0, self.D])
        update_velocity_ti(self.xs, self.ys, self.signs, self.vx, self.vy, shifts)
    
    def dissipation(self, alpha=0.1, alphap=0):
        for j in range(self.N):
            vx0 = self.vx[j]
            vy0 = self.vy[j]

            self.vx[j] = vx0 + alpha*vy0*self.signs[j] - alphap*vx0
            self.vy[j] = vy0 - alpha*vx0*self.signs[j] - alphap*vy0
    
    def annihilate(self, a0=a0):
        shifts = np.array([-self.D, 0, self.D])
        to_annihilate = np.zeros(N)
        annihilate_ti(self.xs, self.ys, self.signs, shifts, a0, to_annihilate)
    
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
        self.xs = np.append(self.xs, np.zeros(len(posy) + len(negy)) + self.D/2)
        self.ys = np.append(self.ys, posy)
        self.ys = np.append(self.ys, negy)
        
        self.vx = np.append(self.vx, np.zeros(len(posy) + len(negy)))
        self.vy = np.append(self.vy, np.zeros(len(posy) + len(negy)))

        self.signs = np.append(self.signs, np.ones_like(posy, dtype=int))
        self.signs = np.append(self.signs, -np.ones_like(negy, dtype=int))

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
    
    def check(self):
        v = sum(self.signs)
        if abs(v) > 0:
            raise RuntimeError("nonzero vorticity")
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--D', type=float, default=1e-2)
    parser.add_argument('--alpha', type=float, default=0.03)
    parser.add_argument('--alphap', type=float, default=1.76e-2)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    D = args.D
    alpha = args.alpha
    alphap = args.alphap
    output = args.output
    save = args.save

    os.makedirs(output, exist_ok=True)

    vp = VortexPoints(N, D)
    fig, ax = plt.subplots()
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    ax.set_aspect('equal')
    pos, = ax.plot(vp.xs[vp.signs > 0], vp.ys[vp.signs > 0], 'o', color='r', ms=2)
    neg, = ax.plot(vp.xs[vp.signs < 0], vp.ys[vp.signs < 0], 'o', color='b', ms=2)
    t = 0
    dt = 0.0001
    last_inject = t
    frame = 0
    while True:
        if t - last_inject > 0.001:
            vp.inject(10)
            vp.check()
            vp.annihilate()
            vp.check()
            last_inject = t
            print("injecting")
        vp.update_velocity()
        vp.check()
        t1 = time.time()
        vp.dissipation(alpha, alphap)
        vp.check()
        vp.step(dt)
        vp.check()
        vp.annihilate()
        vp.check()
        vp.coerce()
        vp.check()
        t2 = time.time()
        t += dt
        pos.set_xdata(vp.xs[vp.signs > 0])
        pos.set_ydata(vp.ys[vp.signs > 0])
        neg.set_xdata(vp.xs[vp.signs < 0])
        neg.set_ydata(vp.ys[vp.signs < 0])
        plt.pause(0.001)
        N = abs(vp.signs).sum()
        print(frame, t2 - t1, t, N, vp.N, sum(vp.signs))
        if save:
            fig.savefig(f'{output}/frame{frame:08d}.png')
        if N == 0:
            break
        frame += 1