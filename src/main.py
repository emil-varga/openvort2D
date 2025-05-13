import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import time
import argparse

import taichi as ti
import os

kappa = 9.96e-4

from VortexPoints import VortexPoints
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--D', type=float, default=1e-2)
    parser.add_argument('--alpha', type=float, default=0.03)
    parser.add_argument('--alphap', type=float, default=1.76e-2)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--inject', action='store_true')
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--polarization', type=float, default=0)
    parser.add_argument('--polarization-type', type=str, default='none')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=1e-9)

    args = parser.parse_args()
    D = args.D
    alpha = args.alpha
    alphap = args.alphap
    output = args.output
    save = args.save

    if args.gpu:
        ti.init(ti.gpu)
    else:
        ti.init(ti.cpu)

    os.makedirs(output, exist_ok=True)

    vp = VortexPoints(args.N, D, polarization=args.polarization, polarization_type=args.polarization_type)
    fig, ax = plt.subplots()
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    ax.set_aspect('equal')
    pos, = ax.plot(vp.xs[vp.signs > 0], vp.ys[vp.signs > 0], 'o', color='r', ms=2)
    neg, = ax.plot(vp.xs[vp.signs < 0], vp.ys[vp.signs < 0], 'o', color='b', ms=2)
    t = 0
    dt = args.dt
    last_inject = t
    frame = 0
    it = 0
    with open(f'{output}/L_t.txt', 'w') as Lfile:
        while True:
            if args.inject and t - last_inject > 0.0005:
                vp.inject(5)
                vp.annihilate()
                vp.check()
                last_inject = t
                # print("injecting")
            vp.update_velocity()
            vp.check()
            vp.dissipation(alpha, alphap)
            vp.step(dt)
            vp.annihilate()
            vp.check()
            vp.coerce()
            t += dt
            pos.set_xdata(vp.xs[vp.signs > 0])
            pos.set_ydata(vp.ys[vp.signs > 0])
            neg.set_xdata(vp.xs[vp.signs < 0])
            neg.set_ydata(vp.ys[vp.signs < 0])
            plt.pause(0.001)
            N = abs(vp.signs).sum()
            print(it, t, N, vp.N, sum(vp.signs))
            Lfile.write(f"{it}\t{t}\t{N}\n")
            Lfile.flush()
            if it%args.save_every == 0 and save:
                fig.savefig(f'{output}/frame{frame:08d}.png')
                frame += 1
            if N == 0:
                break
            it += 1
