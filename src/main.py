#!/usr/bin/env python
# This file is part of the openvort2D project.
#
# Copyright (C) 2024 Emil Varga and superfluid lab, MFF CUNI
#
# openvort2D is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# openvort2D is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with openvort2D. If not, see <https://www.gnu.org/licenses/>.
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from numpy.random import randn
import time
import argparse

import os.path as path
from glob import glob

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
    parser.add_argument('--gridx', type=int)
    parser.add_argument('--gridy', type=int)
    parser.add_argument('--grid-sigma-div', type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=1e-9)
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--walls', action='store_true')
    parser.add_argument('--pinning-v', type=float, default=0)
    parser.add_argument('--probe-v', type=float, default=0)
    parser.add_argument('--probe-v-freq', type=float, default=0)
    parser.add_argument('--probe-grid', type=int, nargs=2, default=[0,0])
    parser.add_argument('--probe-grid-v', type=float, default=0)
    parser.add_argument('--probe-type', type=str, default='uniform', 
                        help="Probe flow type. Options are 'uniform' (default), 'grid' and 'combined'.")
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--pin-type', type=str, default='threshold', 
                        help="Available are 'threshold' (default) where the vortex moves freely if its velocity is" \
                        "higher than depin or not at all or 'drag', where vortex velocity is offset by the pinning" \
                        "velocity.")
    parser.add_argument('--variable-save-rate', type=float, default=0,
                        help="Extend the time between saved frames as t_{i+1} = t_{i}*(1 + vsr)")
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
    
    base_output = output
    suffix_k = 1 
    if os.path.exists(output) and not args.restart:
        while True:
            output = base_output + f"_{suffix_k}"
            if not os.path.exists(output):
                break
            suffix_k += 1
    
    if args.restart:
        vp_files = glob(path.join(output, '*.npz'))
        vp_files.sort()
        print(len(vp_files))
        restart_file = np.load(vp_files[-1], allow_pickle=True)
        vp = restart_file['arr_0'].item()
        Lfile_mode = 'a'
        frame = len(vp_files)
    else:
        os.makedirs(output)    
        vp = VortexPoints(args.N, D, polarization=args.polarization, polarization_type=args.polarization_type,
                          walls=args.walls, vpin=args.pinning_v,
                          probe_v=args.probe_v, probe_v_freq=args.probe_v_freq,
                          gridx=args.gridx, gridy=args.gridy, grid_div=args.grid_sigma_div,
                          probe_type=args.probe_type, probe_grid=args.probe_grid, probe_grid_v=args.probe_grid_v)
        Lfile_mode='w'
        frame = 0
    fig, ax = plt.subplots()
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    ax.set_aspect('equal')
    pos, = ax.plot(vp.xs[vp.signs > 0], vp.ys[vp.signs > 0], 'o', color='r', ms=2)
    neg, = ax.plot(vp.xs[vp.signs < 0], vp.ys[vp.signs < 0], 'o', color='b', ms=2)
    dt = args.dt
    last_inject = 0
    it = 0
    save_rate = args.save_every
    save_countdown = 0
    with open(f'{output}/L_t.txt', Lfile_mode) as Lfile:
        while True:
            if args.inject and vp.t - last_inject > 0.0005:
                vp.inject(5)
                vp.annihilate()
                vp.check()
                last_inject = vp.t
                # print("injecting")
            vp.update_velocity()
            vp.check()
            vp.dissipation(alpha, alphap)
            vp.step(dt)
            vp.annihilate()
            vp.check()
            vp.coerce()
            pos.set_xdata(vp.xs[vp.signs > 0])
            pos.set_ydata(vp.ys[vp.signs > 0])
            neg.set_xdata(vp.xs[vp.signs < 0])
            neg.set_ydata(vp.ys[vp.signs < 0])
            if not args.no_plot:
                plt.pause(0.001)
            N = abs(vp.signs).sum()
            print(it, vp.t, N, vp.N, sum(vp.signs))
            if save_countdown == 0 and save:
                Lfile.write(f"{it}\t{vp.t}\t{N}\n")
                Lfile.flush()
                fig.savefig(f'{output}/frame{frame:08d}.png')
                frame += 1
                np.savez(f'{output}/vp_{frame:08d}.npz', vp)
                save_rate = save_rate*(1 + args.variable_save_rate)
                save_countdown = int(save_rate)
            if N == 0:
                break
            it += 1
            save_countdown -= 1
