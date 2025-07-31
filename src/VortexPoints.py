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
from numpy.random import rand, randn

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
def update_velocity_walls_ti(xs: ti.types.ndarray(), ys: ti.types.ndarray(), signs: ti.types.ndarray(),
                             vx: ti.types.ndarray(), vy: ti.types.ndarray(), D: ti.types.float64):
    N = xs.shape[0]
    for j in range(N):
        if signs[j] == 0:
            continue
        vx[j] = 0
        vy[j] = 0
        for k in range(N):
            for xshift in range(-1, 2):
                x_jk = xs[j] - xs[k] + xshift*D

                # no shift
                if k!=j or xshift!=0:
                    y_jk = ys[j] - ys[k]
                    r2_jk = x_jk**2 + y_jk**2
                    vx[j] += -kappa/2/3.14159/r2_jk*y_jk*signs[k]
                    vy[j] += kappa/2/3.14159/r2_jk*x_jk*signs[k]

                #shift up
                mirror_flip = -1
                y_jk = ys[j] - (2*D - ys[k])
                r2_jk = x_jk**2 + y_jk**2
                vx[j] += -kappa/2/3.14159/r2_jk*y_jk*signs[k]*mirror_flip
                vy[j] += kappa/2/3.14159/r2_jk*x_jk*signs[k]*mirror_flip
                
                #shift down
                mirror_flip = -1
                y_jk = ys[j] + ys[k]
                r2_jk = x_jk**2 + y_jk**2
                vx[j] += -kappa/2/3.14159/r2_jk*y_jk*signs[k]*mirror_flip
                vy[j] += kappa/2/3.14159/r2_jk*x_jk*signs[k]*mirror_flip

@ti.kernel
def calculate_velocity_walls_ti(vort_xs: ti.types.ndarray(), vort_ys: ti.types.ndarray(), signs: ti.types.ndarray(),
                                xs: ti.types.ndarray(), ys: ti.types.ndarray(),
                                vx: ti.types.ndarray(), vy: ti.types.ndarray(), D: ti.types.float64):
    N = vort_xs.shape[0]
    M = xs.shape[0]
    for j in range(M):
        vx[j] = 0
        vy[j] = 0
        for k in range(N):
            if signs[k] == 0:
                continue
            for xshift in range(-1, 2):
                x_jk = xs[j] - vort_xs[k] + xshift*D

                # no shift
                y_jk = ys[j] - vort_ys[k]
                r2_jk = x_jk**2 + y_jk**2
                vx[j] += -kappa/2/3.14159/r2_jk*y_jk*signs[k]
                vy[j] += kappa/2/3.14159/r2_jk*x_jk*signs[k]

                #shift up
                mirror_flip = -1
                y_jk = ys[j] - (2*D - vort_ys[k])
                r2_jk = x_jk**2 + y_jk**2
                vx[j] += -kappa/2/3.14159/r2_jk*y_jk*signs[k]*mirror_flip
                vy[j] += kappa/2/3.14159/r2_jk*x_jk*signs[k]*mirror_flip
                
                #shift down
                mirror_flip = -1
                y_jk = ys[j] + vort_ys[k]
                r2_jk = x_jk**2 + y_jk**2
                vx[j] += -kappa/2/3.14159/r2_jk*y_jk*signs[k]*mirror_flip
                vy[j] += kappa/2/3.14159/r2_jk*x_jk*signs[k]*mirror_flip

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

@ti.kernel
def annihilate_walls_ti(xs: ti.types.ndarray(), ys: ti.types.ndarray(), 
                        signs: ti.types.ndarray(dtype=ti.int64), shifts: ti.types.ndarray(),
                        a0: float, to_annihilate: ti.types.ndarray()) -> int:
    N = xs.shape[0]
    S = shifts.shape[0]

    for j in range(N):
        to_annihilate[j] = 0

    for j in range(N):
        if signs[j] == 0:
            continue
        move_on = False
        for k in range(j+1, N):
            if signs[k] == 0 or signs[j] == signs[k]:
                continue
            for xshift in range(S):
                x_jk = xs[k] - xs[j] + shifts[xshift]
                y_jk = ys[k] - ys[j]
                r_jk = ti.sqrt(x_jk**2 + y_jk**2)
                if r_jk < a0:
                    to_annihilate[j] += 1
                    to_annihilate[k] += 1
                    move_on = True
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
                    x_jk = xs[k] - xs[j] + shifts[xshift]
                    y_jk = ys[k] - ys[j]
                    r_jk = ti.sqrt(x_jk**2 + y_jk**2)
                    if r_jk < a0:
                        signs[j] = 0
                        signs[k] = 0
                        move_on = True
                        break
                if move_on:
                    break
    
    #now annihilate on walls
    for k in range(N):
        if signs[k] == 0:
            continue
        if ys[k] < a0 or shifts[2] - ys[k] < a0:
            signs[k] = 0
    return OK
        

class VortexPoints:
    def __init__(self, N:int|None=None, D:float=1, a0:float=1e-5,
                 polarization:float=0, polarization_type:str='none',
                 walls:bool=False, vpin:float=0, pin_type='threshold',
                 probe_type:str = 'uniform', probe_v:float=0, probe_v_freq:float=0,
                 probe_grid=None, probe_grid_v=0,
                 gridx=None, gridy=None, grid_div=None):
        self.walls = walls
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
        self.signs[int(N/2):] = -1
        self.vpin = vpin
        self.pin_type = pin_type

        self.probe_type = probe_type
        self.probe_v = probe_v
        self.probe_v_freq = probe_v_freq
        self.probe_grid = probe_grid
        self.probe_grid_v = probe_grid_v

        match probe_type:
            case 'uniform':
                self._probe_v = self.uniform_probe_v
            case 'grid':
                self._probe_v = self.grid_probe_v
            case 'combined':
                self._probe_v = self.combined_probe_v
            case _:
                raise ValueError(f"Unknown probe type {probe_type}")

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
            case 'dipole':
                print("initializing dipole")
                N_random = int(N*(1 - polarization))
                N_dipole = int(N*polarization)

                #the polarized part
                x0 = D*np.sqrt(2)/3/2
                n2 = int(N_dipole/2)
                self.xs[:n2] = D/2 - x0 + randn(n2)*D/10
                self.ys[:n2] = D/2 - x0 + randn(n2)*D/10
                self.signs[:n2] = 1
                self.xs[n2:N_dipole] = D/2 + x0 + randn(n2)*D/10
                self.ys[n2:N_dipole] = D/2 + x0 + randn(n2)*D/10
                self.signs[n2:N_dipole] = -1

                #the random part, positions are already random
                self.signs[N_dipole:int(N_dipole + N_random/2)] = +1
                self.signs[int(N_dipole + N_random/2):] = -1

                self.coerce()
            case 'grid':
                n_bunch = int(N/gridx/gridy)
                sigma_x = D/gridx
                sigma_y = D/gridy
                n = 0
                for j in range(gridy):
                    sign = 1 - 2*int(j % 2)
                    for k in range(gridx):
                        cy = (j + 0.5*int((gridy+1)%2))*sigma_y
                        cx = (k + 0.5*int((gridx+1)%2))*sigma_x
                        self.xs[n:(n+n_bunch)] = cx + randn(n_bunch)*sigma_x/grid_div
                        self.ys[n:(n+n_bunch)] = cy + randn(n_bunch)*sigma_y/grid_div
                        self.signs[n:(n+n_bunch)] = sign
                        sign *= -1
                        n += n_bunch
                self.trim()
            case _:
                raise ValueError("Unknown polarization type.")
        self.shifts = np.array([-D, 0, D])
        self.to_annihilate = np.zeros(N)
        self.t = 0
        self.step_n = 0
        self.omega = 0
        self.A = 0
    
    def uniform_probe_v(self):
        probe_vx = self.probe_v*np.cos(2*np.pi*self.probe_v_freq*self.t)
        return np.repeat(probe_vx, len(self.vx)), np.zeros_like(self.vy)
    
    def grid_probe_v(self):
        amplitude = self.probe_grid_v*np.cos(2*np.pi*self.probe_v_freq*self.t)
        n, k = self.probe_grid
        spatial_x = n*np.cos(np.pi/self.D*n*self.xs)*np.cos(np.pi/self.D*k*self.ys)
        spatial_y = -k*np.sin(np.pi/self.D*n*self.xs)*np.sin(np.pi/self.D*k*self.ys)
        return amplitude*spatial_x, amplitude*spatial_y
    
    def combined_probe_v(self):
        vxu, vyu = self.uniform_probe_v()
        vxg, vyg = self.grid_probe_v()
        return vxu+vxg, vyu+vyg

    def plot(self, ax):
        ixp = self.signs > 0
        ixn = self.signs < 0
        ax.scatter(self.xs[ixp], self.ys[ixp], color='r')
        ax.scatter(self.xs[ixn], self.ys[ixn], color='b')
    
    def update_velocity(self):
        if self.walls:
            update_velocity_walls_ti(self.xs, self.ys, self.signs, self.vx, self.vy, self.D)
        else:
            update_velocity_ti(self.xs, self.ys, self.signs, self.vx, self.vy, self.shifts)
        
        probe_vx, probe_vy = self._probe_v()
        self.vx += probe_vx
        self.vy += probe_vy
    
    def dissipation(self, alpha=0.1, alphap=0):
        v2 = self.vx**2 + self.vy**2
        inv_beta2 = v2/self.vpin**2
        depinned = inv_beta2 > 1
        # x = 1/Gamma
        x = np.where(depinned, np.sqrt(inv_beta2 - 1), 0)
        alpha_hat = x*(alpha**2 + alpha*x + alphap**2 - 2*alphap + 1)
        alpha_hat /= (alpha**2 + 2*alpha*x + alphap**2 - 2*alphap + x**2 + 1)
        alphap_hat = (alpha**2 + 2*alpha*x + alphap**2 + alphap*x**2 - 2*alphap + 1)
        alphap_hat /= (alpha**2 + 2*alpha*x + alphap**2 - 2*alphap + x**2 + 1)
        if self.pin_type == 'threshold':
            mf_vx = np.where(depinned, self.vx + alpha*self.vy*self.signs - alphap*self.vx, 0)
            mf_vy = np.where(depinned, self.vy - alpha*self.vx*self.signs - alphap*self.vy, 0)
        elif self.pin_type == 'drag':
            mf_vx = np.where(depinned, self.vx + alpha_hat*self.vy*self.signs - alphap_hat*self.vx, 0)
            mf_vy = np.where(depinned, self.vy - alpha_hat*self.vx*self.signs - alphap_hat*self.vy, 0)
        elif self.pin_type == 'none':
            mf_vx = self.vx + alpha*self.vy*self.signs - alphap*self.vx
            mf_vy = self.vy - alpha*self.vx*self.signs - alphap*self.vy
        else:
            raise ValueError(f"Unknown pin type, {self.pin_type}.")

        self.vx = mf_vx
        self.vy = mf_vy
    
    def annihilate(self):
        if self.walls:
            annihilate_walls_ti(self.xs, self.ys, self.signs, self.shifts, self.a0, self.to_annihilate)
        else:
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
        self.xs += self.vx*dt
        self.ys += self.vy*dt
        self.t += dt
        self.step_n += 1
        if self.step_n % 100 == 0:
            self.cleanup()
    
    def coerce(self):
        """
        Adjusts the coordinates of points to ensure they remain within the bounds [0, D) for both x and y axes.
        This method iterates over all points and, if any coordinate (x or y) falls outside the interval [0, D),
        it wraps the coordinate around by adding or subtracting D as necessary. The process repeats until all
        coordinates are within bounds.
        Attributes used:
            self.xs (list or array): The x-coordinates of the points.
            self.ys (list or array): The y-coordinates of the points.
            self.N (int): The number of points.
            self.D (float): Simulation domain size
        """
        # This function ensures all point coordinates are wrapped into the [0, D) interval, 
        # simulating periodic boundaries (like a torus).

        while True:
            coerced = 0
            for j in range(self.N):
                if self.xs[j] > self.D:
                    self.xs[j] -= self.D
                    coerced += 1
                if self.ys[j] > self.D:
                    self.ys[j] -= self.D
                    coerced += 1
                if self.xs[j] < 0:
                    self.xs[j] += self.D
                    coerced += 1
                if self.ys[j] < 0:
                    self.ys[j] += self.D
                    coerced += 1
            if coerced == 0:
                break
    
    def cleanup(self):
        """
        Removes points that are not active (i.e., have zero vorticity).
        """
        ix_nonzero = abs(self.signs) > 0
        self.xs = self.xs[ix_nonzero]
        self.ys = self.ys[ix_nonzero]
        self.vx = self.vx[ix_nonzero]
        self.vy = self.vy[ix_nonzero]
        self.signs = self.signs[ix_nonzero]
        self.N = len(self.xs)
    
    def trim(self):
        """
        Trims the points, which the random initial condition placed outside of the simulation domain.
        """
        for j in range(self.N):
            if self.xs[j] > self.D or self.xs[j] < 0:
                self.signs[j] = 0
            if self.ys[j] > self.D or self.ys[j] < 0:
                self.signs[j] = 0
    
    def check(self):
        if not self.walls:
            v = sum(self.signs)
            if abs(v) > 0:
                raise RuntimeError("nonzero vorticity")
