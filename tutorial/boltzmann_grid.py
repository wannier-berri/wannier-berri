#!/usr/bin/env python3
import pickle
import wannierberri as wb






with open("tab_result.pickle","rb") as f:
    bgrid = wb.boltzmann_grid.Boltzmann_grid(pickle.load(f),tau = 1e-9,mu=12.6,kBT=0.5,)

for Ez in 0.01,0.1,1.0,2.0:
    E=[0,0,Ez]

    c = bgrid.current(E=E,n_iter = 5)
    print (f"E={E} V/m  c = {c}  A/m^2 , sigma_zz = {c[2]/E[2]/100:.5e} S/cm")