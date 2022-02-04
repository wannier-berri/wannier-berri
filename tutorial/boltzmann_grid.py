#!/usr/bin/env python3
import pickle
import wannierberri as wb






with open("tab_result.pickle","rb") as f:
    bgrid = wb.boltzmann_grid.Boltzmann_grid(pickle.load(f),tau = 1e-9,mu=12.6,kBT=1,)

for E in 0.01,0.1,1.0,2.0:
    c = bgrid.current(E=[0,0,E])
    print (f"E={E} V/m  c = {c}  A/m^2 , sigma = {c[2]/E/100:.5e} S/cm")