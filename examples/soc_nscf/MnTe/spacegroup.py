from irrep.spacegroup import SpaceGroup
from irrep.bandstructure import BandStructure
import numpy as np
from wannierberri.w90files.soc import SOC

theta = np.pi / 2
phi = np.pi / 2

S = SOC.get_S_vss(theta=theta, phi=phi)

print(f"{theta=:.2f}, {phi=:.2f}")
print("Sx= \n", np.round(S[0], 3))
print("Sy= \n", np.round(S[1], 3))
print("Sz= \n", np.round(S[2], 3))

array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)

print(f"{np.issubdtype(array.dtype, np.integer)=}")
print(f"{(array.dtype == int)=}")



bandstructure = BandStructure(code="gpaw", calculator_gpaw="scf.gpw", onlysym=True)
sg = bandstructure.spacegroup
sg.show()

mg = SpaceGroup.from_cell(real_lattice=sg.real_lattice, positions=sg.positions, typat=sg.typat,
                          magmom=[[0, 1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]]
                          # magmom=[[1,0,0],[-1,0,0],[0,0,0],[0,0,0]]
                            )

mg.show()
