import os
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, MixerSum
from irrep.spacegroup import SpaceGroup
from ..grid.path import Path


class WorkflowGpaw:

    def __init__(self,
             ase_atoms,
             prefix=None,
             path=None
             ):
        self.atoms = ase_atoms
        if prefix is None:
            prefix = str(ase_atoms.symbols)
        self.fullpath = prefix
        if path is not None:
            self.fullpath = os.path.join(path, prefix)

    @property
    def path_scf_calculator(self):
        return f"{self.fullpath}-scf.gpw"
    
    @property
    def path_nscf_irred_calculator(self):
        return f"{self.fullpath}-nscf-irred.gpw"
    
    @property
    def path_bands_calculator(self):
        return f"{self.fullpath}-bands.gpw"

    @classmethod
    def from_cell(cls, lattice, positions, symbols, **kwargs):
        ase_atoms = Atoms(symbols=symbols, cell=lattice, pbc=[1, 1, 1], scaled_positions=positions)
        return cls.from_ase_atoms(ase_atoms, **kwargs)


    def run_scf(self, kpts=(4, 4, 4), Ecut=500, xc="PBE", 
                redo=False,
                kwargs_spacegroup={"irreducible": True}, 
                **kwargs):

        if os.path.exists(self.path_scf_calculator) and not redo:
            return GPAW(self.path_scf_calculator, txt=None)
        
        kwargs_loc = dict(
            mode=PW(Ecut),
            xc=xc,
            symmetry={'symmorphic': False},
            kpts={"size": kpts, "gamma": True},
            convergence={"density": 1e-6},
            mixer=MixerSum(0.25, 8, 100),
            txt=f"{self.fullpath}-scf.txt"
        )
        kwargs_loc.update(kwargs)
        calc_scf = GPAW(**kwargs_loc)
        self.atoms.calc = calc_scf
        self.atoms.get_potential_energy()
        calc_scf.write(f"{self.fullpath}-scf.gpw", mode="all")
        self.spacegroup = SpaceGroup.from_gpaw(calc_scf)
        return calc_scf

    def get_path(self, dk=0.05):
        path = Path.seekpath(lattice=np.array(self.atoms.get_cell()),
                             positions=self.atoms.get_scaled_positions(),
                             numbers=self.atoms.get_atomic_numbers(),
                             dk=dk)
        return path

    def run_nscf(self, factor_unocc=1,
                 grid=(4, 4, 4),
                 redo=False,
                 **kwargs):
        if os.path.exists(self.path_nscf_irred_calculator) and not redo:
            return GPAW(self.path_nscf_irred_calculator, txt=None)
        
        calc_scf = self.run_scf(**kwargs)
        nelec = calc_scf.get_number_of_electrons()
        nbands = int(nelec * (factor_unocc + 1) / 2)
        nbands_converge = int(nelec * (factor_unocc * 0.8 + 1) / 2)

        print(f"number of electrons: {nelec}, number of bands: {nbands}")
        
        irred_kpt = self.spacegroup.get_irreducible_kpoints_grid(grid)
        kwargs_loc = dict(kpts=irred_kpt,
            symmetry={'symmorphic': False},
            nbands=nbands,
            convergence={'bands': nbands_converge},
            txt=f'{self.fullpath}-nscf-irred.txt')
        kwargs_loc.update(kwargs)
        calc_nscf_irred = calc_scf.fixed_density(
            **kwargs_loc)
        calc_nscf_irred.write(self.path_nscf_irred_calculator, mode='all')
        return calc_nscf_irred

    def run_nscf_kpath(self, dk=0.05, factor_unocc=1, **kwargs):
        calc = GPAW(f'{self.fullpath}-scf.gpw', txt=None)
        nelec = calc.get_number_of_electrons()
        nbands = int(nelec * (factor_unocc + 1) / 2)
        nbands_converge = int(nelec * (factor_unocc * 0.8 + 1) / 2)
        print(f"number of electrons: {nelec}, number of bands: {nbands}")
        path = self.get_path(dk=dk)
        calc_nscf_kpath = calc.fixed_density(
            kpts=path.get_kpoints(),
            symmetry='off',
            nbands=nbands,
            convergence={'bands': nbands_converge},
            txt=f'{self.fullpath}-nscf-kpath.txt')
        calc_nscf_kpath.write(f'{self.fullpath}-nscf-kpath.gpw', mode='all')
        self.energies_kpath_dft = calc_nscf_kpath.get_eigenvalues()
        self.kpath_dft = path

    def wannierize(self, **kwargs):
        pass
