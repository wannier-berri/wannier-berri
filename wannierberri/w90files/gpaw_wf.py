import numpy as np


class WavefunctionsGpaw:

    def __init__(self, calc, ispin,
                 kptirr, selected_kpoints,
                 kpt_from_kptirr_isym=None,
                 kpt2kptirr=None,
                 spacegroup=None,
                 kpt_red=None,
                 cache=True):
        from gpaw.ibz2bz import IBZ2BZMaps
        from gpaw.wannier90 import get_projections_in_bz
        self.kptirr = kptirr
        self.kpt_red = kpt_red
        self.kpt_from_kptirr_isym = kpt_from_kptirr_isym
        self.kpt2kptirr = kpt2kptirr
        self.selected_kpoints = selected_kpoints
        self.R_asii = calc.setups.atomrotations.get_R_asii()
        self.spos_ac = calc.spos_ac
        self.spacegroup = spacegroup
        self.get_projections_in_bz = get_projections_in_bz
        self.calc = calc
        self.ispin = ispin
        self.nband = self.calc.get_number_of_bands()
        self.wf_dict = {}
        self.proj_dict = {}
        self.cache = cache
        self.ibz2bz = IBZ2BZMaps.from_calculator(self.calc)

    def get_wavefunctions(self, bz_index):
        if bz_index in self.kptirr:
            ibz_calc_index = self.selected_kpoints[bz_index]
            return np.array([self.calc.wfs.get_wave_function_array(n, ibz_calc_index, self.ispin, periodic=True)
                            for n in range(self.nband)])
        else:
            ibz_index = self.kpt2kptirr[bz_index]
            isym = self.kpt_from_kptirr_isym[bz_index]
            symop = self.spacegroup.symmetries[isym]
            kpt_origin = self.kpt_red[ibz_index]
            kpt_target = self.kpt_red[bz_index]
            return rotate_pseudo_wavefunction(
                self(ibz_index), symop, kpt_origin, kpt_target)
            

    def get_projection(self, bz_index):
        if bz_index not in self.proj_dict:
            if bz_index in self.kptirr:
                ibz_index = self.selected_kpoints[bz_index]
                kpt = self.calc.wfs.kpt_qs[ibz_index][self.ispin]
                nbands = self.calc.wfs.bd.nbands
                # Get projections in ibz
                proj = kpt.projections.new(nbands=nbands, bcomm=None)
                proj.array[:] = kpt.projections.array[:nbands]
                self.proj_dict[bz_index] = proj
            else:
                raise NotImplementedError("Projections at non-irreducible k-points are not implemented yet."
                                          f" kptirr={self.kptirr}, bz_index={bz_index}")
        return self.proj_dict[bz_index]


    def __call__(self, bz_index):
        if bz_index not in self.wf_dict:
            wf = self.get_wavefunctions(bz_index)
            if self.cache:
                self.wf_dict[bz_index] = wf
            return wf
        return self.wf_dict[bz_index]


def rotate_pseudo_wavefunction(psi_n_grid, symop, k_origin, k_target):
    """
    Rotate the pseudo wavefunction according to the given symmetry operation

    Parameters
    ----------
    psi_nG : np.ndarray(shape=(NB, n1, n2, n3), dtype=complex)
        the pseudo wavefunction in G-space
    symop : irrep.SymmetryOperation
        the symmetry operation
    k_origin : np.ndarray(shape=(3,), dtype=float)
        the original k-point in the basis of the reciprocal lattice
    k_target : np.ndarray(shape=(3,), dtype=float)
        the target k-point in the basis of the reciprocal lattice

    Returns
    -------
    psi_nG_rot : np.ndarray(shape=(NB, NG), dtype=complex)
        the rotated pseudo wavefunction in G-space
    """
    NB = psi_n_grid.shape[0]
    Nc = psi_n_grid.shape[1:]
    i_trans = symop.translation * np.array(Nc)
    i_trans_int = np.round(i_trans).astype(int)
    assert np.allclose(i_trans, i_trans_int), f"Translation {symop.translation} not compatible with grid {Nc}"
    indx = np.dot(symop.rotation, np.indices(Nc).reshape((3, -1)) + i_trans_int[:, None])
    indx = np.ravel_multi_index(indx, Nc, 'wrap')
    assert len(set(indx)) == np.prod(Nc), f"Rotation {symop.rotation} + translation {symop.translation} leads to duplicate indices on the grid {Nc}"
    psi_n_grid = psi_n_grid.reshape(NB, -1)[:, indx].reshape((NB,) + Nc)

    kpt_shift = k_target - symop.transform_k(k_origin)
    kpt_shift_int = np.round(kpt_shift).astype(int)
    assert np.allclose(kpt_shift, kpt_shift_int), f"k-point shift {kpt_shift} is not a reciprocal lattice vector"
    for i, ksh in enumerate(kpt_shift_int):
        if ksh != 0:
            phase = np.exp(-2j * np.pi * ksh * np.arange(Nc[i]) / Nc[i]).reshape((1,) * (i + 1) + (Nc[i],) + (1,) * (2 - i))
            psi_n_grid = psi_n_grid * phase
    return psi_n_grid
