# decompyle3 version 3.3.2
# Python bytecode 3.8 (3413)
# Decompiled from: Python 3.8.5 (default, Jul 28 2020, 12:59:40) 
# [GCC 9.3.0]
# Embedded file name: /home/stepan/github/wannier-berri/test/zeemanTe/wannierberri/__fermi_ocean.py
# Compiled at: 2021-02-07 23:46:28
# Size of source mod 2**32: 8180 bytes
import numpy as np
from functools import partial
from scipy import constants
from collections import Iterable, defaultdict
import inspect, sys
from .__utility import print_my_name_start, print_my_name_end, TAU_UNIT, alpha_A, beta_A, warning
from . import __result as result
from scipy.constants import Boltzmann, elementary_charge, hbar, electron_mass, physical_constants, angstrom
bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom
fac_ahc = -100000000.0 * elementary_charge ** 2 / hbar

def LFAHE_spin_fsea(data, Efermi, Bdirection=None):
    factor = hbar / (2 * electron_mass) * fac_ahc
    return IterateEf(data.perturbation_derOmegaTr_ZeemanSpin_2(Bdirection=Bdirection, Emin=(Efermi[0]), Emax=(Efermi[(-1)])), data, Efermi, sep=False, TRodd=False, Iodd=False) * factor


def IterateEf(dataIO, data, Efermi, TRodd, Iodd, sep=False, rank=None, kwargs={}):
    """ this is a general function which accepts dataIO  -- a dictionary like {'i':i , 'io':io, ...}
     and sums for a series of Fermi levels
     parameter dataIO can be a dictionary or a funciton. 
     If needed use callable(dataIO) for judgment and run 
     OCC=OccDelta(data.E_K,dataIO(op,ed),op,ed) or OCC=OccDelta(data.E_K(op,ed),dataIO(op,ed),op,ed)"""
    if isinstance(dataIO, FermiOcean):
        return result.EnergyResult(Efermi, (dataIO(Efermi) / (data.NKFFT_tot * data.cell_volume)), TRodd=TRodd, Iodd=Iodd, rank=rank)


def maxocc(E, Ef, A):
    occ = E <= Ef
    if True not in occ:
        return np.zeros(A.shape[1:])
    return A[max(np.where(occ)[0])]


class FermiOcean:

    def __init__(self, data_list):
        """initialize from a list of dictionaries E:value """
        self.data = data_list
        self.shape = self.data[0][(-10000000000.0)].shape
        self.dtype = self.data[0][(-10000000000.0)].dtype

    def __call__(self, Efermi):
        result = np.zeros(Efermi.shape + self.shape, self.dtype)
        for datak in self.data:
            print('next k-point')
            resk = np.zeros(Efermi.shape + self.shape, self.dtype)
            for E in sorted(datak.keys()):
                print('E=', E)
                resk[Efermi >= E] = datak[E][None]
            else:
                print('finished k-point', resk[(-1)])
                result += resk

        else:
            return result


def sea_from_matrix_product(mat_list, EK, Emin, Emax, ndim, dtype):
    """  
        mat_list  list/tuple of type ( ('nl',A, [ ('ln',B1) , ('lpn',B2,C2) , ('lmn',B3,C3),...] )
                               or  ( ('nm',A) )  or (('n',A) )
        EK-  energies of bands
        Emin,Emax  - minimal and maximal energies of interest
        cartesian dimensiopns of A,B and C should match
        returns an array of energies and values
        """
    EK = np.copy(EK)
    assert Emax >= Emin
    nk, nb = EK.shape
    bandmin = np.full(nk, 0)
    sel = EK < Emin
    bandmin[sel[:, 0]] = [max(np.where(s)[0]) for s in sel[sel[:, 0]]]
    EK[sel] = Emin - 1e-06
    bandmax = np.full(nk, 0)
    sel = EK < Emax
    bandmax[sel[:, 0]] = [max(np.where(s)[0]) for s in sel[sel[:, 0]]]
    print('bandmin: {} \n bandmax={} \n'.format(bandmin, bandmax))
    values = [defaultdict(lambda: np.zeros(((3, ) * ndim), dtype=dtype)) for ik in range(nk)]
    for ABC in mat_list:
        Aind = ABC[0]
        A = ABC[1]
        if Aind == 'nl':
            Ashape = A.shape[3:]
            if len(ABC[2]) == 0:
                raise ValueError("with 'nl' for A at least one B matrix shopuld be provided")
            for ik in range(nk):
                for n in range(bandmin[ik], bandmax[ik]):
                    a = A[ik, :n + 1, n + 1:]
                    bc = 0
                    for BC in ABC[2]:
                        if BC[0] == 'ln':
                            bc += BC[1][ik, n + 1:, :n + 1]
                        else:
                            if BC[0] == 'lpn':
                                bc += np.einsum('lp...,pn...->ln...', BC[1][ik, n + 1:, n + 1:], BC[2][ik, n + 1:, :n + 1])
                            else:
                                if BC[0] == 'lmn':
                                    bc += np.einsum('lm...,mn...->ln...', BC[1][ik, n + 1:, :n + 1], BC[2][ik, :n + 1, :n + 1])
                                else:
                                    raise ValueError('Wron index for B,C : {}'.format(BC[0]))
                                values[ik][EK[(ik, n)]] += np.einsum('nl...,ln...->...', a, bc).real

                if Aind == 'mn':
                    if len(ABC[0] > 2):
                        warning("only one matrix should be given for 'mn'")
                    else:
                        for ik in range(nk):
                            for n in range(bandmin[ik], bandmax[ik]):
                                values[ik][EK[(ik, n)]] += A[ik, :n + 1, :n + 1].sum(axis=(0,
                                                                                           1))

                else:
                    if Aind == 'n':
                        if len(ABC > 2):
                            warning("only one matrix should be given for 'n'")
                        else:
                            for ik in range(nk):
                                for n in range(bandmin[ik], bandmax[ik]):
                                    values[ik][EK[(ik, n)]] += A[ik, :n + 1].sum(axis=0)

                    else:
                        raise RuntimeError('Wrong indexing for array A : {}'.format(Aind))
            else:
                return values
# okay decompiling /home/stepan/tmp/__fermi_ocean.cpython-38.pyc
