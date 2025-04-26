import os.path

import numpy as np
import multiprocessing

from ..fourier.rvectors import Rvectors
from . import System_w90
from .system_R import System_R
from scipy import constants as const
from ..factors import Ry_eV

# from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom  #, Boltzmann

# print ( 1/(hbar*2*np.pi*1e12/elementary_charge) )

# These constants are taken from QE
AMU_RY = const.m_u / const.m_e / 2


def _str2array(s, dtype=float):
    if dtype == float:
        return np.array([_.split() for _ in s.strip().split("\n")], dtype=float)
    elif dtype == complex:
        s = s.replace(",", " ")
        s = _str2array(s)
        return s[:, 0] + 1j * s[:, 1]
    else:
        raise ValueError(f"dtype = '{dtype}' is not supported by _str2array")


class System_Phonon_QE(System_w90):
    """Class to represent dynamical matrices from QuantumEspresso

    reads the '*.dyn*.xml' files and
    allows to interpolate the dynamical matrix and get the phonon frequencies
    so far onl;y DOS, cumDOS and tabulation is tested.
    Other calculators, in principle, may be applied on your own risk.

    Parameters
    ----------
    asr : bool
        imposes a simple acoustic sum rule (equivalent to `zasr = 'simple'` in QuantumEspresso)

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System`

    """

    def __init__(self,
                 seedname,
                 fftlib='fftw',
                 npar=multiprocessing.cpu_count(),
                 asr=True,
                 **parameters):
        if "name" not in parameters:
            parameters["name"] = os.path.split(seedname)[-1]
        System_R.__init__(self, **parameters)
        self.is_phonon = True
        with open(seedname + ".dyn0", "r") as f:
            mp_grid = np.array(f.readline().split(), dtype=int)
            nqirr = int(f.readline().strip())
        #        print (self.mp_grid,nqirr)
        q_points = []
        dynamical_mat = []
        cnt = 0
        import untangle
        for ifile in range(1, nqirr + 1):
            fname = f"{seedname}.dyn{ifile}.xml"
            data = untangle.parse(open(fname).read().lower()).root
            geometry = data.geometry_info
            if ifile == 1:
                number_of_types = int(geometry.number_of_types.cdata)
                masses_tp = np.array(
                    [float(geometry.__getattr__(f'mass_{i + 1}').cdata) for i in range(number_of_types)])
                self.real_lattice = _str2array(geometry.at.cdata)
                self.number_of_atoms = int(geometry.number_of_atoms.cdata)
                self.number_of_phonons = 3 * self.number_of_atoms
                atom_positions_cart = np.array([geometry.__getattr__(f'atom_{i + 1}')['tau'].split()
                                                for i in range(self.number_of_atoms)], dtype=float)
                self.atom_positions = atom_positions_cart.dot(np.linalg.inv(self.real_lattice))
                types = np.array([geometry.__getattr__(f'atom_{i + 1}')['index'] for i in range(self.number_of_atoms)],
                                 dtype=int) - 1
                masses = masses_tp[types]
                self.num_wann = self.number_of_phonons
                self.set_wannier_centers(wannier_centers_red=np.array([atom for atom in self.atom_positions for _ in range(3)])
                                         )
            number_of_q = int(geometry.number_of_q.cdata)
            for iq in range(number_of_q):
                dynamical = data.__getattr__(f"dynamical_mat__{iq + 1}")
                q = _str2array(dynamical.q_point.cdata).reshape(3)
                q = self.real_lattice.dot(q)  # converting from cartisean(2pi/alatt) to reduced coordinates
                dyn_mat = np.zeros((3 * self.number_of_atoms, 3 * self.number_of_atoms), dtype=complex)
                for i in range(self.number_of_atoms):
                    for j in range(self.number_of_atoms):
                        phi = _str2array(dynamical.__getattr__(f'phi_{i + 1}_{j + 1}').cdata, dtype=complex
                                         ).reshape(3, 3, order='F')
                        dyn_mat[i * 3:i * 3 + 3, j * 3:j * 3 + 3] = phi
                dyn_mat = 0.5 * (dyn_mat + dyn_mat.T.conj())
                dynamical_mat.append(dyn_mat)
                q_points.append(q)
                cnt += 1
        self.wannier_centers_cart = self.wannier_centers_red.dot(self.real_lattice)



        self.rvec = Rvectors(lattice=self.real_lattice,
                             shifts_left_red=self.wannier_centers_red,
                             )
        self.rvec.set_Rvec(mp_grid=mp_grid, ws_tolerance=self.ws_dist_tol)
        self.rvec.set_fft_q_to_R(
            kpt_red=q_points,
            fftlib=fftlib,
            numthreads=npar,
        )

        qpoints_found = np.zeros(mp_grid, dtype=float)
        for iq in self.rvec.kpt_mp_grid[i]:
            qpoints_found[iq] = True
        assert np.all(qpoints_found), ('some qpoints were not found in the files:\n' + '\n'.join(str(x))
                                       for x in np.where(np.logical_not(qpoints_found)))

        Ham_R = self.rvec.q_to_R(np.array(dynamical_mat))

        self.set_R_mat('Ham', Ham_R * Ry_eV ** 2)

        self.do_at_end_of_init()

        iR0 = self.rvec.iR0
        if asr:
            for i in range(3):
                for j in range(3):
                    for a in range(self.number_of_atoms):
                        self.Ham_R[iR0, 3 * a + i, 3 * a + j] -= self.Ham_R[:, 3 * a + i, j::3].sum()

        for i in range(self.number_of_atoms):
            for j in range(self.number_of_atoms):
                self.Ham_R[:, 3 * i:3 * i + 3, 3 * j:3 * j + 3] /= np.sqrt(masses[i] * masses[j]) * AMU_RY
