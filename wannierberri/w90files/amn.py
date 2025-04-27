from datetime import datetime
import multiprocessing
import numpy as np
from irrep.bandstructure import BandStructure
from ..symmetry.projections import ProjectionsSet

from ..symmetry.orbitals import Bessel_j_exp_int, Projector
from .utility import str2arraymmn
from .w90file import W90_file


class AMN(W90_file):
    """
    Class to store the projection of the wavefunctions on the initial Wannier functions
    AMN.data[ik, ib, iw] = <u_{i,k}|w_{i,w}>

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extension `.amn`)
    npar : int
        the number of parallel processes to be used for reading

    Notes
    -----


    Attributes
    ----------
    NB : int
        number of bands
    NW : int
        number of Wannier functions
    NK : int
        number of k-points
    data : numpy.ndarray( (NK, NB, NW), dtype=complex)
        the data projections
    """

    @property
    def NB(self):
        return self.data.shape[1]

    def select_bands(self, selected_bands):
        """
        Select the bands to be used in the calculation, the rest are excluded

        Parameters
        ----------
        selected_bands : list of int or bool
            the indices of the bands to be used, or a boolean mask
        """
        print(f"selecting bands {selected_bands} in amn")
        if selected_bands is not None:
            self.data = self.data[:, selected_bands, :]

    @property
    def NW(self):
        return self.data.shape[2]

    @property
    def num_wann(self):
        return self.NW

    def __init__(self, seedname="wannier90", npar=multiprocessing.cpu_count(),
                 **kwargs):
        self.npz_tags = ["data"]
        super().__init__(seedname=seedname, ext="amn", npar=npar, **kwargs)

    def from_w90_file(self, seedname, npar):
        f_amn_in = open(seedname + ".amn", "r").readlines()
        print(f"reading {seedname}.amn: " + f_amn_in[0].strip())
        s = f_amn_in[1]
        NB, NK, NW = np.array(s.split(), dtype=int)
        block = NW * NB
        allmmn = (f_amn_in[2 + j * block:2 + (j + 1) * block] for j in range(NK))
        p = multiprocessing.Pool(npar)
        self.data = np.array(p.map(str2arraymmn, allmmn)).reshape((NK, NW, NB)).transpose(0, 2, 1)

    def to_w90_file(self, seedname):
        f_amn_out = open(seedname + ".amn", "w")
        f_amn_out.write(f"created by WannierBerri on {datetime.now()} \n")
        print(f"writing {seedname}.amn: ")
        f_amn_out.write(f"  {self.NB:3d} {self.NK:3d} {self.NW:3d}  \n")
        for ik in range(self.NK):
            for iw in range(self.NW):
                for ib in range(self.NB):
                    f_amn_out.write(f"{ib + 1:4d} {iw + 1:4d} {ik + 1:4d} {self.data[ik, ib, iw].real:17.12f} {self.data[ik, ib, iw].imag:17.12f}\n")

    # def get_disentangled(self, v_left, v_right):
    #     print(f"v shape  {v_left.shape}  {v_right.shape} , amn shape {self.data.shape} ")
    #     data = np.einsum("klm,kmn->kln", v_left, self.data)
    #     print(f"shape of data {data.shape} , old {self.data.shape}")
    #     return self.__class__(data=data)

    def spin_order_block_to_interlace(self):
        """
        If you are using an old VASP version, you should change the spin_ordering from block to interlace
        """
        data = np.zeros((self.NK, self.NB, self.NW), dtype=complex)
        data[:, :, 0::2] = self.data[:, :, :self.NW // 2]
        data[:, :, 1::2] = self.data[:, :, self.NW // 2:]
        self.data = data

    def spin_order_interlace_to_block(self):
        """ the reverse of spin_order_block_to_interlace"""
        data = np.zeros((self.NK, self.NB, self.NW), dtype=complex)
        data[:, :, :self.NW // 2] = self.data[:, :, 0::2]
        data[:, :, self.NW // 2:] = self.data[:, :, 1::2]
        self.data = data

    # def write(self, seedname, comment="written by WannierBerri"):
    #     comment = comment.strip()
    #     f_amn_out = open(seedname + ".amn", "w")
    #     print(f"writing {seedname}.amn: " + comment + "\n")
    #     f_amn_out.write(comment + "\n")
    #     f_amn_out.write(f"  {self.NB:3d} {self.NK:3d} {self.NW:3d}  \n")
    #     for ik in range(self.NK):
    #         f_amn_out.write("".join(" {:4d} {:4d} {:4d} {:17.12f} {:17.12f}\n".format(
    #             ib + 1, iw + 1, ik + 1, self.data[ik, ib, iw].real, self.data[ik, ib, iw].imag)
    #             for iw in range(self.NW) for ib in range(self.NB)))
    #     f_amn_out.close()


def amn_from_bandstructure_s_delta(bandstructure: BandStructure, positions, normalize=True, return_object=True):
    """
    Create an AMN object from a BandStructure object
    NOTE!!: Only for delta-localised s-orbitals

    more complete implementation is in amn_from_bandstructure()

    Parameters
    ----------
    bandstructure : BandStructure
        the band structure object
    positions : array( (N, 3), dtype=float)
        the positions of the orbitals
    normalize : bool
        if True, the wavefunctions are normalised
    return_object : bool
        if True, return an AMN object, otherwise return the data as a numpy array
    """
    data = []
    pos = np.array(positions)
    for kp in bandstructure.kpoints:
        igk = kp.ig[:3, :] + kp.k[:, None]
        exppgk = np.exp(-2j * np.pi * (pos @ igk))
        wf = kp.WF.conj()
        if normalize:
            wf /= np.linalg.norm(wf, axis=1)[:, None]
        data.append(wf @ exppgk.T)
    data = np.array(data)
    if return_object:
        return AMN(data=data)
    else:
        return data


def amn_from_bandstructure(bandstructure: BandStructure, projections: ProjectionsSet,
                           normalize=True, return_object=True, spinor=False):
    """
    Create an AMN object from a BandStructure object
    So far only delta-localised s-orbitals are implemented

    Parameters
    ----------
    bandstructure : BandStructure
        the band structure object
    projections : ProjectionsSet
        the projections set as an object
    normalize : bool
        if True, the wavefunctions are normalised
    return_object : bool
        if True, return an AMN object, otherwise return the data as a numpy array
    """
    positions = []
    orbitals = []
    basis_list = []
    print(f"Creating amn. Using projections_set \n{projections}")
    for proj in projections.projections:
        pos, orb = proj.get_positions_and_orbitals()
        positions += pos
        orbitals += orb
        basis_list += [bas  for bas in proj.basis_list for _ in range(proj.num_wann_per_site)]
        print(f"proj {proj} pos {pos} orb {orb} basis_list {basis_list}")
    spinor = projections.spinor


    print(f"Creating amn. Positions = {positions} \n orbitals = {orbitals} \n basis_list = \n{basis_list}")
    data = []
    pos = np.array(positions)
    rec_latt = bandstructure.RecLattice
    bessel = Bessel_j_exp_int()
    for kp in bandstructure.kpoints:
        igk = kp.ig[:3, :] + kp.k[:, None]
        expgk = np.exp(-2j * np.pi * (pos @ igk))
        wf = kp.WF.conj()
        if normalize:
            wf /= np.linalg.norm(wf, axis=1)[:, None]
        if spinor:
            wf_up = wf[:, :wf.shape[1] // 2]
            wf_down = wf[:, wf.shape[1] // 2:]

        gk = igk.T @ rec_latt
        projector = Projector(gk, bessel)
        prj = list([projector(orb, basis) for orb, basis in zip(orbitals, basis_list)])
        # print(f"expgk shape {expgk.shape} igk shape {igk.shape} pos shape {pos.shape}")
        # print(f"prj shapes {[p.shape for p in prj]} total {np.array(prj).shape}")
        proj_gk = np.array(prj) * expgk
        if spinor:
            proj_up = wf_up @ proj_gk.T
            proj_down = wf_down @ proj_gk.T
            datak = []
            for u, d in zip(proj_up.T, proj_down.T):
                datak.append(u)
                datak.append(d)
            data.append(np.array(datak).T)
        else:
            data.append(wf @ proj_gk.T)
    data = np.array(data)
    if return_object:
        return AMN().from_dict(data=data)
    else:
        return data
