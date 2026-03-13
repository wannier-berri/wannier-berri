from datetime import datetime
import multiprocessing
import numpy as np
from ..symmetry.projections import ProjectionsSet

from ..symmetry.orbitals import Bessel_j_radial_int, Projector
from .w90file import W90_file, auto_kptirr, check_shape


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

    extension = "amn"

    def __init__(self, data, NK=None):
        super().__init__(data=data, NK=NK)
        self.NB, self.NW = check_shape(self.data)

    @property
    def num_wann(self):
        return self.NW


    @classmethod
    def from_w90_file(cls, seedname, npar=None):
        if npar is None:
            npar = multiprocessing.cpu_count()
        f_amn_in = open(seedname + ".amn", "r").readlines()
        print(f"reading {seedname}.amn: " + f_amn_in[0].strip())
        s = f_amn_in[1]
        NB, NK, NW = np.array(s.split(), dtype=int)
        block = NW * NB
        allmmn = (f_amn_in[2 + j * block:2 + (j + 1) * block] for j in range(NK))
        p = multiprocessing.Pool(npar)
        from .utility import str2arraymmn
        data = np.array(p.map(str2arraymmn, allmmn)).reshape((NK, NW, NB)).transpose(0, 2, 1)
        return AMN(data=data)

    def to_w90_file(self, seedname):
        f_amn_out = open(seedname + ".amn", "w")
        f_amn_out.write(f"created by WannierBerri on {datetime.now()} \n")
        print(f"writing {seedname}.amn: ")
        f_amn_out.write(f"  {self.NB:3d} {self.NK:3d} {self.NW:3d}  \n")
        for ik in range(self.NK):
            for iw in range(self.NW):
                for ib in range(self.NB):
                    f_amn_out.write(f"{ib + 1:4d} {iw + 1:4d} {ik + 1:4d} {self.data[ik, ib, iw].real:17.12f} {self.data[ik, ib, iw].imag:17.12f}\n")


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


    @classmethod
    def from_bandstructure(cls, bandstructure, projections: ProjectionsSet,
                           normalize=True, verbose=False,
                           selected_kpoints=None,
                           kptirr=None,
                           NK=None):
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
        """

        NK, selected_kpoints, kptirr = auto_kptirr(
            bandstructure, selected_kpoints=selected_kpoints, kptirr=kptirr, NK=NK)


        positions = []
        orbitals = []
        radial_nodes_list = []
        basis_list = []
        spread_list = []
        print(f"Creating amn. Using projections_set \n{projections}")
        for proj in projections.projections:
            pos, orb = proj.get_positions_and_orbitals()
            positions += pos
            orbitals += orb
            radial_nodes_list += [proj.radial_nodes] * proj.num_wann_scalar
            spread_list += [proj.spread_factor] * proj.num_wann_scalar
            basis_list += [bas  for bas in proj.basis_list for _ in range(proj.num_wann_per_site_scalar)]
            if verbose:
                print(f"proj {proj} pos {pos} orb {orb} basis_list {basis_list}")
        spinor = projections.spinor



        if verbose:
            print(f"Creating amn. Positions = {positions} \n orbitals = {orbitals} \n basis_list = \n{basis_list}")
        data = {}
        pos = np.array(positions)
        rec_latt = bandstructure.RecLattice
        unit_cell_volume = np.linalg.det(bandstructure.spacegroup.lattice)
        bessel = Bessel_j_radial_int()

        for ikirr in kptirr:
            kp = bandstructure.kpoints[selected_kpoints[ikirr]]
            igk = kp.ig[:, :3] + kp.k[None, :]
            expgk = np.exp(-2j * np.pi * (pos @ igk.T))
            wf = kp.WF
            wf = wf.conj()
            if normalize:
                norms = np.linalg.norm(wf, axis=(1, 2))
                wf = wf / norms[:, None, None]
            if spinor:
                wf_up = wf[:, :, 0]
                wf_down = wf[:, :, 1]

            gk = igk @ rec_latt
            prj = []
            projector_dict = {}
            for orb, basis, radial_nodes, spread_factor in zip(orbitals, basis_list, radial_nodes_list, spread_list):
                if spread_factor not in projector_dict:
                    projector_dict[spread_factor] = Projector(gk, bessel, spread_factor=spread_factor)
                projector = projector_dict[spread_factor]
                prj.append(projector(orb, basis, radial_nodes))
            proj_gk = np.array(prj) * expgk / np.sqrt(unit_cell_volume)
            if spinor:
                proj_up = wf_up @ proj_gk.T
                proj_down = wf_down @ proj_gk.T
                datak = []
                for u, d in zip(proj_up.T, proj_down.T):
                    datak.append(u)
                    datak.append(d)
                data[ikirr] = np.array(datak).T
            else:
                data[ikirr] = wf[:, :, 0] @ proj_gk.T
        return AMN(data=data, NK=NK)

    def equals(self, other, tolerance=1e-8):
        iseq, message = super().equals(other, tolerance)
        if not iseq:
            return iseq, message
        if self.NW != other.NW:
            return False, f"the number of Wannier functions is not equal: {self.NW} and {other.NW} correspondingly"
        return True, ""

    def get_high_projectability(self, threshold=0.5, select_WF=None):
        """
        Get the maximum projection value over all k-points and bands
        """
        if select_WF is None:
            select_WF = range(self.NW)
        result = {}
        for ik, data in self.data.items():
            proj = (np.abs(data[:, select_WF])**2).sum(axis=1)
            print(f"ik={ik} proj = {proj}")
            result[ik] = (proj >= threshold)
        return result
