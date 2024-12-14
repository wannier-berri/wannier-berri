from datetime import datetime
from functools import cached_property
import multiprocessing

import numpy as np
from scipy import special
from .utility import str2arraymmn
from .w90file import W90_file
from irrep.bandstructure import BandStructure


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

    def apply_window(self, selected_bands):
        print(f"apply_window amn, selected_bands={selected_bands}")
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
        super().__init__(seedname, ext="amn", npar=npar, **kwargs)

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

    def get_disentangled(self, v_left, v_right):
        print(f"v shape  {v_left.shape}  {v_right.shape} , amn shape {self.data.shape} ")
        data = np.einsum("klm,kmn->kln", v_left, self.data)
        print(f"shape of data {data.shape} , old {self.data.shape}")
        return self.__class__(data=data)

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
    So far only delta-localised s-orbitals are implemented

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
        igk = kp.ig[:3,:]+kp.k[:,None]
        exppgk = np.exp(-2j*np.pi*(pos @ igk))
        wf = kp.WF.conj()
        if normalize:
            wf /= np.linalg.norm(wf, axis=1)[:,None]
        data.append(wf @ exppgk.T)
    data = np.array(data)
    if return_object:
        return AMN(data=data)
    else:
        return data
    

def amn_from_bandstructure(bandstructure: BandStructure, positions, orbitals, normalize=True, return_object=True):
    """
    Create an AMN object from a BandStructure object
    So far only delta-localised s-orbitals are implemented

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
    rec_latt = bandstructure.RecLattice
    needed_lm = set()
    orb_to_l = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    for orb in orbitals:
        if orb in orb_to_l:
            l = orb_to_l[orb]
            for m in range(-l, l+1):
                needed_lm.add((l, m))
        else:
            raise ValueError(f"orbital {orb} not implemented")
    needed_l = set(l for l, m in needed_lm)
    needed_m = set(m for l, m in needed_lm)
    for kp in bandstructure.kpoints:
        igk = kp.ig[:3,:]+kp.k[:,None]
        exppgk = np.exp(-2j*np.pi*(pos @ igk))
        wf = kp.WF.conj()
        if normalize:
            wf /= np.linalg.norm(wf, axis=1)[:,None]
        datak = wf @ exppgk.T
        gk = igk.T @ rec_latt
        g_abs = np.linalg.norm(gk, axis=1)
        g_costheta = gk[:,2]/g_abs
        g_sintheta = np.sqrt(1-g_costheta**2)
        g_phi = np.arctan2(gk[:,1], gk[:,0])
        exp_phi_m = {}
        if 0 in needed_m:
            exp_phi_m[0] = np.ones(g_phi.shape[0])
        if len (needed_m- set([0])) > 0:
            exp_phi_m[1] = np.exp(1j*g_phi)
            for m in needed_m - set([0, 1]):
                exp_phi_m[m] = exp_phi_m[1]**m
        
            
        
    data = np.array(data)
    if return_object:
        return AMN(data=data)
    else:
        return data
    

class LegendrePolynomials:
    def __init__(self, costheta, sintheta):
        self.costheta = costheta
        self.sintheta = sintheta
        self.P = {}

    def __call__(self, l, m):
        if (l, m) not in self.P:
            self.P[(l, m)] = self._legendre(l, m)
        return self.P[(l, m)]
    
    def _legendre(self, l, m):
        if m < 0:
            return (-1)**m * np.sqrt(2) * self._legendre(l, -m)
        if m == 0:
            return np.real(special.lpmv(m, l, self.costheta))
        else:
            return np.sqrt(2) * np.real(special.lpmv(m, l, self.costheta) * np.cos(m*self.sintheta))
    

from scipy.special import spherical_jn
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid

class Bessel_j_exp_int:
    """
    a class to evaluate the integral

    :math:`\int_0^{\infty} j_l(k*x) e^{-x} dx`
    """

    def __init__(self, l_list, 
                 k0=5, kmax=100, dk=0.01, dtk=0.2, kmin = 1e-3,
                 x0=5, xmax=100, dx=0.01, dtx=0.2,
                 ):
        self.splines = {}
        self.kmax = kmax
        self.kgrid = self._get_grid(k0, kmax, dk, dtk)
        self.xgrid = self._get_grid(x0, xmax, dx, dtx)
        print (f"the xgrid has {len(self.xgrid)} points")
        print (f"the kgrid has {len(self.kgrid)} points")
        self.kmin = kmin
        for l in l_list:
            self.set_spline(l)
        
    def _get_grid(self, x0, xmax, dx, dt):
        xgrid = list(np.arange(0, x0, dx ))
        t=dt
        x0 = xgrid[-1]
        while xgrid[-1] < xmax:
            xgrid.append(x0*np.exp(t))
            t+=dt
        return np.array(xgrid)
        
    def set_spline(self, l):
        if l not in self.splines:
            self.splines[l] = self.get_spline(l)
        return self.splines[l]
    
    def get_spline(self, l):
        e = np.exp(-self.xgrid)
        fourier = []
        for k in self.kgrid:
            if k<self.kmin and l>0:
                fourier.append(0)
            else:
                j = spherical_jn(l, k*self.xgrid)
                fourier.append( trapezoid(y=j * e, x=self.xgrid) )
        return CubicSpline(self.kgrid, fourier)

        
    def __call__(self, l, k):
        self.set_spline(l)
        res = np.zeros(len(k))
        select = (k<=self.kmax)
        res[select] = self.splines[l](k[select])
        return res

class SphericalHarmonics:

    def __init__(self, costheta, phi):
        self.costheta = costheta
        self.sintheta = np.sqrt(1-costheta**2)
        self.phi = phi
        self.harmonics = {}
        self.sqpi = 1/np.sqrt(np.pi)
        
    @cached_property
    def cosphi(self):
        return np.cos(self.phi)
    
    @cached_property
    def sinphi(self):
        return np.sin(self.phi)
    
    @cached_property
    def sin2phi(self):
        return 2*self.sinphi*self.cosphi
    
    @cached_property
    def cos2phi(self):
        return 2*self.cosphi**2 - 1
    
    @cached_property
    def cos2theta(self):
        return 2*self.costheta**2 - 1
    
    @cached_property
    def sin2theta(self):
        return 2*self.costheta*self.sintheta
    
    def __call__(self, orbital):
        if orbital not in self.harmonics:
            self.harmonics[orbital] = self._harmonics(orbital)
        return self.harmonics[orbital]
    
    def _harmonics(self, orbital):
        """from here https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        
        hybrids - according to Wannier90 manual"""
        from numpy import sqrt as sq
        from numpy import pi
        match orbital:
            case 's':
                return 1/(2*sq(pi))
            case 'pz':
                return sq(3/(4*pi)) * self.costheta
            case 'px':
                return sq(3/(4*pi)) * self.sintheta * self.cosphi
            case 'py':
                return sq(3/(4*pi)) * self.sintheta * self.sinphi
            case 'dz2':
                return sq(5/(16*pi)) * (3*self.costheta**2 - 1)
            case 'dx2_y2':
                return sq(15/(16*pi)) * self.sintheta**2 * self.cos2phi
            case 'dxy':
                return sq(15/(16*pi)) * self.sintheta**2 * self.sin2phi        
            case 'dxz':
                return sq(15/(16*pi)) * self.sin2theta * self.cosphi
            case 'dyz':
                return sq(15/(16*pi)) * self.sin2theta * self.sinphi
            case 'sp-1':
                return (self('s')+self('px'))/sq(2)
            case 'sp-2':
                return (self('s')-self('px'))/sq(2)
            case 'sp2-1':
                return self('s')/sq(3) - self('px')/sq(6) + self('py')/sq(2)
            case 'sp2-2':
                return self('s')/sq(3) - self('px')/sq(6) - self('py')/sq(2)
            case 'sp2-3':
                return self('s'/sq(3) + 2*self('px')/sq(6))
            case 'sp3-1':
                return (self('s')+self('px')+self('py')+self('pz'))/2
            case 'sp3-2':
                return (self('s')+self('px')-self('py')-self('pz'))/2
            case 'sp3-3':
                return (self('s')-self('px')+self('py')-self('pz'))/2
            case 'sp3-4':
                return (self('s')-self('px')-self('py')+self('pz'))/2
            case 'sp3d2-1':
                return self('s')/sq(6) - self('px')/sq(2)-self('dz2')/sq(12)+self('dx2_y2')/2
            case 'sp3d2-2':
                return self('s')/sq(6) + self('px')/sq(2)-self('dz2')/sq(12)+self('dx2_y2')/2
            case 'sp3d2-3':
                return self('s')/sq(6) - self('py')/sq(2)-self('dz2')/sq(12)+self('dx2_y2')/2
            case 'sp3d2-4':
                return self('s')/sq(6) + self('py')/sq(2)-self('dz2')/sq(12)+self('dx2_y2')/2
            case 'sp3d2-5':
                return self('s')/sq(6) - self('pz')/sq(2)+self('dz2')/sq(3)
            case 'sp3d2-6':
                return self('s')/sq(6) + self('pz')/sq(2)+self('dz2')/sq(3)
            


