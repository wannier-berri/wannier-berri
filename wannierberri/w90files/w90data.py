#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#   some parts of this file are originate                    #
# from the translation of Wannier90 code                     #
# ------------------------------------------------------------#

from functools import cached_property
from copy import copy, deepcopy
import numpy as np
from ..wannierise import disentangle

from .win import WIN
from .eig import EIG
from .mmn import MMN
from .amn import AMN
from .xxu import UIU, UHU, SIU, SHU
from .spn import SPN
from .dmn import DMN
from .chk import CheckPoint, CheckPoint_bare



class Wannier90data:
    """A class to describe all input files of wannier90, and to construct the Wannier functions
     via disentanglement procedure

    Parameters:
        seedname : str
            the prefix of the file (including relative/absolute path, but not including the extensions, like `.chk`, `.mmn`, etc)	
        formatted : tuple(str)
            list of files which should be read as formatted files (uHu, uIu, etc)
        read_npz : bool
            if True, try to read the files converted to npz (e.g. wanier90.mmn.npz instead of wannier90.
        write_npz_list : list(str)
            for which files npz will be written
        write_npz_formatted : bool
            write npz for all formatted files
        overwrite_npz : bool
            overwrite existing npz files  (incompatinble with read_npz)
        read_chk : bool
            if True, read the checkpoint file, otherwise create a '~wannierberri.system.w90_files.CheckPoint_bare' object and prepare for disentanglement
        kmesh_tol : float
            see `~wannierberri.system.w90_files.CheckPoint`
        bk_complete_tol : float
            see `~wannierberri.system.w90_files.CheckPoint`

    Attributes
    ----------
    chk : `~wannierberri.system.w90_files.CheckPoint` or `~wannierberri.system.w90_files.CheckPoint_bare`
        the checkpoint file
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extensions, like `.chk`, `.mmn`, etc)
    wannierised : bool
        if True, the data is already wannierised (so, it can be used to create a System_w90 object)
    read_npz : bool
        if True, try to read the files converted to npz (e.g. wanier90.mmn.npz instead of wannier90.
    write_npz_list : set(str)
        for which files npz will be written
    write_npz_formatted : bool
        write npz for all formatted files
    formatted_list : list(str)
        list of files which should be read as formatted files (uHu, uIu, etc)
    _files : dict(str, `~wannierberri.system.w90_files.W90_file`)
        the dictionary of the files (e.g. the keys are 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn', 'dmn')
    """
    # todo :  rotate uHu and spn
    # todo : symmetry

    def __init__(self, seedname="wannier90", read_chk=False,
                 kmesh_tol=1e-7, bk_complete_tol=1e-5,
                 read_npz=True,
                 write_npz_list=('mmn', 'eig', 'amn', 'dmn'),
                 write_npz_formatted=True,
                 overwrite_npz=False,
                 formatted=tuple(),
                 files={},
                 ):  # ,sitesym=False):
        assert not (read_npz and overwrite_npz), "cannot read and overwrite npz files"
        self.seedname = copy(seedname)
        self.__files_classes = {'win': WIN,
                                'eig': EIG,
                                'mmn': MMN,
                                'amn': AMN,
                                'uiu': UIU,
                                'uhu': UHU,
                                'siu': SIU,
                                'shu': SHU,
                                'spn': SPN,
                                'dmn': DMN,
                                }
        self.read_npz = read_npz
        self.write_npz_list = set([s.lower() for s in write_npz_list])
        formatted = [s.lower() for s in formatted]
        if write_npz_formatted:
            self.write_npz_list.update(formatted)
            self.write_npz_list.update(['mmn', 'eig', 'amn', 'dmn'])
        self.formatted_list = formatted
        self._files = {}
        for key, val in files.items():
            self.set_file(key, val)

        if read_chk:
            self.chk = CheckPoint(seedname, kmesh_tol=kmesh_tol, bk_complete_tol=bk_complete_tol)
            self.wannierised = True
        else:
            self.chk = CheckPoint_bare(win=self.win, eig=self.eig, mmn=self.mmn, amn=self.amn)
            self.kpt_mp_grid = [tuple(k) for k in
                                np.array(np.round(self.chk.kpt_latt * np.array(self.chk.mp_grid)[None, :]),
                                         dtype=int) % self.chk.mp_grid]
            self.mmn.set_bk(mp_grid=self.chk.mp_grid, kpt_latt=self.chk.kpt_latt, recip_lattice=self.chk.recip_lattice)
            self.win_index = [np.arange(self.eig.NB)] * self.chk.num_kpts
            self.wannierised = False
        self.set_file(key='chk', val=self.chk)

    def set_file(self, key, val=None, overwrite=False,
                 **kwargs):
        """
        Set the file with the key `key` to the value `val`

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn', 'dmn'
        val : `~wannierberri.system.w90_files.W90_file`
            the value of the file
        overwrite : bool
            if True, overwrite the file if it was already set, otherwise raise an error
        kwargs : dict
            the keyword arguments to be passed to the constructor of the file
            see `~wannierberri.system.w90_files.W90_file`, 
            `~wannierberri.system.w90_files.MMN`, `~wannierberri.system.w90_files.EIG`, `~wannierberri.system.w90_files.AMN`, `~wannierberri.system.w90_files.UIU`, `~wannierberri.system.w90_files.UHU`, `~wannierberri.system.w90_files.SIU`, `~wannierberri.system.w90_files.SHU`, `~wannierberri.system.w90_files.SPN`
            for more details        
        """
        kwargs_auto = self.auto_kwargs_files(key)
        kwargs_auto.update(kwargs)
        if not overwrite and key in self._files:
            raise RuntimeError(f"file '{key}' was already set")
        if val is None:
            val = self.__files_classes[key](self.seedname, **kwargs_auto)
        self.check_conform(key, val)
        self._files[key] = val

    def write(self, seedname, files=None):
        """
        writes the files on disk

        Parameters
        ----------
        seedname : str
            the seedname of the files (including relative/absolute path, but not including the extensions, like `.chk`, `.mmn`, etc)
        files : list(str)
            the list of files extensions to be written. If None, all files are written
        """
        if files is None:
            files = self._files.keys()
        for key in files:
            self.get_file(key).to_w90_file(seedname)

    def auto_kwargs_files(self, key):
        """
        Returns the default keyword arguments for the file with the key `key`

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'

        Returns
        -------
        dict(str, Any)
            the keyword arguments for the file
        """
        kwargs = {}
        if key in ["uhu", "uiu", "shu", "siu"]:
            kwargs["formatted"] = key in self.formatted_list
        if key not in ["chk", "win"]:
            kwargs["read_npz"] = self.read_npz
            kwargs["write_npz"] = key in self.write_npz_list
        print(f"kwargs for {key} are {kwargs}")
        return kwargs


    def get_file(self, key, **kwargs):
        """
        Get the file with the key `key`

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'
        kwargs : dict
            the keyword arguments to be passed to the constructor of the file
            see `~wannierberri.system.w90_files.W90_file`, 
            `~wannierberri.system.w90_files.MMN`, `~wannierberri.system.w90_files.EIG`, `~wannierberri.system.w90_files.AMN`, `~wannierberri.system.w90_files.UIU`, `~wannierberri.system.w90_files.UHU`, `~wannierberri.system.w90_files.SIU`, `~wannierberri.system.w90_files.SHU`, `~wannierberri.system.w90_files.SPN`
            for more details

        Returns
        -------
        `~wannierberri.system.w90_files.W90_file`
            the file with the key `key`
        """
        if key not in self._files:
            self.set_file(key, **kwargs)
        return self._files[key]

    def check_conform(self, key, this):
        """
        Check if the file `this` conforms with the other files

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'
        this : `~wannierberri.system.w90_files.W90_file`
            the file to be checked

        Raises
        ------
        AssertionError
            if the file `this` does not conform with the other files
        """
        for key2, other in self._files.items():
            for attr in ['NK', 'NB', 'NW', 'NNB']:
                if hasattr(this, attr) and hasattr(other, attr):
                    a = getattr(this, attr)
                    b = getattr(other, attr)
                    if None not in (a, b):
                        assert a == b, f"files {key} and {key2} have different attribute {attr} : {a} and {b} respectively"

    @property
    def win(self):
        """
        Returns the WIN file
        """
        return self.get_file('win')

    @property
    def amn(self):
        """	
        Returns the AMN file
        """
        return self.get_file('amn')

    @property
    def eig(self):
        """
        Returns the EIG file
        """
        return self.get_file('eig')

    @property
    def mmn(self):
        """	
        Returns the MMN file
        """
        return self.get_file('mmn')

    @property
    def dmn(self):
        """
        Returns the DMN file
        """
        return self.get_file('dmn')

    @property
    def uhu(self):
        """	
        Returns the UHU file
        """
        return self.get_file('uhu')

    @property
    def uiu(self):
        """	
        Returns the UIU file
        """
        return self.get_file('uiu')

    @property
    def spn(self):
        """
        Returns the SPN file
        """
        return self.get_file('spn')

    @property
    def siu(self):
        """
        Returns the SIU file
        """
        return self.get_file('siu')

    @property
    def shu(self):
        """
        Returns the SHU file
        """
        return self.get_file('shu')

    @property
    def iter_kpts(self):
        """
        Returns the iterator over the k-points
        """
        return range(self.chk.num_kpts)

    @cached_property
    def wannier_centers(self):
        """
        Returns the Wannier centers stored in the checkpoint file
        """
        return self.chk.wannier_centers

    def check_wannierised(self, msg=""):
        """	
        Check if the system was wannierised

        Parameters
        ----------
        msg : str
            the message to be printed in case of error

        Raises
        ------
        RuntimeError
            if the system was not wannierised
        """
        if not (self.wannierised):
            raise RuntimeError(f"no wannieruisation was performed on the w90 input files, cannot proceed with {msg}")

    def disentangle(self, **kwargs):
        """
        Perform the disentanglement procedure calling `~wannierberri.system.disentangle`

        Parameters
        ----------
        kwargs : dict
            the keyword arguments to be passed to `~wannierberri.system.disentangle`     
        """
        disentangle(self, **kwargs)

    def get_disentangled(self, files=[]):
        """
        after disentanglement, get the Wannier90data object with 
        num_wann == num_bands

        Parameters
        ----------
        files : list(str)
            the extensions of the files to be read (e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn', 'dmn')
        """
        assert self.wannierised, "wannierisation was not performed"
        new_files = {}

        v = self.chk.v_matrix
        ham_tmp = np.einsum('kml,km,kmn->kln', v.conj(), self.eig.data, v)
        EV = [np.linalg.eigh(h_tmp) for h_tmp in ham_tmp]
        eig_new = EIG(data=np.array([ev[0] for ev in EV]))
        # v_right = np.array([_v @ ev[1].T.conj() for ev, _v in zip(EV,v)])
        v_right = np.array([_v @ ev[1] for ev, _v in zip(EV, v)])

        v_left = v_right.conj().transpose(0, 2, 1)


        for file in files:
            if file == 'eig':
                new_files['eig'] = eig_new
            else:
                new_files[file] = self.get_file(file).get_disentangled(v_left, v_right)
                if file == 'mmn':
                    new_files[file].neighbours = self.mmn.neighbours
                    new_files[file].ib_unique_map = self.mmn.ib_unique_map
                    new_files[file].G = self.mmn.G
        win = deepcopy(self.win)
        win.NB = self.chk.num_wann
        new_files['win'] = win
        other = Wannier90data(read_chk=False, files=new_files)
        return other

    def check_symmetry(self, silent=False):
        """
        Check the symmetry of the system
        """

        err_eig = self.dmn.check_eig(self.eig)
        err_amn = self.dmn.check_amn(self.amn)
        if not silent:
            print(f"eig symmetry error : {err_eig}")
            print(f"amn symmetry error : {err_amn}")
        return err_eig, err_amn

    # TODO : allow k-dependent window (can it be useful?)
    # def apply_outer_window(self,
    #                  win_min=-np.Inf,
    #                  win_max=np.Inf ):
    #     raise NotImplementedError("outer window does not work so far")
    #     "Excludes the bands from outside the outer window"
    #
    #     def win_index_nondegen(ik,thresh=DEGEN_THRESH):
    #         "define the indices of the selected bands, making sure that degenerate bands were not split"
    #         E=self.Eig[ik]
    #         ind=np.where( ( E<=win_max)*(E>=win_min) )[0]
    #         while ind[0]>0 and E[ind[0]]-E[ind[0]-1]<thresh:
    #             ind=[ind[0]-1]+ind
    #         while ind[0]<len(E) and E[ind[-1]+1]-E[ind[-1]]<thresh:
    #             ind=ind+[ind[-1]+1]
    #         return ind
    #
    #     # win_index_irr=[win_index_nondegen(ik) for ik in self.Dmn.kptirr]
    #     # self.excluded_bands=[list(set(ind)
    #     # self.Dmn.select_bands(win_index_irr)
    #     # win_index=[win_index_irr[ik] for ik in self.Dmn.kpt2kptirr]
    #     win_index=[win_index_nondegen(ik) for ik in self.iter_kpts]
    #     self._Eig=[E[ind] for E, ind in zip(self._Eig,win_index)]
    #     self._Mmn=[[self._Mmn[ik][ib][win_index[ik],:][:,win_index[ikb]] for ib,ikb in enumerate(self.mmn.neighbours[ik])] for ik in self.iter_kpts]
    #     self._Amn=[self._Amn[ik][win_index[ik],:] for ik in self.iter_kpts]