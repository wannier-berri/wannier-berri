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

import datetime
from functools import cached_property
from copy import copy
import os
import warnings
import numpy as np

from wannierberri.utility import cached_einsum
from ..wannierise import wannierise
from ..symmetry.sawf import SymmetrizerSAWF
from .utility import grid_from_kpoints

from .win import WIN
from .eig import EIG
from .mmn import MMN
from .amn import AMN
from .xxu import UIU, UHU, SIU, SHU
from .spn import SPN
from .unk import UNK
from .chk import CheckPoint

FILES_CLASSES = {'win': WIN,
                'eig': EIG,
                'mmn': MMN,
                'amn': AMN,
                'uiu': UIU,
                'uhu': UHU,
                'siu': SIU,
                'shu': SHU,
                'spn': SPN,
                'unk': UNK,
                'chk': CheckPoint,
                }


class Wannier90data:
    """A class to describe all input files of wannier90, and to construct the Wannier functions

    Parameters:
        seedname : str
            the prefix of the file (including relative/absolute path, but not including the extensions, like `.chk`, `.mmn`, etc)	
        formatted : tuple(str)
            list of files which should be read as formatted files (uHu, uIu, etc)
        read_npz : bool
            if True, try to read the files converted to npz (e.g. wanier90.mmn.npz instead of wannier90.mmn)
        write_npz_list : list(str)
            for which files npz will be written
        write_npz_formatted : bool
            write npz for all formatted files
        overwrite_npz : bool
            overwrite existing npz files  (incompatinble with read_npz)
        read_chk : bool
            if True, read the :class:`~wannierberri.w90files.CheckPoint` file, 
            otherwise create a "bare" :class:`~wannierberri.w90files.chk.CheckPoint` object and prepare for wannierisation
        kmesh_tol : float
            see :class:`~wannierberri.w90files.chk.CheckPoint`
        bk_complete_tol : float
            see :class:`~wannierberri.w90files.chk.CheckPoint`mp_grid

    Attributes
    ----------
    chk : `~wannierberri.w90files.CheckPoint` or `~wannierberri.w90files.CheckPoint_bare`
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
    readfiles : list(str)
        the list of files to be read durint the initialization. Others may be read later.
    """
    # todo :  rotate uHu and spn
    # todo : symmetry

    def __init__(self, ):
        self.bands_were_selected = False
        self.irreducible = False
        self._files = {}

    def from_bandstructure(self, bandstructure,
                          seedname="wannier90",
                          files=("mmn", "eig", "amn", "symmetrizer"),
                          read_npz_list=None,
                          write_npz_list=None,
                          projections=None,
                          unk_grid=None,
                          normalize=True,
                          irreducible=None,
                          ecut_sym=100,
                          mp_grid=None,):
        """
        Create a Wannier90data object from a bandstructure object

        Parameters
        ----------
        bandstructure : `irrep.bandstructure.BandStructure`
            the bandstructure object contself.data = np.array(data)aining the kpoints, lattice, and number of bands
        seedname : str
            the prefix of the file (including relative/absolute path, but not including the extensions, like `.chk`, `.mmn`, etc) to save the npz files (if write_npz_list is not empty)
        files : tuple(str)
            the list of files to be created. Possible values are 'mmn', 'eig', 'amn', 'symmetrizer', 'unk'
        write_npz_list : list(str)
            the list of files to be written as npz files. If empty, no files are written as npz
        projections : `~wannierberri.symmetry.projections.ProjectionSet`
            the projections to be used for the AMN and symmetrizer files.
        unk_grid : tuple(int)
            the grid to be used for the UNK file. If None, the grid is calculated from the g-vectors
        normalize : bool
            if True, the wavefunctions are normalized to have unit norm.

        Returns
        -------
        Wannier90data
            the Wannier90data object containing the files specified in `files`

        """
        print(f"got irreducible={irreducible}, mp_grid={mp_grid}, seedname={seedname}, files={files}, read_npz_list={read_npz_list}, write_npz_list={write_npz_list}, projections={projections}, unk_grid={unk_grid}, normalize={normalize}")
        if irreducible is None:
            from irrep.utility import grid_from_kpoints as grid_from_kpoints_irrep
            kpt_latt_grid = np.array([KP.K  for KP in bandstructure.kpoints])
            print(f"kpt_latt_grid={kpt_latt_grid}")
            grid, selected_kpoints = grid_from_kpoints_irrep(kpt_latt_grid, grid=None, allow_missing=True)
            print(f"detected grid={grid}, selected_kpoints={selected_kpoints}")
            if len(selected_kpoints) < np.prod(grid):
                warnings.warn(f"detected grid {grid} od {np.prod(grid)} kpoints, "
                              f"but only {len(selected_kpoints)} kpoints are available."
                              "assuming that only irreducible kpoints are needed.")
                irreducible = True
            else:
                irreducible = False
        self.irreducible = irreducible
        print(f"self.irreducible={self.irreducible}")


        self.seedname = copy(seedname)

        if self.irreducible:
            if "symmetrizer" not in files:
                warnings.warn("irreducible=True, but symmetrizer is not requested. Adding it automatically")
                files = list(files) + ["symmetrizer"]

        if read_npz_list is None:
            read_npz_list = files
        if write_npz_list is None:
            write_npz_list = files
        read_npz_list = set([s.lower() for s in read_npz_list])
        write_npz_list = set([s.lower() for s in write_npz_list])



        if "symmetrizer" in files:
            fname = seedname + ".symmetrizer.npz"
            symmetrizer_read_ok = False
            if "symmetrizer" in read_npz_list:
                try:
                    symmetrizer = SymmetrizerSAWF().from_npz(fname)
                    symmetrizer_read_ok = True
                except Exception as e:
                    warnings.warn(f"Failed to read symmetrizer from {fname}: {e}")
            if not symmetrizer_read_ok:
                symmetrizer = SymmetrizerSAWF().from_irrep(bandstructure,
                                                           grid=mp_grid,
                                                           irreducible=irreducible,
                                                           ecut=ecut_sym)
                if projections is not None:
                    symmetrizer.set_D_wann_from_projections(projections)
            self.set_symmetrizer(symmetrizer)
            if "symmetrizer" in write_npz_list and not symmetrizer_read_ok:
                symmetrizer.to_npz(seedname + ".symmetrizer.npz")

        if self.has_symmetrizer:
            kpt_latt = self.symmetrizer.kpoints_all
            mp_grid = self.symmetrizer.grid
            if hasattr(self.symmetrizer, "selected_kpoints"):
                selected_kpoints = self.symmetrizer.selected_kpoints
            else:
                selected_kpoints = np.arange(bandstructure.num_kpts)
        else:
            kpt_latt = np.array([kp.k for kp in bandstructure.kpoints])
            mp_grid = grid_from_kpoints(kpt_latt)
            selected_kpoints = None
        if irreducible:
            kptirr = self.symmetrizer.kptirr
            kpt_from_kptirr_isym = self.symmetrizer.kpt_from_kptirr_isym
            kpt2kptirr = self.symmetrizer.kpt2kptirr
            NK = np.prod(mp_grid)
        else:
            kptirr = None
            kpt_from_kptirr_isym = None
            kpt2kptirr = None
            NK = None
        kwargs_bandstructure = {"selected_kpoints":
                        selected_kpoints, "kptirr": kptirr,
                        "NK": NK}



        if "chk" in read_npz_list and os.path.exists(seedname + ".chk.npz"):
            chk = CheckPoint.from_npz(seedname + ".chk.npz")
        else:
            chk = CheckPoint(real_lattice=bandstructure.lattice,
                             num_wann=projections.num_wann,
                             num_bands=bandstructure.num_bands,
                             kpt_latt=kpt_latt,
                             mp_grid=mp_grid,)

        self.set_file('chk', chk)
        if "eig" in files:
            eig = EIG.autoread(seedname=seedname, read_npz=("eig" in read_npz_list),
                               read_w90=False,
                               write_npz="eig" in write_npz_list,
                               bandstructure=bandstructure,
                               kwargs_bandstructure=kwargs_bandstructure
                               )
            self.set_file('eig', eig)
        if "amn" in files:
            amn = AMN.autoread(seedname=seedname, read_npz=("amn" in read_npz_list),
                               read_w90=False,
                               write_npz="amn" in write_npz_list,
                               bandstructure=bandstructure,
                               kwargs_bandstructure={"normalize": normalize,
                                                     "projections": projections} |
                               kwargs_bandstructure)
            self.set_file('amn', amn)
        if "mmn" in files:

            mmn = MMN.autoread(seedname=seedname, read_npz=("mmn" in read_npz_list),
                               read_w90=False,
                               write_npz="mmn" in write_npz_list,
                               bandstructure=bandstructure,
                               kwargs_bandstructure={"normalize": normalize,
                                                     "kpt_latt_grid": kpt_latt,
                                                     "kpt2kptirr": kpt2kptirr,
                                                     "kpt_from_kptirr_isym": kpt_from_kptirr_isym} |
                               kwargs_bandstructure)
            self.set_file('mmn', mmn)
        if "spn" in files:
            spn = SPN.autoread(seedname=seedname, read_npz=("spn" in read_npz_list),
                               read_w90=False,
                               write_npz="spn" in write_npz_list,
                               bandstructure=bandstructure,
                               kwargs_bandstructure={"normalize": normalize} | kwargs_bandstructure)
            self.set_file('spn', spn)
        # TODO : use a cutoff ~100eV for symmetrizer
        if "unk" in files:
            unk = UNK.autoread(seedname=seedname, read_npz=("unk" in read_npz_list),
                               read_w90=False,
                               write_npz="unk" in write_npz_list,
                               bandstructure=bandstructure,
                               kwargs_bandstructure={"normalize": normalize,
                                                     "grid_size": unk_grid} |
                               kwargs_bandstructure)
            self.set_file('unk', unk)
        return self

    def from_npz(self,
                seedname="wannier90",
                files=("mmn", "eig", "amn"),
                irreducible=False):
        for f in files:
            try:
                if f == "symmetrizer":
                    val = SymmetrizerSAWF().from_npz(seedname + ".symmetrizer.npz")
                elif f in FILES_CLASSES:
                    val = FILES_CLASSES[f].from_npz(seedname + "." + f + ".npz")
                else:
                    raise ValueError(f"file {f} is not a valid w90 file")
                self.set_file(f, val=val)
            except FileNotFoundError as e:
                warnings.warn(f"file {seedname}.{f}.npz not found, cannot read {f} file ({e}).\n Set it manually, if needed")
        for f in self._files:
            ff = self.get_file(f)
            if f == "symmetrizer":
                continue
            if f == "chk":
                if hasattr(ff, "v_matrix") and ff.v_matrix is not None:
                    nkeys = len(ff.v_matrix.keys())
                    NK = ff.num_kpts
                else:
                    continue
            else:
                nkeys = len(ff.data.keys())
                NK = ff.NK
            if nkeys < NK:
                warnings.warn(f"file {f} cntains {nkeys} k-points less than NK ({NK}) , "
                              "so we assume the files contain only on irreducible k-points")
                irreducible = True
        self.irreducible = irreducible
        return self



    def to_npz(self,
               seedname="wannier90",
               files=None
               ):
        if files is None:
            files = self._files.keys()
        for f in files:
            if f in self._files:
                self.get_file(f).to_npz(seedname + "." + f + ".npz")
            else:
                warnings.warn(f"file {f} is not set, cannot write to npz")



    @property
    def num_kpts(self):
        """
        Returns the number of k-points in the system
        """
        return self.chk.num_kpts

    @property
    def kptirr_system(self):
        """
        Returns the list of kptirr to iterate for construction of the system, nad the weight for each kptirr

        Returns
        -------
        kptirr : list of int
            the list of kptirr to iterate for construction of the system
        weight : list of float
            the weight for each kptirr
        """
        if not self.irreducible:
            return np.arange(self.num_kpts), np.ones(self.num_kpts)
        kptirr = self.symmetrizer.kptirr
        weight = self.symmetrizer.kptirr_weights
        return kptirr, weight


    def from_w90_files(self, seedname="wannier90",
                     read_npz=True,
                     write_npz_list=('mmn', 'eig', 'amn'),
                     write_npz_formatted=True,
                     overwrite_npz=False,
                     formatted=tuple(),
                     readfiles=tuple(),
                     ):
        assert not (read_npz and overwrite_npz), "cannot read and overwrite npz files"
        self.seedname = copy(seedname)
        # self.read_npz = read_npz
        self.write_npz_list = set([s.lower() for s in write_npz_list])
        formatted = [s.lower() for s in formatted]
        if write_npz_formatted:
            self.write_npz_list.update(formatted)
            self.write_npz_list.update(['mmn', 'eig', 'amn'])
        self.formatted_list = formatted

        _read_files_loc = [f.lower() for f in readfiles]
        assert 'win' in _read_files_loc or 'chk' in _read_files_loc, "either 'win' or 'chk' should be in readfiles"
        if 'win' in _read_files_loc:
            win = WIN(seedname=seedname, autoread=True)
            self.set_file('win', win, read_npz=read_npz)
            _read_files_loc.remove('win')
        if 'chk' in _read_files_loc:
            self.set_chk(read=True)
            _read_files_loc.remove('chk')
        else:
            self.set_chk(read=False)
        if 'mmn' in _read_files_loc:
            self.set_file('mmn', kwargs_w90=dict(kpt_latt=self.chk.kpt_latt, recip_lattice=self.chk.recip_lattice), read_npz=read_npz)
            _read_files_loc.remove('mmn')
        for f in _read_files_loc:
            self.set_file(f, read_npz=read_npz)
        return self

    @property
    def num_wann(self):
        return self.chk.num_wann

    @property
    def mp_grid(self):
        return self.chk.mp_grid

    @property
    def kpt_latt(self):
        """ Returns the k-points or the grid in lattice coordinates
        """
        return self.chk.kpt_latt

    @cached_property
    def atomic_positions_red(self):
        """
        Returns the atomic positions in reduced coordinates
        """
        if not hasattr(self, "_atomic_positions_red"):
            win = self.get_file("win")
            if "atoms_frac" in win:
                self._atomic_positions_red = win.atoms_frac
            elif "atoms_cart" in win:
                self._atomic_positions_red = win.atoms_cart @ np.linalg.inv(win.lattice)
            self._atomic_positions_red = win.atomic_positions
        return self.chk.atomic_positions_red

    def set_atomic_positions_red(self, atomic_positions_red):
        self._atomic_positions_red = atomic_positions_red

    def get_spacegroup(self):
        """
        Get the spacegroup of the system from the symmetrizer object
        if the symmetrizer is not set return None

        Returns
        -------
        `~irrep.spacegroup.SpaceGroupBare`
            the spacegroup of the system
        """
        if self.has_file('symmetrizer') and self.symmetrizer is not None:
            return self.symmetrizer.spacegroup
        else:
            return None

    def set_symmetrizer(self, symmetrizer=None,
                        overwrite=True,
                        allow_selected_bands=False,
                        read_npz=False):
        """
        Set the symmetrizer of the system

        Parameters
        ----------
        symmetrizer : `~wanierberri.symmetry.symmetrizer_sawf.SymmetrizerSAWF`
            the symmetrizer object
        """
        self.set_file("symmetrizer", val=symmetrizer,
                      overwrite=overwrite,
                      allow_selected_bands=allow_selected_bands,
                      read_npz=read_npz,)

    @property
    def symmetrizer(self):
        """
        Returns the symmetrizer object of the system

        Returns
        -------
        `~wannierberri.symmetry.symmetrizer_sawf.SymmetrizerSAWF`
            the symmetrizer object
        """
        return self.get_file("symmetrizer")

    @property
    def has_symmetrizer(self):
        """
        Check if the symmetrizer is set

        Returns
        -------
        bool
            True if the symmetrizer is set, False otherwise
        """
        return self.has_file("symmetrizer")

    def set_file(self, key, val=None, overwrite=False, allow_selected_bands=False,
                 read_npz=True,
                 kwargs_w90={},
                 **kwargs):
        """
        Set the file with the key `key` to the value `val`

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'
        val : `~wannierberri.w90files.W90_file`
            the value of the file. If None, the file is read from the disk: first try from the npz file (if available), then from the w90 file
        overwrite : bool
            if True, overwrite the file if it was already set, otherwise raise an error
        allow_selected_bands : bool
            if True, allow to set the file even if the bands were already selected, otherwise raise an error
        kwargs : dict
            the keyword arguments to be passed to the constructor of the file
            see `~wannierberri.w90files.W90_file`, 
            `~wannierberri.w90files.MMN`, `~wannierberri.w90files.EIG`, `~wannierberri.w90files.AMN`, `~wannierberri.w90files.UIU`, `~wannierberri.w90files.UHU`, `~wannierberri.w90files.SIU`, `~wannierberri.w90files.SHU`, `~wannierberri.w90files.SPN`
            for more details        
        """
        if self.bands_were_selected:
            if allow_selected_bands:
                warnings.warn("window was applied, so new added files may be inconsistent with the window. It is your responsibility to check it")
            else:
                raise RuntimeError("window was applied, so new added files may be inconsistent with the window. To allow it, set allow_selected_bands=True (on your own risk)")
        if key == "chk":
            self.set_chk(val=val, read=True, overwrite=overwrite, **kwargs)
            return
        if not overwrite and self.has_file(key):
            raise RuntimeError(f"file '{key}' was already set")
        if val is None:
            if key not in FILES_CLASSES:
                raise ValueError(f"key '{key}' is not a valid w90 file")
            # kwargs_auto = self.auto_kwargs_files(key)
            # kwargs_auto.update(kwargs)
            kwargs_w90 = copy(kwargs_w90)
            if key in ['uhu', 'uiu', 'shu', 'siu']:
                assert self.has_file('mmn'), "cannot read uHu/uIu/sHu/sIu without mmn file"
                assert self.has_file('chk'), "cannot read uHu/uIu/sHu/sIu without chk file"
                kwargs_w90['bk_reorder'] = self.get_file('mmn').bk_reorder

            val = FILES_CLASSES[key].autoread(self.seedname, read_npz=read_npz, kwargs_w90=kwargs_w90)
        self.check_conform(key, val)
        if key == 'amn' and self.has_file('chk'):
            self.get_file('chk').num_wann = val.NW
        self._files[key] = val

    def has_file(self, key):
        """
        Check if the file with the key `key` is set

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'

        Returns
        -------
        bool
            True if the file is set, False otherwise
        """
        return key in self._files

    def set_chk(self, val=None, kmesh_tol=1e-7, bk_complete_tol=1e-5, read=False, overwrite=False):
        if not overwrite and self.has_file("chk"):
            raise RuntimeError("chk file was already set")
        if val is None:
            if read:
                val = CheckPoint().from_w90_file(self.seedname, kmesh_tol=kmesh_tol, bk_complete_tol=bk_complete_tol)
            else:
                val = CheckPoint().from_win(win=self.win)

        self._files['chk'] = val
        self.wannierised = read
        self.kpt_mp_grid = [tuple(k) for k in
                            np.array(np.round(self.chk.kpt_latt * np.array(self.chk.mp_grid)[None, :]),
                                     dtype=int) % self.chk.mp_grid]
        self.win_index = [np.arange(self.chk.num_bands)] * self.chk.num_kpts


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

    # def auto_kwargs_files(self, key):
    #     """
    #     Returns the default keyword arguments for the file with the key `key`

    #     Parameters
    #     ----------
    #     key : str
    #         the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'

    #     Returns
    #     -------
    #     dict(str, Any)
    #         the keyword arguments for the file
    #     """
    #     kwargs = {}
    #     if key in ["uhu", "uiu", "shu", "siu"]:
    #         kwargs["formatted"] = key in self.formatted_list
    #     if key not in ["chk", "win", "unk"]:
    #         kwargs["read_npz"] = self.read_npz
    #         kwargs["write_npz"] = key in self.write_npz_list
    #     if key == "chk":
    #         kwargs["bk_complete_tol"] = 1e-5
    #         kwargs["kmesh_tol"] = 1e-7
    #     print(f"kwargs for {key} are {kwargs}")
    #     return kwargs


    def get_file(self, key):
        """
        Get the file with the key `key`

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'

        Returns
        -------
        `~wannierberri.w90files.W90_file`
            the file with the key `key`
        """
        # if key not in self._files:
        #     self.set_file(key, **kwargs)
        if key not in self._files:
            raise RuntimeError(f"file '{key}' was not set. Note : implicit set of files is not allowed anymore. Please use set_file() method of the `readfiles` parameter of the constructor")
        return self._files[key]

    def check_conform(self, key, this):
        """
        Check if the file `this` conforms with the other files

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'
        this : `~wannierberri.w90files.W90_file`
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
    def chk(self):
        """
        Returns the checkpoint file
        """
        return self.get_file('chk')

    @property
    def shu(self):
        """
        Returns the SHU file
        """
        return self.get_file('shu')

    @property
    def unk(self):
        """
        Returns the UNK files
        """
        return self.get_file('unk')

    @property
    def iter_kpts(self):
        """
        Returns the iterator over the k-points
        """
        return range(self.chk.num_kpts)

    @cached_property
    def wannier_centers_cart(self):
        """
        Returns the Wannier centers stored in the checkpoint file
        """
        return self.chk.wannier_centers_cart

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
            raise RuntimeError(f"no wannierisation was performed on the w90 input files, cannot proceed with {msg}")

    def wannierise(self, **kwargs):
        """
        Perform the wannierisation procedure calling `~wannierberri.wannierise.wannierise`

        Parameters
        ----------
        kwargs : dict
            the keyword arguments to be passed to `~wannierberri.wannierise.wannierise`
        """
        wannierise(self,
                   irreducible=self.irreducible,
                   **kwargs)


    def apply_window(self, *args, **kwargs):
        raise NotImplementedError("apply_window is deprecated. Use select_bands instead")

    def select_bands(self,
                     win_min=-np.inf, win_max=np.inf,
                     band_start=None, band_end=None,
                     selected_bands=None,
                     allow_again=False,
                     verbose=False):
        """
        exclude some bands from the data

        Parameters
        ----------
        win_min, win_max : float
            the minimum and maximum energy of the window. Only the bands that are ENTIRELY below win_min or ENTIRELY above win_max are EXCLUDED
        band_start, band_end : int
            the range of bands to be included (the numeration in the current system, not in the original (if bands are selected more than once))
            note : the pythonic indexing is used, so the included bands are [band_start, band_start+1, ... band_end-1]
        selected_bands : array((NB,), dtype=bool) or list of int
            if present, the those bands are selected, and the other parameters are ignored
        allow_again : bool
            if False, but bands were already selected, raise an error. 

        Returns
        -------
        new_selected_bands : list of int
            the indices of bands that were actually selected. If data files are added to the dataset later, one should manually select those bands 
            before adding, using the `select_bands` method of those objects
        """
        if self.bands_were_selected and not allow_again:
            raise RuntimeError("bands were already selected, cannot select them again. To allow it, set allow_again=True (on your own risk)")
        if selected_bands is not None:
            if isinstance(selected_bands, list):
                selected_bands = np.array(selected_bands)
            if selected_bands.dtype == np.bool:
                selected_bands = np.where(selected_bands)[0]
                assert selected_bands.shape == (self.eig.NB,), f"selected bands should be of shape (NB,), not {selected_bands.shape}"
            else:
                assert selected_bands.dtype == np.int, f"selected_bands should be a list of int or a boolean array, not {selected_bands.dtype}"
                assert np.all(selected_bands >= 0), f"selected bands should be non-negative, not {selected_bands}"
                assert np.all(selected_bands < self.eig.NB), f"selected bands should be less than {self.eig.NB}, not {selected_bands}"
        else:
            selected_bands_bool = np.ones((self.eig.NB), dtype=bool)
            if band_end is not None:
                selected_bands_bool[band_end:] = False
            if band_start is not None:
                selected_bands_bool[:band_start] = False
            if win_min > -np.inf or win_max < np.inf:
                assert self.has_file('eig'), "eig file is not set - needed to apply window"
                select_energy = [(E < win_max) * (E > win_min) for E in self.eig.data.values()]
                select_energy = np.any(select_energy, axis=0)
                selected_bands_bool = selected_bands_bool * select_energy
            selected_bands = np.where(selected_bands_bool)[0]

        print(f"selected_bands = {selected_bands}")
        if verbose:
            print("before selecting bands")
            for key, val in self._files.items():
                if key != 'win' and key != 'chk':
                    print(f"key = {key} ,number of bands = {val.NB}")
                if key == 'chk':
                    print(f"key = {key} ,number of bands = {val.num_bands}")
        for key in FILES_CLASSES:
            if key in self._files:
                if key == 'win':
                    win = self._files['win']
                    win['dis_win_min'] = win_min
                    win['dis_win_max'] = win_max
                else:
                    # print(f"applying window to {key} {val}")
                    self.get_file(key).select_bands(selected_bands)
        if self.has_file('symmetrizer'):
            self.symmetrizer.select_bands(selected_bands)
        if verbose:
            print("after selecting bands")
            for key, val in self._files.items():
                if key != 'win' and key != 'chk':
                    print(f"key = {key} ,number of bands = {val.NB}")
                    if hasattr(val, 'data') and key != 'unk':
                        print(f"key = {key} ,shape of data= {[d.shape for d in val.data.values()]}")
                if key == 'chk':
                    print(f"key = {key} ,number of bands = {val.num_bands}")
        self.bands_were_selected = True
        return selected_bands


    def check_symmetry(self, silent=False):
        """
        Check the symmetry of the system
        """

        err_eig = self.symmetrizer.check_eig(self.eig)
        err_amn = self.symmetrizer.check_amn(self.amn)
        if not silent:
            print(f"eig symmetry error : {err_eig}")
            print(f"amn symmetry error : {err_amn}")
        return err_eig, err_amn

    def set_random_symmetric_projections(self):
        """
        Set random symmetric projections for the system,
        according to the symmetrizer object
        """
        self.set_file("amn", self.symmetrizer.get_random_amn(), overwrite=True)



    def calc_WF_real_space(self,
                           sc_min=-1, sc_max=1,
                           select_WF=None,
                           reduce_r_points=1,
                           make_close_to_real_real=True):
        """
        calculate Wanier functions on a real-space grid

        Parameters
        ----------
        sc_min, sc_max : int or array-like
            the minimum and maximum supercell indices in the real space (sc_min is typically negative)
            if sc_max+1-sc_min exceeds the mp_grid, the grid is truncated
        select_WF : list(int)
            the list of Wannier functions to be calculated
        reduce_r_points : int or array-like
            the factor by which the grid is reduced in each direction (the grid shopuld be divisible by this factor)
        make_close_to_real_real : bool
            if True, apply to each Wannier function a phase such that the value at the maximum density is real

        Returns
        -------
        sc_origin : array((3,))
            the origin of the supercell in the real space
        sc_basis : array((3,3))
            the basis of the supercell in the real space
        WF : array((nWF, nr0, nr1, nr2, nspinor))
            the Wannier functions on the real space grid
        rho : array((nWF,nr0, nr1, nr2))
            the density of the Wannier functions on the real space grid (rho = |WF|^2fl)

        Note
        ----

        the norm is such that sum_r |WF|^2 = 1
        """
        self.check_wannierised("cannot calculate Wannier functions in the real space ")
        assert self.has_file("unk"), "UNK files are not set"

        def to_3array(x: int):
            if isinstance(x, int):
                return np.array([x] * 3)
            else:
                return np.array(x)

        sc_min_vec = to_3array(sc_min)
        sc_max_vec = to_3array(sc_max + 1)
        mp_grid = np.array(self.chk.mp_grid)
        sc_min_vec = np.maximum(sc_min_vec, -mp_grid // 2)
        sc_max_vec = np.minimum(sc_max_vec, (mp_grid + 1) // 2)
        sc_size_vec = sc_max_vec - sc_min_vec

        reduce_r_points = to_3array(reduce_r_points)
        print(f"self.unk.grid_size={self.unk.grid_size}")
        print(f"reduc_r_points = {reduce_r_points}")
        real_grid = np.array(self.unk.grid_size)
        assert np.all(real_grid % reduce_r_points == 0), f"cannot reduce grid {real_grid} by factors {reduce_r_points} - not divisible"
        real_grid = real_grid // reduce_r_points
        nr0, nr1, nr2 = real_grid

        nr_tot = nr0 * nr1 * nr2

        if select_WF is None:
            select_WF = range(self.chk.num_wann)

        mp_grid = np.array(self.chk.mp_grid)

        kpoints = self.chk.kpt_latt
        kpoints_int = self.chk.kpt_latt_int

        exp_one = np.exp(2j * np.pi / (mp_grid * real_grid))

        exp_grid = [np.cumprod([exp_one[i]] * real_grid[i]) for i in range(3)]

        output_grid_size = real_grid * sc_size_vec

        nspinor = 2 if self.unk.spinor else 1
        real_lattice = self.chk.real_lattice

        WF = np.zeros(tuple(output_grid_size) + (nspinor, len(select_WF),), dtype=complex)
        sc_origin = sc_min_vec @ real_lattice
        sc_basis = sc_size_vec[:, None] * real_lattice

        for ik in range(self.chk.num_kpts):
            assert ik in self.unk.data, "plotWF from irreducible kpoints is not implemented yet"
            U = self.unk.data[ik].copy()
            U = U[:, ::reduce_r_points[0], ::reduce_r_points[1], ::reduce_r_points[2], :]
            U = cached_einsum("m...,mn->...n", U, self.chk.v_matrix[ik][:, select_WF])
            k_int = kpoints_int[ik]
            k_latt = kpoints[ik]
            exp_loc = [exp_grid[i]**k_int[i] for i in range(3)]
            U[:] *= exp_loc[0][:, None, None, None, None]
            U[:] *= exp_loc[1][None, :, None, None, None]
            U[:] *= exp_loc[2][None, None, :, None, None]
            for i0 in range(sc_size_vec[0]):
                for i1 in range(sc_size_vec[1]):
                    for i2 in range(sc_size_vec[2]):
                        iR = np.array([i0, i1, i2]) + sc_min_vec
                        phase = np.exp(2j * np.pi * np.dot(iR, k_latt))
                        WF[i0 * nr0:(i0 + 1) * nr0, i1 * nr1:(i1 + 1) * nr1, i2 * nr2:(i2 + 1) * nr2, :, :] += U * phase

        WF = WF.transpose((4, 0, 1, 2, 3)) / np.prod(mp_grid) / np.sqrt(nr_tot)

        if make_close_to_real_real and not self.unk.spinor:
            for i in range(WF.shape[0]):
                data = WF[i].copy()
                shape = data.shape
                data = data.reshape(-1)
                pos = np.argmax(abs(data))
                w = data[pos]
                data *= w.conj() / abs(w)
                imag_max = abs(data.imag).max()
                print(f"wannier function {select_WF[i]} : Im/Re ratio {imag_max / abs(w)} ({data[pos]})")
                WF[i] = data.reshape(shape)

        rho = np.sum((WF * WF.conj()).real, axis=4)

        return sc_origin, sc_basis, WF, rho

    def get_xsf(self, sc_origin=None, sc_basis=None, data=None, atoms_cart=None, atoms_names=None, conv_cell=None, ):
        """
        get the string for XSF file from the data

        sc_origin, sc_basis 
            see calc_WF_real_space()
        data : array((nr0, nr1, nr2))
            the data to be plotted (only one spinor component is acepted)
        atoms_cart : array((natoms, 3))
            the atomic positions in cartesian coordinates (if None, the atomic positions are taken from the WIN file)
        atoms_names : list(str)
            the atomic names (if None, the atomic names are taken from the WIN file)
        conv_cell : array((3,3))
            the conventional cell (if None, the conventional cell is not written)

        Returns
        -------
        str
            the string for the XSF file


        """
        A = self.chk.real_lattice
        if atoms_cart is None:
            if hasattr(self, "atomic_positions_frac"):
                atoms_cart = self.atomic_positions_frac @ A
            elif hasattr(self, "atomic_positions_cart"):
                atoms_cart = self.atomic_positions_cart
            elif "atoms_cart" in self.win:
                atoms_cart = self.win["atoms_cart"]
            elif "atoms_frac" in self.win:
                atoms_cart = self.win["atoms_frac"] @ A
            else:
                atoms_cart = []
        if atoms_names is None:
            if hasattr(self, "atomic_names"):
                atoms_names = self.atomic_names
            elif "atoms_names" in self.win:
                atoms_names = self.win["atoms_names"]
            else:
                atoms_names = ["X"] * len(atoms_cart)

        out = f"""  #
        # Produced by WannierBerri https://wannier-berri.org
        # On {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        #
CRYSTAL
PRIMVEC
{A[0, 0]: 20.10f} {A[0, 1]: 20.10f} {A[0, 2]: 20.10f}
{A[1, 0]: 20.10f} {A[1, 1]: 20.10f} {A[1, 2]: 20.10f}
{A[2, 0]: 20.10f} {A[2, 1]: 20.10f} {A[2, 2]: 20.10f}
"""
        if conv_cell is not None:
            out += f"""CONVVEC
{conv_cell[0, 0]: 20.10f} {conv_cell[0, 1]: 20.10f} {conv_cell[0, 2]: 20.10f}
{conv_cell[1, 0]: 20.10f} {conv_cell[1, 1]: 20.10f} {conv_cell[1, 2]: 20.10f}
{conv_cell[2, 0]: 20.10f} {conv_cell[2, 1]: 20.10f} {conv_cell[2, 2]: 20.10f}
        """
        out += f"""PRIMCOORD
{len(atoms_cart)} 1
"""
        for a, c in zip(atoms_names, atoms_cart):
            out += f"{a} {c[0]: 20.10f} {c[1]: 20.10f} {c[2]: 20.10f}\n"
        out += "\n\n\n"

        # in order to get the "non-periodic" grid for xcrysden
        shape = np.array(data.shape[:3])
        sc_basis_loc = sc_basis * ((shape - 1) / shape)[:, None]

        def get_data_block(ind, dat):
            out_loc = f"""BEGIN_DATAGRID_3D_wannier_function_{ind}
{dat.shape[0]} {dat.shape[1]} {dat.shape[2]}
{sc_origin[0]: 20.10f} {sc_origin[1]: 20.10f} {sc_origin[2]: 20.10f}
{sc_basis_loc[0, 0]: 20.10f} {sc_basis_loc[0, 1]: 20.10f} {sc_basis_loc[0, 2]: 20.10f}
{sc_basis_loc[1, 0]: 20.10f} {sc_basis_loc[1, 1]: 20.10f} {sc_basis_loc[1, 2]: 20.10f}
{sc_basis_loc[2, 0]: 20.10f} {sc_basis_loc[2, 1]: 20.10f} {sc_basis_loc[2, 2]: 20.10f}
"""
            dat = dat.reshape(-1, order='F')
            num_per_string = 6
            for i in range(0, len(dat), num_per_string):
                out_loc += " ".join(f"{x: 20.10f}" for x in dat[i:i + num_per_string]) + "\n"
            out_loc += f"END_DATAGRID_3D_wannier_function_{ind}\n"
            return out_loc


        if data is not None:
            out += "BEGIN_BLOCK_DATAGRID_3D\nwannier_functions\n"
            if data.ndim == 3:
                data = data[None, ...]
            for i, dat in enumerate(data):
                out += get_data_block(i, dat)
            out += "END_BLOCK_DATAGRID_3D\n"

        return out

    def plotWF(self,
               sc_min=-1, sc_max=1,
               select_WF=None,
               reduce_r_points=1,
               make_real=True,
               atoms_cart=None, atoms_names=None,
               path=None
               ):
        """
        plot the Wannier functions on the real space grid and save them in the XSF format
        the output files are named as `path` + `index`.xsf

        Parameters
        ----------
        sc_min, sc_max : int or array-like
            the minimum and maximum supercell indices in the real space (sc_min is typically negative)
            if sc_max+1-sc_min exceeds the mp_grid, the grid is truncated
        select_WF : list(int)
            the list of Wannier functions to be calculated
        reduce_r_points : int or array-like
            the factor by which the grid is reduced in each direction (the grid shopuld be divisible by this factor)
        make_real : bool
            if True, apply to each Wannier function a phase such that the value at the maximum density is real
        atoms_cart : array((natoms, 3))
            the atomic positions in cartesian coordinates (if None, the atomic positions are taken from the WIN file)
        atoms_names : list(str)
            the atomic names (if None, the atomic names are taken from the WIN file)
        path : str
            the path to save the files (the files are named as `path` + `index`.xsf)
            if None, the files are saved as `seedname`.WF`index`.xsf

        Returns
        -------
        sc_origin : array((3,))
            the origin of the supercell in the real space
        sc_basis : array((3,3))
            the basis of the supercell in the real space
        WF : array((nWF, nr0, nr1, nr2, nspinor))
            the Wannier functions on the real space grid
        rho : array((nWF,nr0, nr1, nr2))
            the density of the Wannier functions on the real space grid (rho = |WF|^2)
        """
        if path is None:
            path = f"{self.seedname}.WF"
        if select_WF is None:
            select_WF = range(self.chk.num_wann)

        sc_origin, sc_basis, WF, rho = self.calc_WF_real_space(sc_min=sc_min, sc_max=sc_max,
                                                               select_WF=select_WF,
                                                               reduce_r_points=reduce_r_points,
                                                               make_close_to_real_real=make_real)
        assert not self.unk.spinor, "plotting Wannier functions is not implemented for spinors"
        WF = WF[..., 0]  # take the only spinor component
        for i, j in enumerate(select_WF):
            filename = path + f"{j:04d}.xsf"
            xsf_str = self.get_xsf(sc_origin=sc_origin, sc_basis=sc_basis,
                                   atoms_cart=atoms_cart, atoms_names=atoms_names,
                                   data=WF[i].real)
            with open(filename, "w") as f:
                f.write(xsf_str)

        return sc_origin, sc_basis, WF, rho
