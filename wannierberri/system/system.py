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
#                                                            #
# ------------------------------------------------------------

import os
import sys
import warnings
import numpy as np
from functools import cached_property

from ..utility import real_recip_lattice
from ..symmetry.point_symmetry import PointSymmetry, PointGroup, TimeReversal


def num_cart_dim(key):
    """
    returns the number of cartesian dimensions of a matrix by key
    """
    if key in ["Ham", "overlap_up_down"]:
        return 0
    elif key in ["AA", "BB", "CC", "SS", "SH", "OO", "dV_soc_wann_0_0", "dV_soc_wann_0_1", "dV_soc_wann_1_1"]:
        return 1
    elif key in ["SHA", "SA", "SR", "SHR", "GG", "FF"]:
        return 2
    else:
        raise ValueError(f"unknown matrix {key}")


class System:

    """
    The base class for describing a system. Does not have its own constructor,
    please use the child classes, e.g  :class:`System_w90` or :class:`System_tb`


    Parameters
    -----------
    periodic : [bool,bool,bool]
        set ``True`` for periodic directions and ``False`` for confined (e.g. slab direction for 2D systems). If less then 3 values provided, the rest are treated as ``False`` .
    frozen_max : float
        position of the upper edge of the frozen window. Used in the evaluation of orbital moment. But not necessary.
    NKFFT :
        the FFT grid which further will be used in calculations by default
    force_internal_terms_only : bool
        only internal terms will be evaluated in all formulae, the external or cross terms will be excluded.
        the internal terms are defined only by the Hamiltonian and spin
    name : str
        name that will be used by default in names of output files
    silent : bool
        if ``True``, the code will not print any information about the system
    spinor : bool or None
        if ``True``, the system is spifull, if ``False``, the system is spinless. 
        ''Non'' if it is unknow

    Notes
    -----
    + The system is described by its real lattice, symmetry group, and Hamiltonian.
    + The lattice is given by the lattice vectors in the Cartesian coordinates.
    + The symmetry group is given by the generators of the group. The group is further evaluated by the code.
    + The Hamiltonian is given by the hopping terms between the orbitals. The hopping terms are given in the real space. 
    + The system can be either periodic or confined in some directions..
    """

    def __init__(self,
                 frozen_max=-np.inf,
                 periodic=(True, True, True),
                 NKFFT=None,
                 force_internal_terms_only=False,
                 name='wberri',
                 silent=False,
                 spinor=None,
                 ):

        # TODO: move some initialization to child classes
        self.frozen_max = frozen_max
        self.periodic = periodic
        self.name = name
        self.silent = silent
        self.spinor = spinor


        if NKFFT is not None:
            self._NKFFT_recommended = NKFFT

        self.periodic = np.zeros(3, dtype=bool)
        self.periodic[:len(self.periodic)] = periodic
        self.is_phonon = False
        self.force_internal_terms_only = force_internal_terms_only

    @property
    def logfile(self):
        if self.silent:
            return open(os.devnull, 'w')
        else:
            return sys.stdout



    def set_real_lattice(self, real_lattice=None, recip_lattice=None):
        """
        Set the real lattice of the :class:`System`

        Parameters
        ----------
        real_lattice : np.ndarray(3,3)
            The real lattice vectors in the Cartesian coordinates. If not provided, the code will evaluate it by the reciprocal lattice.
        recip_lattice : np.ndarray(3,3)
            The reciprocal lattice vectors in the Cartesian coordinates. If not provided, the code will evaluate it by the real lattice.

        Notes
        -----
        + Only one of the parameters should be provided. The other will be evaluated by the code.
        """
        assert not hasattr(self, 'real_lattice')
        self.real_lattice, _ = real_recip_lattice(real_lattice=real_lattice, recip_lattice=recip_lattice)



    @cached_property
    def recip_lattice(self):
        """	
        Returns
        -------
        np.ndarray(3,3)
            The reciprocal lattice vectors in the Cartesian coordinates
        """
        _, recip = real_recip_lattice(real_lattice=self.real_lattice)
        return recip

    def set_symmetry(self, symmetry_gen=(), pointgroup=None, spacegroup=None):
        """
        a wrapper for the :meth:`set_pointgroup` method, to keep backward compatibility. 
        Will be removed in the future.
        """
        # deprecation warning
        warnings.warn("The method `set_symmetry` is deprecated. Use `set_pointgroup` instead", DeprecationWarning)
        self.set_pointgroup(symmetry_gen=symmetry_gen, pointgroup=pointgroup, spacegroup=spacegroup)


    def set_pointgroup(self, symmetry_gen=(), pointgroup=None, spacegroup=None, use_symmetries_index=None):
        """
        Set the symmetry group of the :class:`System`, which will be used for symmetrization
        in k-space and for reducing the number of k-points in the BZ.

        Parameters
        ----------
        symmetry_gen : list of :class:`symmetry.Symmetry` or str
            The generators of the symmetry group.
        spacegroup : :class:`irrep.spacegroup.SpaceGroup`
            The space group of the system. The point group will be evaluated by the space group.
        pointgroup : :class:`~wannieberri.point_symmetry.PointGroup`
            The point group of the system. If provided, the code will use it as the symmetry group.

        Notes
        -----
        + Only the generators of the symmetry group are essential. However, no problem if more symmetries are provided.
          The code further evaluates all possible products of symmetry operations, until the full group is restored.
        + Providing `Identity` is not needed. It is included by default
        + Operations are given as objects of :class:`symmetry.Symmetry` or by name as `str`, e.g. ``'Inversion'`` , ``'C6z'``, or products like ``'TimeReversal*C2x'``.
        + ``symetyry_gen=[]`` is equivalent to not calling this function at all
        + Only the **point group** operations are important. Hence, for non-symmorphic operations, only the rotational part should be given, neglecting the translation.
        """
        if pointgroup is not None:
            assert np.allclose(pointgroup.recip_lattice, self.recip_lattice), f"the provided pointgroup has a different recip_lattice:\n{pointgroup.recip_lattice}\n than the system:\n{self.recip_lattice}"
            self.pointgroup = pointgroup
        elif spacegroup is not None:
            self.pointgroup = PointGroup(spacegroup=spacegroup, recip_lattice=self.recip_lattice, real_lattice=self.real_lattice, use_symmetries_index=use_symmetries_index)
            self.set_structure_from_sg(spacegroup, ignore_no_atoms=True)
        else:
            self.pointgroup = PointGroup(symmetry_gen, recip_lattice=self.recip_lattice, real_lattice=self.real_lattice)


    @cached_property
    def cell_volume(self):
        """
        Returns
        -------
        float
            The volume of the unit cell in Angstrom^3
        """
        return abs(np.linalg.det(self.real_lattice))

    @cached_property
    def range_wann(self):
        """
        Returns
        -------
        path object
            The range of Wannier functions, i.e. the list of Wannier functions. By default, it is set to all Wannier functions, but it can be changed by the user.
        """
        return np.arange(self.num_wann)

    def set_structure(self, positions, atom_labels, magnetic_moments=None):
        """
        Set atomic structure of the system.

        Parameters
        ----------
        positions : (num_atom, 3) array_like of float
            Atomic positions in fractional coordinates.
        atom_labels: (num_atom,) list
            labels (integer, string, etc.) to distinguish species.
        magnetic_moments: (num_atom, 3) array_like of float (optional)
            Magnetic moment vector of each atom.
        """
        if len(positions) != len(atom_labels):
            raise ValueError("length of positions and atom_labels must be the same")
        if magnetic_moments is not None:
            if len(magnetic_moments) != len(positions):
                raise ValueError("length of positions and magnetic_moments must be the same")
            if not all([len(x) == 3 for x in magnetic_moments]):
                raise ValueError("magnetic_moments must be a list of 3d vector")
        self.positions = positions
        self.atom_labels = atom_labels
        self.magnetic_moments = magnetic_moments

    def set_structure_from_sg(self, spacegroup, ignore_no_atoms=False):
        """
        Set the atomic structure from the space group. This method is useful when the space group is provided, but the atomic structure is not provided.

        Parameters
        ----------
        spacegroup : :class:`irrep.spacegroup.SpaceGroup`
            The space group of the system. The atomic structure will be extracted from the space group.

        """
        assert np.allclose(spacegroup.Lattice, self.real_lattice), f"the provided spacegroup has a different real_lattice:\n{spacegroup.Lattice}\n than the system:\n{self.real_lattice}"
        if hasattr(spacegroup, 'positions') and hasattr(spacegroup, 'typat'):
            self.positions = spacegroup.positions
            self.atom_labels = spacegroup.typat
        else:
            if ignore_no_atoms:
                self.positions = [[0, 0, 0]]
                self.atom_labels = [0]
            else:
                raise AttributeError("spacegroup does not have positions or typat attributes")

    def get_spglib_cell(self, ignore_no_atoms=False):
        """Returns the atomic structure as a cell tuple in spglib format"""
        # assign integer to self.atom_labels
        has_atoms = hasattr(self, 'atom_labels') and hasattr(self, 'positions') and len(self.atom_labels) > 0 and len(self.positions) > 0
        if not has_atoms:
            if ignore_no_atoms:
                Warning("atom_labels or positions are not set, using single atom in the unit cell")
                atom_labels = [0]
                positions = [[0, 0, 0]]
            else:
                raise AttributeError("atom_labels or positions are not set, cannot get spglib cell\n set_structure must be called before get_spglib_cell")
        else:
            atom_labels = self.atom_labels
            positions = self.positions
        atom_labels_unique = list(set(atom_labels))
        atom_numbers = [atom_labels_unique.index(label) for label in atom_labels]
        if not hasattr(self, 'magnetic_moments') or self.magnetic_moments is None:
            return self.real_lattice, positions, atom_numbers
        else:
            return self.real_lattice, positions, atom_numbers, self.magnetic_moments

    def set_symmetry_from_structure(self):
        """
        a wrapper for set_pointgroup_from_structure
        This method is deprecated and will be removed in future versions.
        """
        warnings.warn(
            "set_symmetry_from_structure is deprecated. Use set_pointgroup_from_structure instead.",
            DeprecationWarning
        )
        self.set_pointgroup_from_structure()

    def set_pointgroup_from_structure(self):
        """
        Set the symmetry group of the :class:`System`. Requires spglib to be installed.
        :meth:`System.set_structure` must be called in advance.

        For magnetic systems, symmetries involving time reversal are not detected because
        spglib does not support time reversal symmetry for noncollinear systems.
        """
        import spglib

        spglib_symmetry = spglib.get_symmetry(self.get_spglib_cell())
        symmetry_gen = []
        for isym, W in enumerate(spglib_symmetry["rotations"]):
            # spglib gives real-space rotations in reduced coordinates. Here,
            # 1) convert to Cartesian coordinates, and
            # 2) take transpose to go to reciprocal space.
            W = spglib_symmetry["rotations"][isym]
            Wcart = self.real_lattice.T @ W @ np.linalg.inv(self.real_lattice).T
            R = Wcart.T
            try:
                TR = spglib_symmetry['time_reversals'][isym]
                tr_found = True
            except KeyError:
                TR = False
                tr_found = False
            symmetry_gen.append(PointSymmetry(R, TR=TR))

        if self.magnetic_moments is None:
            symmetry_gen.append(TimeReversal)
        elif not tr_found:
            warnings.warn(
                "you specified magnetic moments but spglib did not detect symmetries involving time-reversal. "
                f"proobably it is because you have an old spglib version {spglib.__version__}."
                "We suggest upgrading to spglib>=2.0.2")
        else:
            if not all([len(x) for x in self.magnetic_moments]):
                raise ValueError("magnetic_moments must be a list of 3d vector")
            warnings.warn("spglib does not find symmetries including time reversal operation. "
                          "To include such symmetries, use set_symmetry.")


        self.pointgroup = PointGroup(symmetry_gen, recip_lattice=self.recip_lattice, real_lattice=self.real_lattice)

    def get_bandstructure(self, dk=0.05, parallel=True, return_path=True):
        from ..grid.path import Path
        from ..evaluate_k import evaluate_k_path
        cell = self.get_spglib_cell(ignore_no_atoms=True)
        if self.periodic.sum() == 0:
            path = Path.from_nodes(cell, nodes=[[0, 0, 0]], labels=['G'], dk=dk)
        elif self.periodic.sum() == 1:
            directionk = np.where(self.periodic)[0][0]
            X = np.zeros(3)
            X[directionk] = 0.5
            mX = np.zeros(3)
            mX[directionk] = -0.5
            path = Path.from_nodes(cell, nodes=[mX, [0, 0, 0], X], labels=['-X', 'G', 'X'], dk=dk)
        else:
            if self.periodic.sum() == 2:
                direction = np.where(~self.periodic)[0][0]
            else:
                direction = None
            path = Path.seekpath(cell, dk=dk, twoD_direction=direction)
        bandstructure = evaluate_k_path(system=self,
                                     path=path,
                                     return_path=False,
                                     parallel=parallel,
        )
        if return_path:
            return path, bandstructure
        else:
            return bandstructure




class System_k(System):
    """
    The base class for describing a system in the k-space. Does not have its own constructor,
    please use the child classes, e.g  :class:`SystemKP`.
    """
    pass
