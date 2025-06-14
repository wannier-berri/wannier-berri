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
from ..symmetry.point_symmetry import PointGroup
from ..utility import real_recip_lattice


def num_cart_dim(key):
    """
    returns the number of cartesian dimensions of a matrix by key
    """
    if key in ["Ham"]:
        return 0
    elif key in ["AA", "BB", "CC", "SS", "SH", "OO"]:
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
                 ):

        # TODO: move some initialization to child classes
        self.frozen_max = frozen_max
        self.periodic = periodic
        self.name = name
        self.silent = silent


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


    def set_pointgroup(self, symmetry_gen=(), pointgroup=None, spacegroup=None):
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
            self.pointgroup = pointgroup
        elif spacegroup is not None:
            assert np.allclose(spacegroup.Lattice, self.real_lattice)
            self.pointgroup = PointGroup(spacegroup=spacegroup, recip_lattice=self.recip_lattice, real_lattice=self.real_lattice)
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
        np.ndarray(num_wann) = [0,1,2,...,num_wann-1]
            The range of the Wannier functions in the Cartesian coordinates
        """
        return np.arange(self.num_wann)



class System_k(System):
    """
    The base class for describing a system in the k-space. Does not have its own constructor,
    please use the child classes, e.g  :class:`SystemKP`.
    """
    pass
