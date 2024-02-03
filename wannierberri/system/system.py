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

import numpy as np
import lazy_property
from ..symmetry import Group
import multiprocessing

pauli_x = [[0, 1], [1, 0]]
pauli_y = [[0, -1j], [1j, 0]]
pauli_z = [[1, 0], [0, -1]]
pauli_xyz = np.array([pauli_x, pauli_y, pauli_z]).transpose((1, 2, 0))


class System:
    default_parameters = {
        'seedname': 'wannier90',
        'frozen_max': -np.Inf,
        'berry': False,
        'morb': False,
        'spin': False,
        'SHCryoo': False,
        'SHCqiao': False,
        'use_ws': True,
        'mp_grid': None,
        'periodic': (True, True, True),
        'use_wcc_phase': False,
        'wannier_centers_cart': None,
        'wannier_centers_reduced': None,
        'npar': None,
        '_getFF': False,
    }

    __doc__ = """
    The base class for describing a system. Does not have its own constructor,
    please use the child classes, e.g  :class:`System_w90` or :class:`System_tb`


    Parameters
    -----------
    seedname : str
        the seedname used in Wannier90. Default: ``{seedname}``
    berry : bool
        set ``True`` to enable evaluation of external term in  Berry connection or Berry curvature and their
        derivatives. Default: ``{berry}``
    spin : bool
        set ``True`` if quantities derived from spin  will be used. Default:``{spin}``
    morb : bool
        set ``True`` to enable calculation of external terms in orbital moment and its derivatives.
        Requires the ``.uHu`` file. Default: ``{morb}``
    periodic : [bool,bool,bool]
        set ``True`` for periodic directions and ``False`` for confined (e.g. slab direction for 2D systems). If less then 3 values provided, the rest are treated as ``False`` . Default : ``{periodic}``
    SHCryoo : bool
        set ``True`` if quantities derived from Ryoo's spin-current elements will be used. (RPS 2019) Default: ``{SHCryoo}``
    SHCqiao : bool
        set ``True`` if quantities derived from Qiao's approximated spin-current elements will be used. (QZYZ 2018). Default: ``{SHCqiao}``
    use_ws : bool
        minimal distance replica selection method :ref:`sec-replica`.  equivalent of ``use_ws_distance`` in Wannier90. Default: ``{use_ws}``
    mp_grid : [nk1,nk2,nk3]
        size of Monkhorst-Pack frid used in ab initio calculation. Needed when `use_ws=True`, and only if it cannot be read from input file, i.e.
        like :class:`.System_tb`, :class:`.System_PythTB`, :class:`.System_TBmodels` ,:class:`.System_fplo`, but only if
        the data originate from ab initio data, not from toy models.
        In contrast, for :class:`.System_w90` and :class:`.System_ase` it is not needed,  but can be provided and will override the original value
        (if you know what and why you are doing)
        Default: ``{mp_grid}``
    frozen_max : float
        position of the upper edge of the frozen window. Used in the evaluation of orbital moment. But not necessary. Default: ``{frozen_max}``
    _getFF : bool
        generate the FF_R matrix based on the uIu file. May be used for only testing so far. Default : ``{_getFF}``
    use_wcc_phase: bool
        using wannier centers in Fourier transform. Correspoinding to Convention I (True), II (False) in Ref."Tight-binding formalism in the context of the PythTB package". Default: ``{use_wcc_phase}``
    wannier_centers_cart :  array-like(num_wann,3)
        use the given wannier_centers (cartesian) instead of those determined automatically. Incompatible with `wannier_centers_reduced`
    wannier_centers_reduced :  array-like(num_wann,3)
        use the given wannier_centers (reduced) instead of those determined automatically. Incompatible with `wannier_centers_cart`
    npar : int
        number of nodes used for parallelization in the `__init__` method. Default: `multiprocessing.cpu_count()`

    Notes
    -----
    + for tight-binding models it is recommended to use `use_wcc_phase = True`. In this case the external terms vanish, and
    + one can safely use `berry=False, morb=False`, and also set `'external_terms':False` in the parameters of the calculation

    """.format(**default_parameters)

    def set_parameters(self, **parameters):

        for param in self.default_parameters:
            if param in parameters:
                vars(self)[param] = parameters[param]
            else:
                vars(self)[param] = self.default_parameters[param]

        for param in parameters:
            if param not in self.default_parameters:
                print(f"WARNING: parameter {param} was passed to data_K, which is not recognised")

        if self.npar is None:
            self.npar = multiprocessing.cpu_count()
        if self.mp_grid is not None:
            self.mp_grip = np.array(self.mp_grid)

        periodic = np.zeros(3, dtype=bool)
        periodic[:len(self.periodic)] = self.periodic
        self.periodic = periodic
        self.needed_R_matrices = set(['Ham'])
        if self.morb:
            self.needed_R_matrices.update(['AA', 'BB', 'CC'])
        if self.berry:
            self.needed_R_matrices.add('AA')
        if self.spin:
            self.needed_R_matrices.add('SS')
        if self._getFF:
            self.needed_R_matrices.add('FF')
        if self.SHCryoo:
            self.needed_R_matrices.update(['AA', 'SS', 'SA', 'SHA', 'SR', 'SH', 'SHR'])
        if self.SHCqiao:
            self.needed_R_matrices.update(['AA', 'SS', 'SR', 'SH', 'SHR'])

        self._XX_R = dict()

        if self.wannier_centers_cart is not None:
            self.wannier_centers_cart = np.array(self.wannier_centers_cart)
            if self.wannier_centers_reduced is not None:
                raise ValueError(
                    "one should not specify both wannier_centers_cart and wannier_centers_reduced,"
                    "or, set_wannier_centers should not be called twice")
            else:
                self.wannier_centers_reduced = self.wannier_centers_cart.dot(np.linalg.inv(self.real_lattice))
        elif self.wannier_centers_reduced is not None:
            self.wannier_centers_reduced = np.array(self.wannier_centers_reduced)
            self.wannier_centers_cart = self.wannier_centers_reduced.dot(self.real_lattice)
        if self.wannier_centers_cart is not None:
            self.num_wann = self.wannier_centers_cart.shape[0]

    def set_symmetry(self, symmetry_gen=[]):
        """
        Set the symmetry group of the :class:`System`

        Parameters
        ----------
        symmetry_gen : list of :class:`symmetry.Symmetry` or str
            The generators of the symmetry group.

        Notes
        -----
        + Only the generators of the symmetry group are essential. However, no problem if more symmetries are provided.
          The code further evaluates all possible products of symmetry operations, until the full group is restored.
        + Providing `Identity` is not needed. It is included by default
        + Operations are given as objects of :class:`symmetry.Symmetry` or by name as `str`, e.g. ``'Inversion'`` , ``'C6z'``, or products like ``'TimeReversal*C2x'``.
        + ``symetyry_gen=[]`` is equivalent to not calling this function at all
        + Only the **point group** operations are important. Hence, for non-symmorphic operations, only the rotational part should be given, neglecting the translation.

        """
        self.symgroup = Group(symmetry_gen, recip_lattice=self.recip_lattice, real_lattice=self.real_lattice)

    @lazy_property.LazyProperty
    def cell_volume(self):
        return abs(np.linalg.det(self.real_lattice))



class System_k(System):

    pass
