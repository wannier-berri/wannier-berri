from functools import cached_property, lru_cache
import numpy as np
import sympy
from .unique_list import UniqueListMod1, all_close_mod1



class WyckoffPosition:
    """
    Wyckoff position defined by a sympy string

    Parameters
    ----------
    wyckoff_position_str : str
        The string defining the Wyckoff position. E.g. "x,y,z", "x,y,x-y", "1/2,1/2,z"
    spacegroup : irrep.spacegroup.SpaceGroup
        The spacegroup which transforms the coordinates

    Attributes
    ----------
    string : str
        The string defining the Wyckoff position.
    sympy : sympy.Expr
        The sympy expression defining the Wyckoff position.

    """

    xdef = 1 / np.e
    ydef = 1 / np.pi
    zdef = 1 / np.sqrt(2)

    def __init__(self, position_str, spacegroup, free_var_values=None):
        self.string = position_str
        self.spacegroup = spacegroup
        self.sympy = sympy.sympify(self.string)
        self.free_vars = set.union(*[s.free_symbols for s in self.sympy])
        self.free_vars = list(self.free_vars)
        self.free_vars.sort(key=lambda x: x.name)
        self.num_free_vars = len(self.free_vars)
        self.free_var_values = free_var_values
        self.wyckoff_position_lambda = sympy.lambdify(self.free_vars, self.sympy)
        orbit, self.rotations, self.translations = orbit_and_rottrans(spacegroup, self.position_numeric())
        self.num_points = orbit.shape[0]

    def vars_dict_to_array(self, d):
        return np.array([d[v] for v in self.free_vars], dtype=float)

    def __eq__(self, other):
        assert isinstance(other, WyckoffPosition), "other has to be a WyckoffPosition"
        if self.num_free_vars != other.num_free_vars:
            return False
        if self.num_free_vars == 0 and other.num_free_vars == 0:
            for p1 in self.positions:
                for p2 in other.positions:
                    if all_close_mod1(p1, p2):
                        return True
        return self.string == other.string


    def position_numeric(self, x=xdef, y=ydef, z=zdef):
        return np.array([s.subs({"x": x, "y": y, "z": z}) for s in self.sympy], dtype=float)

    @cached_property
    def map_on_free_vars(self):
        """
        Returns the matrices R and T, such that given a vector V = [x,y,..] of free variables
        the point Wyckoff position is given by R@V + T

        Returns
        -------
        np.ndarray(shape=(3,self.num_free_vars), dtype=float)
            The rotation matrix.
        np.ndarray(shape=(3,), dtype=float)
            The translation vector. (values when all free variables are 0)
        """
        rot = []
        trans = []
        for i in range(3):
            s = self.sympy[i].as_coefficients_dict()
            rot.append([s[v] for v in self.free_vars])
            trans.append(s[1])
        return np.array(rot, dtype=float), np.array(trans, dtype=float)

    @cached_property
    def map_orbit_on_free_vars(self):
        """
        Returns the matrices R and T, such that given a vector V = [x,y,..] of free variables
        all points of the Wyckoff position is given by R@V + T

        Returns
        -------
        np.ndarray(shape=(self.num_points, 3,self.num_free_vars), dtype=float)
            The rotation matrix.
        np.ndarray(shape=(self.num_points, 3), dtype=float)
            The translation vectors (values when all free variables are 0)
        """
        rot, trans = self.map_on_free_vars
        return self.rotations.dot(rot), self.rotations.dot(trans) + self.translations

    @property
    def positions(self):
        rot, trans = self.map_orbit_on_free_vars
        return rot.dot(self.free_var_values) + trans

    # @cached_property
    # def numeric_lambda(self):
    #     return sympy.lambdify(self.free_vars, self.wyckoff_position_sympy)

    # @cached_property
    # def orbit_lambda(self):
    #     return lambda *args: self.rotations.dot(self.wyckoff_position_lambda(*args)) + self.translations

    @cached_property
    def rotations_cart(self):
        lattice_T = self.spacegroup.lattice.T
        lattice_inv_T = np.linalg.inv(lattice_T)
        return [lattice_T @ R @ lattice_inv_T for R in self.rotations]

    @cached_property
    def free_vars(self):
        free_vars = set.union(*[s.free_symbols for s in self.wyckoff_position_sympy])
        free_vars = list(free_vars)
        free_vars.sort(key=lambda x: x.name)
        return free_vars

    @property
    def free_var_values(self):
        if not hasattr(self, "_free_var_values"):
            self._free_var_values = np.random.rand(self.num_free_vars)
        return self._free_var_values

    @free_var_values.setter
    def free_var_values(self, values):
        if values is None:
            if hasattr(self, "_free_var_values"):
                del self._free_var_values
        else:
            assert len(values) == self.num_free_vars, f"Values has to have length {self.num_free_vars}"
            self._free_var_values = values

    @cached_property
    def num_free_vars(self):
        return len(self.free_vars)


    def orbit_str(self):
        string = ""
        for rot, trans in zip(self.rotations, self.translations):
            pos = (np.dot(rot, self.sympy) + trans)
            if self.num_free_vars == 0:
                pos = pos % 1
            string += ", ".join(f"{x}" for x in pos) + "\n"
        return string


    def contains_position(self, position):
        if self.num_free_vars == 0:
            if all_close_mod1(position, self.positions[0]):
                return []
            else:
                return None
        position = np.array(position)
        rot, trans = self.map_orbit_on_free_vars
        for r, t in zip(rot, trans):
            sol = find_solution_mod1(r, position - t)
            if sol is not None:
                return sol
        return None

    def __str__(self):
        # var = np.random.rand(self.num_free_vars)
        return self.orbit_str()
        # s = ("\n" +
        #      "\n".join(str(l) for l in orbit) +
        #      "\n"
        #     )
        # return s

    def stick_to_atoms(self, atoms, atoms_filled):
        """
        ccheck if any of the atoms belong to this wyckoff position
        and find free variables that correspond to this atom (modulo 1)

        Parameters
        ----------
        atoms : list of np.ndarray(shape=(3,num_atoms), dtype=float)
            List of atomic positions.
        atoms_filled : list of bool
            List of flags for filled atoms. (The list is modified, the filled atoms are marked as True.)

        Returns
        -------
        np.ndarray(shape=(num_free_vars,), dtype=float)
            The free variables corresponding to the atom.
        """
        # print ("looking to stick wyckoff position {self.string} to atoms {atoms} (filled {atoms_filled})")
        for iat, atom in enumerate(atoms):
            orbit_atom = get_orbit(self.spacegroup, atom)
            # print ("orbit_atom",orbit_atom, orbit_atom.shape, self.num_points)
            if not atoms_filled[iat]:
                # we do not want to stick a more general wyckoff position to a more specific one
                if orbit_atom.shape[0] == self.num_points:
                    # print ("looking for solutions")
                    sol = self.contains_position(atom)
                    if sol is not None:
                        atoms_filled[iat] = True
                        orbit = self.orbit_lambda(*sol)
                        for iat1, at1 in enumerate(atoms):
                            if not atoms_filled[iat1]:
                                for orb in orbit:
                                    if all_close_mod1(at1, orb):
                                        atoms_filled[iat1] = True
                                        break
                        return sol
        return None

    def as_numeric(self):
        return WyckoffPositionNumeric(self.positions % 1, self.spacegroup)


class WyckoffPositionNumeric(WyckoffPosition):

    def __init__(self, positions, spacegroup):
        """
        Wyckoff position defined by a list of positions
        Parameters
        ----------
        positions : list of np.ndarray(shape=(3,), dtype=float) or np.ndarray(shape=(3), dtype=float)
            List of positions or only first position
        spacegroup : irrep.spacegroup.SpaceGroup
            The spacegroup which transforms the coordinates

        Note
        ----
        positions shouls transform into each other under the symmetry operations of the spacegroup
        (multiple orbits are not allowed)
        """

        positions = np.array(positions)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)
        self.spacegroup = spacegroup
        self.string = ", ".join(f"{x}" for x in positions[0])
        orbit0, rotations, translations = orbit_and_rottrans(spacegroup, positions[0])
        self.rotations = []
        self.translations = []
        orbit0 = UniqueListMod1(orbit0)
        orbit = UniqueListMod1()

        def add_pos_and_rottrans(pos):
            ind = orbit0.index(pos)
            l = len(orbit)
            orbit.append(pos)
            if len(orbit) > l:
                self.rotations.append(rotations[ind])
                self.translations.append(translations[ind])

        for pos in positions:
            assert pos in orbit0, f"Position {pos} is not in the orbit of the first position {positions[0]}"
            add_pos_and_rottrans(pos)
        # now add the positions that are not in the input
        for pos in orbit0:
            add_pos_and_rottrans(pos)
        assert len(orbit) == len(self.rotations), f"len(orbit) {len(orbit)} != len(rotations) {len(self.rotations)}"
        assert len(orbit) == len(self.translations), f"len(orbit) {len(orbit)} != len(translations) {len(self.translations)}"
        self._positions = np.array(orbit)
        self.num_points = self._positions.shape[0]

    @property
    def positions(self):
        return self._positions

    def __str__(self):
        var = np.random.rand(self.num_free_vars)
        orbit = self.orbit_lambda(*var)
        s = ("\n" +
             "\n".join(str(l) for l in orbit) +
             "\n"
            )
        return s

    def orbit_str(self):
        string = ""
        for pos in self._positions:
            string += ", ".join(f"{x}" for x in pos) + "\n"
        return string

    @property
    def num_free_vars(self):
        return 0

    @cached_property
    def map_orbit_on_free_vars(self):
        return np.zeros((self.num_points, 3, 0)), self._positions

    def as_numeric(self):
        return self




def get_orbit(spacegroup, p, tol=1e-5):
    """
    Get the orbit of a point p under the symmetry operations of a structure.

    Parameters
    ----------
    spacegroup : irrep.spacegroup.SpaceGroup
        The spacegroup of the structure. If None, the orbit is just the point p.
    p : np.ndarray(shape=(3,), dtype=float)
        Point for which to calculate the orbit in the reduced coordinates.

    Returns
    -------
    UniqueListMod1 of np.ndarray(shape=(3,), dtype=float)
        The orbit of v under the symmetry operations of the structure.
    """
    return UniqueListMod1([symop.transform_r(p) % 1 for symop in spacegroup.symmetries], tol=tol)


def orbit_and_rottrans(spacegroup, p):
    """
    Get the orbit of a point p under the symmetry operations of a structure.
    and the corresponding rotation matrices and translation vectors.

    Parameters
    ----------
    spacegroup : irrep.spacegroup.SpaceGroup
        The spacegroup object.
    p : np.ndarray(shape=(3,), dtype=float)
        Point for which to calculate the orbit in the reduced coordinates.

    Returns
    -------
    np.ndarray(shape=(N, 3)
        The orbit of v under the symmetry operations of the structure.
    np.ndarray(shape=(N, 3, 3)
        The rotation matrices of the symmetry operations
    np.ndarray(shape=(N, 3)
        The translation vectors of the symmetry operations
    """
    orbit = get_orbit(spacegroup, p)
    ind_oper = orbit.appended_indices
    rotations = []
    translations = []
    for i in ind_oper:
        symop = spacegroup.symmetries[i]
        rotations.append(symop.rotation)
        translations.append(symop.translation)
    return np.array(orbit), np.array(rotations), np.array(translations)


def find_solution_mod1(A, B, max_shift=2):
    """
    Find a solution such that A@x = B mod 1

    Parameters
    ----------
    A : np.ndarray (n,m)
        The matrix of the system.   
    B : np.ndarray (n,)
        The right hand side.
    max_shift : int
        The maximum shift.

    Returns
    -------
    list of np.ndarray
        The shifts that are compatible with the system.
    """
    A = np.array(A)
    B = np.array(B)
    r1 = np.linalg.matrix_rank(A)
    assert (r1 == A.shape[1]), f"overdetermined system {r1} != {A.shape}[1]"
    dim = A.shape[0]
    for shift in get_shifts(max_shift, ndim=dim):
        B_loc = B + shift
        if np.linalg.matrix_rank(np.hstack([A, B_loc[:, None]])) == r1:
            x, residuals, rank, s = np.linalg.lstsq(A, B_loc, rcond=None)
            if len(residuals) > 0:
                assert np.max(np.abs(residuals)) < 1e-7
            assert rank == r1
            return x
    return None


@lru_cache
def get_shifts(max_shift, ndim=3):
    """return all possible shifts of a 3-component vector with integer components
    recursively by number of dimensions

    Parameters
    ----------
    max_shift : int
        The maximum absolute value of the shift.
    ndim : int
        The number of dimensions.

    Returns
    -------
    array_like(n, ndim)
        The shifts. n=(max_shift*2+1)**ndim
        sorted by the norm of the shift
    """
    if ndim == 1:
        shifts = np.arange(-max_shift, max_shift + 1)[:, None]
    else:
        shift_1 = get_shifts(max_shift, ndim - 1)
        shift1 = get_shifts(max_shift, 1)
        shifts = np.vstack([np.hstack([shift_1, [s1] * shift_1.shape[0]]) for s1 in shift1])
    # more probably that equality happens at smaller shift, so sort by norm
    srt = np.linalg.norm(shifts, axis=1).argsort()
    return shifts[srt]


def split_into_orbits(positions, spacegroup):
    """
    Split a list of positions into orbits under the symmetry operations of a structure.

    Parameters
    ----------
    positions : list of np.ndarray(shape=(3,), dtype=float)
        The positions to split into orbits.
    spacegroup : irrep.spacegroup.SpaceGroup
        The spacegroup of the structure.

    Returns
    -------
    list of list of int
        the indices of the positions in the input list that belong to the same orbit.
    """
    orbits = []
    orbits_ind = []
    for ip, pos in enumerate(positions):
        for io, orb in enumerate(orbits):
            if pos in orb:
                orbits_ind[io].append(ip)
                break
        else:
            orb = get_orbit(spacegroup, pos)
            orbits.append(orb)
            orbits_ind.append([ip])
    return orbits_ind
