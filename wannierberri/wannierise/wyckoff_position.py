from functools import cached_property
import numpy as np
import sympy
from ..__utility import UniqueListMod1, all_close_mod1
from .utility import find_solution_mod1



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
        positions = np.array(positions)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)
        self.spacegroup = spacegroup
        self.string = ", ".join(f"{x}" for x in positions[0])
        positions = np.array(positions)
        orbit0 = get_orbit(spacegroup, positions[0])
        orbit = UniqueListMod1()
        for pos in positions:
            assert pos in orbit0, f"Position {pos} is not in the orbit of the first position {positions[0]}"
            orbit.append(pos)
        for pos in orbit0:
            orbit.append(pos)
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
