from .grid import GridAbstract
from .Kpoint import KpointBZpath
import warnings
from collections.abc import Iterable
import numpy as np
import seekpath
from .path_order import flatten_path


class Path(GridAbstract):
    """ A class containing information about the k-path

    Parameters
    -----------
    system : :class:`~wannierberri.system.System`
        which the calculations will be made
    length :  float
        (angstroms) -- in this case the grid is NK[i]=length*||B[i]||/2pi  B- reciprocal lattice
    dk :  float
        (inverse angstroms) -- in this case the grid is NK[i]=length*||B[i]||/2pi  B- reciprocal lattice
    nodes : list
        cordinates of the nodes in the the reduced coordinates. Some entries may be None - which means that the segment should be skipped
        | No labels or nk's should be assigned to None nodes
    nk : int  or list or numpy.array(3)
        number of k-points along each directions
    k_list : list or str
        | if k_list is a list  - Coordinatres of all k-points in the reduced coordinates
        | if k_list = 'sphere' - Automatically generate k-points on a sphere (request r1 origin ntheta nphi)
        | if k_list = 'spheroid' - Automatically generate k-points on a spheroid (request r1 r2 origin ntheta nphi)
    labels : list  of dict
        | if k_list is set - it is a dict {i:lab} with i - index of k-point, lab - corresponding label (not all kpoints need to be labeled
        | if nodes is set - it is a list of labels, one for every node
    r1,r2 : float
        radius.
        sphere: x^2+y^2+z^2 = r1^2
        spheroid: (x^2+y^2)/r1^2+z^2/r2^2 = 1
    origin : array
        origin of sphere or spheroid in k-space
    nphi,ntheta: int
        number of k-points along each angle in polar coordinates
    Notes
    -----
    user needs to specify either `k_list` or (`nodes` + (`length` or `nk` or dk))

    """

    def __init__(self,
                 system,
                 k_list,
                 labels=None,
                 breaks=None):
        super().__init__(system=system, use_symmetry=False)
        if breaks is None:
            breaks = []
        if labels is None:
            labels = {}
        self.breaks = breaks
        self.labels = labels
        self.K_list = np.array(k_list)
        self.div = np.shape(self.K_list)[0]

    @classmethod
    def spheroid(cls, system, r1, r2, ntheta, nphi, origin=None):
        if origin is None:
            origin = np.array([0.0, 0.0, 0.0])
        theta = np.linspace(0, np.pi, ntheta, endpoint=True)
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=True)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        sphere = [
            r1 * np.cos(phi_grid) * np.sin(theta_grid), r1 * np.sin(phi_grid) * np.sin(theta_grid),
            r2 * np.cos(theta_grid)
        ]
        cart_k_list = np.array(sphere).reshape(3, ntheta * nphi).transpose(1, 0)
        k_list = cart_k_list.dot(np.linalg.inv(system.recip_lattice)) - origin[None, :]
        with open("klist_cart.txt", "w") as f:
            for i in range(ntheta * nphi):
                f.write(f"{cart_k_list[i, 0]:12.6f}{cart_k_list[i, 1]:12.6f}{cart_k_list[i, 2]:12.6f}\n")

        return cls(
            system=system,
            k_list=k_list,
            labels=['sphere']
        )

    @classmethod
    def sphere(cls, system, r1, ntheta, nphi, origin=None):
        return cls.spheroid(system, r1, r1, ntheta, nphi, origin)

    @classmethod
    def from_nodes(cls, system, nodes, labels=None, length=None, dk=None,
                   nk=None):
        if labels is None:
            labels = [str(i + 1) for i, k in enumerate([k for k in nodes if k is not None])]
        labels = (l for l in labels)
        labels = [None if k is None else next(labels) for k in nodes]
        new_labels = {}
        breaks = []

        if length is not None:
            assert length > 0
            if dk is not None:
                raise ValueError("'length' and  'dk' cannot be set together")
            dk = 2 * np.pi / length
        if dk is not None:
            if nk is not None:
                raise ValueError("'nk' cannot be set together with 'length' or 'dk' ")

        if isinstance(nk, Iterable):
            nkgen = (x for x in nk)
        else:
            nkgen = (nk for _ in nodes)
        K_list = np.zeros((0, 3))
        for start, end, l1, l2 in zip(nodes, nodes[1:], labels, labels[1:]):
            if start is not None and end is not None:
                new_labels[K_list.shape[0]] = l1
                start = np.array(start)
                end = np.array(end)
                assert start.shape == end.shape == (3, )
                if nk is not None:
                    _nk = next(nkgen)
                else:
                    if hasattr(system, "recip_lattice"):
                        rec_lattice = system.recip_lattice
                    else:  # assuming it is array-like
                        rec_lattice = 2 * np.pi * np.linalg.inv(system).T
                    _nk = round(np.linalg.norm((start - end).dot(rec_lattice)) / dk) + 1
                    if _nk == 1:
                        _nk = 2
                K_list = np.vstack(
                    (
                        K_list, start[None, :] + np.linspace(0, 1., _nk - 1, endpoint=False)[:, None] *
                        (end - start)[None, :]))
            elif end is None:
                breaks.append(K_list.shape[0] - 1)
        K_list = np.vstack((K_list, nodes[-1]))
        new_labels[K_list.shape[0] - 1] = labels[-1]
        return cls(
            system=system,
            k_list=K_list,
            labels=new_labels,
            breaks=breaks
        )

    @classmethod
    def seekpath(cls, cell=None, dk=0.05, with_time_reversal=True,
                 lattice=None, positions=None, numbers=None, twoD_direction=None):
        if cell is None:
            if lattice is None or positions is None or numbers is None:
                raise ValueError("Either 'cell' or ('lattice', 'positions', 'numbers') should be set")
            cell = (lattice, positions, numbers)
        path = seekpath.get_path_orig_cell(cell, with_time_reversal=with_time_reversal)
        point_coords = path['point_coords']
        path_seek = path['path']
        point_coords, path_seek = flatten_path(point_coords, path_seek, direction=twoD_direction)
        nodes = []
        labels = []
        last_point = None
        for segment in path_seek:
            if segment[0] == last_point:
                nodes.append(point_coords[segment[1]])
                labels.append(segment[1])
            else:
                nodes.extend([None, point_coords[segment[0]], point_coords[segment[1]]])
                labels.extend([segment[0], segment[1]])
            last_point = segment[1]
        return cls.from_nodes(cell[0], nodes=nodes, labels=labels, dk=dk)

    def get_refined(self, factor=2):
        """ returns a refined path with more k-points by linear interpolation"""
        K_list_refined = []
        labels_refined = {}
        breaks_refined = []
        last_point_index = len(self.K_list) - 1
        for i in range(last_point_index):
            K_list_refined.append(self.K_list[i])
            if i in self.labels:
                labels_refined[len(K_list_refined) - 1] = self.labels[i]
            if i in self.breaks:
                breaks_refined.append(len(K_list_refined) - 1)
            if i not in self.breaks:
                segment = (self.K_list[i + 1] - self.K_list[i]) / factor
                for j in range(1, factor):
                    K_list_refined.append(self.K_list[i] + j * segment)
        K_list_refined.append(self.K_list[-1])
        if last_point_index in self.labels:
            labels_refined[len(K_list_refined) - 1] = self.labels[last_point_index]
        if last_point_index in self.breaks:
            breaks_refined.append(len(K_list_refined) - 1)
        return Path(
            system=self.pointgroup,
            k_list=K_list_refined,
            labels=labels_refined,
            breaks=breaks_refined
        )

    def get_kpoints(self):
        return np.array(self.K_list)

    def get_kpoints_cart(self):
        return self.K_list.dot(self.recip_lattice)

    @property
    def str_short(self):
        return f"Path() with {len(self.K_list)} points and labels {self.labels}"

    @property
    def recip_lattice(self):
        return self.pointgroup.recip_lattice

    def __str__(self):
        return (
            "\n" + "\n".join(
                "  ".join(f"{x:10.6f}"
                          for x in k) + ((" <--- " + self.labels[i]) if i in self.labels else "") + (
                              ("\n" + "-" * 20) if i in self.breaks else "") for i, k in enumerate(self.K_list)))

    def get_K_list(self, use_symmetry=False):
        """ returns the list of K-points"""
        if use_symmetry:
            warnings.warn("symmetry is not used for a tabulation along path")
        print("generating K_list")
        K_list = [
            KpointBZpath(K=K, pointgroup=self.pointgroup)
            for K in self.K_list
        ]
        print("Done ")
        return K_list



    def getKline(self, break_thresh=np.inf):
        KPcart = self.K_list.dot(self.recip_lattice)
        K = np.zeros(KPcart.shape[0])
        k = np.linalg.norm(KPcart[1:, :] - KPcart[:-1, :], axis=1)
        k[k > break_thresh] = 0.0
        if len(self.breaks) > 0:
            k[self.breaks] = 0.0
        K[1:] = np.cumsum(k)
        return K
