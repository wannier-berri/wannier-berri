from .__grid import GridAbstract
from .__Kpoint import KpointBZpath
from ..__utility import warning
from collections.abc import Iterable
import numpy as np


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
    k_nodes : list
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
        | if k_nodes is set - it is a list of labels, one for every node
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
    user needs to specify either `k_list` or (`k_nodes` + (`length` or `nk` or dk))

    """

    def __init__(
            self,
            system,
            k_list=None,
            k_nodes=None,
            length=None,
            dk=None,
            nk=None,
            labels=None,
            breaks=[],
            r1=None,
            r2=None,
            ntheta=None,
            nphi=None,
            origin=None):

        self.symgroup = system.symgroup
        self.FFT = np.array([1, 1, 1])
#        self.findif = None
        self.breaks = breaks

        if k_list == 'sphere':
            self.K_list = self.sphere(r1, r1, ntheta, nphi, origin)
            self.labels = ['sphere']
        elif k_list == 'spheroid':
            self.K_list = self.sphere(r1, r2, ntheta, nphi, origin)
            self.labels = ['spheroid']

        elif k_list is None:
            if k_nodes is None:
                raise ValueError("need to specify either 'k_list' of 'k_nodes'")

            if labels is None:
                labels = [str(i + 1) for i, k in enumerate([k for k in k_nodes if k is not None])]
            labels = (l for l in labels)
            labels = [None if k is None else next(labels) for k in k_nodes]

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
                nkgen = (nk for x in k_nodes)

            self.K_list = np.zeros((0, 3))
            self.labels = {}
            self.breaks = []
            for start, end, l1, l2 in zip(k_nodes, k_nodes[1:], labels, labels[1:]):
                if start is not None and end is not None:
                    self.labels[self.K_list.shape[0]] = l1
                    start = np.array(start)
                    end = np.array(end)
                    assert start.shape == end.shape == (3, )
                    if nk is not None:
                        _nk = next(nkgen)
                    else:
                        _nk = round(np.linalg.norm((start - end).dot(self.recip_lattice)) / dk) + 1
                        if _nk == 1:
                            _nk = 2
                    self.K_list = np.vstack(
                        (
                            self.K_list, start[None, :] + np.linspace(0, 1., _nk - 1, endpoint=False)[:, None] *
                            (end - start)[None, :]))
                elif end is None:
                    self.breaks.append(self.K_list.shape[0] - 1)
            self.K_list = np.vstack((self.K_list, k_nodes[-1]))
            self.labels[self.K_list.shape[0] - 1] = labels[-1]
        else:
            self.K_list = np.array(k_list)
            assert self.K_list.shape[1] == 3, "k_list should contain 3-vectors"
            assert self.K_list.shape[0] > 0, "k_list should not be empty"
            for var in 'k_nodes', 'length', 'nk', 'dk':
                if locals()[var] is not None:
                    warning("k_list was entered manually, ignoring {}".format(var))
            self.labels = {} if labels is None else labels
            self.breaks = [] if breaks is None else breaks
        self.div = np.shape(self.K_list)[0]
        self.breaks = np.array(self.breaks, dtype=int)

    def sphere(self, r1, r2, ntheta, nphi, origin):
        theta = np.linspace(0, np.pi, ntheta, endpoint=True)
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=True)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        sphere = [
            r1 * np.cos(phi_grid) * np.sin(theta_grid), r1 * np.sin(phi_grid) * np.sin(theta_grid),
            r2 * np.cos(theta_grid)
        ]
        cart_k_list = np.array(sphere).reshape(3, ntheta * nphi).transpose(1, 0)
        list_k = cart_k_list.dot(np.linalg.inv(self.recip_lattice)) - origin[None, :]
        with open("klist_cart.txt", "w") as f:
            for i in range(ntheta * nphi):
                f.write(f"{cart_k_list[i, 0]:12.6f}{cart_k_list[i, 1]:12.6f}{cart_k_list[i, 2]:12.6f}\n")
        return list_k

    @property
    def str_short(self):
        return "Path() with {} points and labels {}".format(len(self.K_list), self.labels)

    @property
    def recip_lattice(self):
        return self.symgroup.recip_lattice

    def __str__(self):
        return (
            "\n" + "\n".join(
                "  ".join("{:10.6f}".format(x)
                          for x in k) + ((" <--- " + self.labels[i]) if i in self.labels else "") + (
                              ("\n" + "-" * 20) if i in self.breaks else "") for i, k in enumerate(self.K_list)))

    def get_K_list(self, use_symmetry=False):
        """ returns the list of K-points"""
        if use_symmetry:
            print("WARNING : symmetry is not used for a tabulation along path")
        print("generating K_list")
        K_list = [
            KpointBZpath(K=K, symgroup=self.symgroup)
            for K in self.K_list
        ]
        print("Done ")
        return K_list

    def getKline(self, break_thresh=np.Inf):
        KPcart = self.K_list.dot(self.recip_lattice)
        K = np.zeros(KPcart.shape[0])
        k = np.linalg.norm(KPcart[1:, :] - KPcart[:-1, :], axis=1)
        k[k > break_thresh] = 0.0
        if len(self.breaks) > 0:
            k[self.breaks] = 0.0
        K[1:] = np.cumsum(k)
        return K
