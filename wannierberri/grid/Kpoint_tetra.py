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
# This is an auxiliary class for the __evaluate.py  module

import numpy as np
from functools import cached_property
from .Kpoint import KpointBZ

# fixing the order of the edges by their vortices
EDGES = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
EDGES_COMPLEMENT = [list({0, 1, 2, 3} - set(e)) for e in EDGES]


class KpointBZtetra(KpointBZ):
    """describes the K-point and surrounding tetrahedron"""

    def __init__(self, vertices, basis=np.eye(3), K=0, NKFFT=np.ones(3), factor=1., refinement_level=0, split_level=0):
        cntr = np.mean(vertices, axis=0)
        super().__init__(K=np.array(K) + cntr, NKFFT=NKFFT, factor=factor, pointgroup=None,
                         refinement_level=refinement_level)
        assert (vertices.shape == (4, 3))
        self.basis = basis
        self.vertices = vertices - cntr
        self.split_level = split_level

    def copy(self):
        return KpointBZtetra(vertices=self.vertices, K=self.K, NKFFT=self.NKFFT, factor=self.factor, basis=self.basis,
                             refinement_level=self.refinement_level, split_level=self.split_level)

    @property
    def vertices_fullBZ(self):
        return self.vertices / self.NKFFT

    @cached_property
    def __edge_lengths(self):
        edges = np.array([self.vertices[i[1]] - self.vertices[i[0]] for i in EDGES]).dot(self.basis)
        return np.linalg.norm(edges, axis=1)

    @cached_property
    def __i_max_edge(self):
        """returns the index of the maximal edge.
        If there are equal edges, the edge with the smallest index (inthe EDGES array) is returned. This is done for reproducibility of the tests
        """
        lengths = self.__edge_lengths
        srt = np.argsort(lengths)
        #        lengths_sorted=lengths[srt]
        srt_short = srt[np.where(lengths[srt[-1]] - lengths[srt] < 1e-6)]
        #        print ("srt",srt
        return min(srt_short)

    @cached_property
    def size(self):
        return max(self.__edge_lengths)

    def divide(self, ndiv=2, periodic=(True, True, True), use_symmetry=True, refine=True):
        """
            we either 'split' (if the tetrahedra is too big) refine = False
             or 'refine' (if the result is big), but it only matters for the counters
        """
        if not np.all(periodic):
            raise ValueError("tetrahedron grid can be used only for 3D-periodic systems")
        i_edge = self.__i_max_edge
        edge = EDGES[i_edge]
        edge_comp = EDGES_COMPLEMENT[i_edge]
        v0 = self.vertices[edge[0]]
        dv = (self.vertices[edge[1]] - v0) / ndiv
        add_list = []
        # TODO - account for the case of odd ndiv - return the same point instead of creating new one
        for i in range(ndiv):
            add_list.append(
                KpointBZtetra(
                    K=self.K,
                    vertices=np.array([self.vertices[edge_comp[0]],
                                       self.vertices[edge_comp[1]],
                                       v0 + i * dv, v0 + (i + 1) * dv]),
                    factor=self.factor / ndiv,
                    basis=self.basis,
                    refinement_level=self.refinement_level + int(refine),
                    split_level=self.split_level + int(not refine),
                    NKFFT=self.NKFFT
                ))
        #        print (f"split {self.size} intp {[k.size for k in add_list]}")
        self.factor = 0.
        return add_list
