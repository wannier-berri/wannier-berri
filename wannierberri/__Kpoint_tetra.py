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
#------------------------------------------------------------
# This is an auxilary class for the __evaluate.py  module

import numpy as np
import lazy_property
from .__Kpoint import KpointBZ
from collections.abc import Iterable
# fixing the order of the edges by their vortices
EDGES = [ [0,1], [0,2], [0,3], [1,2], [1,3], [2,3] ]
EDGES_COMPLEMENT  = [list(set([0,1,2,3]) - set(e) ) for e in EDGES ] 

class KpointBZtetra(KpointBZ):

    def __init__(self, vertices, basis=np.eye(3), K=0, NKFFT=np.ones(3), factor=1.,  refinement_level=0, split_level=0):
        cntr = np.mean(vertices, axis=0)
        assert (vertices.shape == (4,3) )
        self.K = K+cntr
        self.basis = basis
        self.vertices = vertices - cntr
        self.factor = factor
        self.res = None
        self.NKFFT = np.copy(NKFFT)
#        self.symgroup = symgroup
        self.refinement_level = refinement_level
        self.split_level = split_level

    def copy(self):
        return KpointBZtetra(vertices=self.vertices, K=self.K, NKFFT=self.NKFFT, factor=self.factor, basis=self.basis,
            refinement_level=self.refinement_level, split_level=self.split_level)

    @property
    def vertices_fullBZ(self):
        return  self.vertices/self.NKFFT

    @lazy_property.LazyProperty
    def __edge_lengths(self):
        edges = np.array([self.vertices[i[1]]-self.vertices[i[0]] for i in EDGES]).dot(self.basis)
        return np.linalg.norm(edges,axis = 1)

    @lazy_property.LazyProperty
    def size(self):
        return max(self.__edge_lengths)
    
    

    def divide(self,  ndiv=2, periodic=[True,True,True], use_symmetry=True,refine = True,):
        """
            we either 'split' (if the tetrahedra is too big) refine = False
             or 'refine' (if the result is big), but it only matters for the counters
        """
#        print (f"splitting into {ndiv} pieces")
        if isinstance(ndiv, Iterable):
            ndiv = ndiv[0]
        if not(np.all(periodic)):
            raise ValueError("tetrahedron grid can be used only for 3D-periodic systems")
        i_edge = np.argmax(self.__edge_lengths)
        edge = EDGES[ i_edge ]
        edge_comp = EDGES_COMPLEMENT[ i_edge ]
        v0 = self.vertices[edge[0]]
        dv = (self.vertices[edge[1]]-v0)/ndiv
        add_list = []
        # TODO - account for the case of odd ndiv - return the same point instead of creating new one
        for i in range(ndiv):
            add_list.append(
                KpointBZtetra(
                    K = self.K,
                    vertices = np.array([self.vertices[edge_comp[0]],
                                         self.vertices[edge_comp[1]],
                                         v0+i*dv, v0+(i+1)*dv] ),
                    factor = self.factor/ndiv,
                    basis = self.basis, 
                    refinement_level = self.refinement_level + int(refine),
                    split_level = self.split_level+int (not refine),

            ) )
#        print (f"splitted {self.size} intp {[k.size for k in add_list]}")
        self.factor = 0.
        return add_list

