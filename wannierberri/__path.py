from .__grid import Grid
from  .__Kpoint import KpointBZ
from .__utility import warning
from collections import Iterable
import numpy as np

class Path(Grid):
    """ A class containing information about the k-path

    Parameters
    -----------
    system : :class:`~wannierberri.__system.System` 
        which the calculations will be made
    length :  float
        (angstroms) -- in this case the grid is NK[i]=length*||B[i]||/2pi  B- reciprocal lattice
    dk :  float
        (inverse angstroms) -- in this case the grid is NK[i]=length*||B[i]||/2pi  B- reciprocal lattice
    k_nodes : list
        | cordinates of the nodes in the the reduced coordinates. Some entries may be None - which means that the segment should be skipped
        | No labels or nk's should be assigned to None nodes
    nk : int  or list or numpy.array(3) 
        number of k-points along each directions 
    k_list : array-like
        coordinatres of all k-points in the reduced coordinates
    labels : list  of dict
        | if k_list is set - it is a dict {i:lab} with i - index of k-point, lab - corresponding label (not alll kpoints need to be labeled
        | if k_nodes is set - it is a list of labels, one for every node
    Notes
    -----
    user needs to specify either `k_list` or (`k_nodes` + (`length` or `nk` or dk))

    """

    def __init__(self,system,k_list=None,k_nodes=None,length=None,dk=None,nk=None,labels=None,breaks=[]):

        self.symgroup=system.symgroup
        self.FFT=np.array([1,1,1])
        self.findif=None
        self.breaks=breaks
        if k_list is not None:
            self.K_list=np.array(k_list)
            assert  self.K_list.shape[1]==3, "k_list should contain 3-vectors"
            assert  self.K_list.shape[0]>0, "k_list should not be empty"
            for var in 'k_nodes','length','nk','dk':
                if locals()[var] is not None:
                    warning("k_list was entered manually, ignoring {}".format(var))
            self.labels={} if labels is None else labels
            self.breaks=[] if breaks is None else breaks
        else:
            if k_nodes is None:
                raise ValueError("need to specify either 'k_list' of 'k_nodes'")

            if labels is None:
                labels=[str(i+1) for i,k in enumerate([k for k in self.nodes if k is not None])]
            labels=(l for l in labels)
            labels=[None if k is None else next(labels)  for k in k_nodes]

            if length is not None:
                assert length>0
                if dk is not None:
                    raise ValueError("'length' and  'dk' cannot be set together")
                dk=2*np.pi/length
            if dk is not None:
                if nk is not None:
                    raise ValueError("'nk' cannot be set together with 'length' or 'dk' ")

            if isinstance(nk, Iterable):
                nkgen=(x for x in nk)
            else:
                nkgen=(nk for x in k_nodes)

            self.K_list=np.zeros((0,3))
            self.labels={}
            self.breaks=[]
            for start,end,l1,l2 in zip(k_nodes,k_nodes[1:],labels,labels[1:]) :
                if None not in (start,end):
                    self.labels[self.K_list.shape[0]]=l1
                    start=np.array(start)
                    end=np.array(end)
                    assert start.shape==end.shape==(3,)
                    if nk is not None:
                        _nk=nkgen
                    else: 
                        _nk=round( np.linalg.norm((start-end).dot(self.recip_lattice))/dk )+1
                        if _nk==1 : _nk=2
                    self.K_list=np.vstack( (self.K_list,start[None,:]+np.linspace(0,1.,_nk)[:,None]*(end-start)[None,:] ) )
                    self.labels[self.K_list.shape[0]-1]=l2
                elif end is None:
                    self.breaks.append(self.K_list.shape[0]-1)
        self.breaks=np.array(self.breaks)

    @property 
    def recip_lattice(self):
        return self.symgroup.recip_lattice

    def __str__(self):
        return ("\n"+"\n".join(
                         "  ".join("{:10.6f}".format(x) for x in k) + 
                                    ((" <--- "+self.labels[i]) if i in self.labels else "" )
                                     +   (("\n"+"-"*20) if i in self.breaks else "" )
                                for i,k in enumerate(self.K_list)
                   )  )

    def get_K_list(self):
        """ returns the list of Symmetry-irreducible K-points"""
        dK=np.array([1.,1.,1.])
        factor=1.
        print ("generating K_list")
        K_list =[KpointBZ(K=K,dK=dK,NKFFT=self.FFT,factor=factor,symgroup=self.symgroup,refinement_level=0) 
                    for K in self.K_list]
        print ("Done " )
        return K_list

    def getKline(self,break_thresh=np.Inf):
        KPcart = self.K_list.dot(self.recip_lattice)
        K = np.zeros(KPcart.shape[0])
        k = np.linalg.norm(KPcart[1:, :] - KPcart[:-1, :], axis=1)
        k[k > break_thresh] = 0.0
        k[self.breaks]=0.0
        K[1:] = np.cumsum(k)
        return K
