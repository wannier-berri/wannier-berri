#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                                                            #
#------------------------------------------------------------#
#                                                            #
#  written  by                                               #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#

import numpy as np
from scipy import constants as constants
from collections import Iterable

from utility import  print_my_name_start,print_my_name_end,voidsmoother
import result
from copy import deepcopy
from berry import calcImf_band,calcImgh_band,calcV_band
from spin import calcSpin_band
import symmetry

#If one whants to add  new quantities to tabulate, just modify the following dictionaries
calculators={ 
         'spin'  : calcSpin_band, 
         'V'     : calcV_band  , 
         'morb'  : calcImgh_band,
         'berry' : calcImf_band }


transformators={ 
         'E'     : 'scalar', 
         'spin'  : 'axial_vector', 
         'V'     : 'v_vector', 
         'morb'  : 'axial_vector', 
         'berry' : 'axial_vector'  }


#the rest should work itself 
# so far only for vector quantities
#TODO : generalize to scalars and tensors

def tabXnk(data,quantities=[],ibands=None):

    if ibands is None:
        ibands=np.arange(data.nbands)


    Enk=data.E_K[:,ibands]
       
    degen_thresh=1e-5
    A=[np.hstack( ([0],np.where(E[1:]-E[:1]>degen_thresh)[0]+1, [E.shape[0]]) ) for E in Enk ]
    deg= [[(ib1,ib2) for ib1,ib2 in zip(a,a[1:])] for a in A]

    results={'E':Enk}
    for q in quantities:
        A=calculators[q](data)[:,ibands]
        results[q]=A
        for i,D in enumerate(deg):
           for ib1,ib2 in D:
                  A[i,ib1:ib2]=A[i,ib1:ib2].mean(axis=0)

    kpoints=data.kpoints_all
    return TABresult(kpoints=kpoints,basis=data.recip_lattice,results=results )



class TABresult(result.Result):

    def __init__(self,kpoints,basis,results={}):
        self.nband=results['E'].shape[1]
        self.grid=None
        self.gridorder=None
        self.basis=basis
        self.kpoints=np.array(kpoints,dtype=float)%1

        self.results=results
        for r in results:
            print (r,self.nband,len(self.kpoints),results[r].shape)
            assert len(kpoints)==results[r].shape[0]
            assert self.nband==results[r].shape[1]
            
    @property 
    def Enk(self):
        return self.results['E']

        
    def __mul__(self,other):
    #k-point factors do not play arole in tabulating quantities
        return self
    
    def __add__(self,other):
        if other == 0:
            return self
        if self.nband!=other.nband:
            raise RuntimeError ("Adding results with different number of bands {} and {} - not allowed".format(
                self.nband,other.nband) )
        results={r: np.vstack( (self.results[r],other.results[r]) ) for r in self.results if r in other.results }
        return TABresult(np.vstack( (self.kpoints,other.kpoints) ), basis=self.basis,results=results) 

    def write(self,name):
        return   # do nothing so far

    def transform(self,sym):
        results={r:sym.transform(transformators[r],self.results[r]) for r in self.results}
        kpoints=[sym.transform_k_vector(k,self.basis) for k in self.kpoints]
        return TABresult(kpoints=kpoints,basis=self.basis,results=results)

    def to_grid(self,grid,order='C'):
        grid1=[np.linspace(0.,1.,g,False) for g in grid]
        print ("setting new kpoints")
        k_new=np.array(np.meshgrid(grid1[0],grid1[1],grid1[2],indexing='ij')).reshape((3,-1),order=order).T
        k_map=[[] for i in range(np.prod(grid))]
        print ("finding equivalent kpoints")
        for ik,k in enumerate(self.kpoints):
            k1=k*grid
            ik1=np.array(k1.round(),dtype=int)
            if np.linalg.norm(k1-ik1)<1e-8 : 
                ik1=ik1%grid
                ik2=ik1[2]+grid[2]*(ik1[1] + grid[1]*ik1[0])
#                print (ik,k,ik1,ik2)
                k_map[ik1[2]+grid[2]*(ik1[1] + grid[1]*ik1[0])].append(ik)
            else:
                print ("WARNING: k-point {}={} is skipped".format(ik,k))

        
        print ("collecting")
        results={r:np.array( [sum(self.results[r][ik] for ik in km)/len(km)   for km in k_map])  for r in self.results}
        res=TABresult( k_new,basis=self.basis,results=results)
        res.grid=np.copy(grid)
        res.gridorder=order
        return res
            
    
    def fermiSurfer(self,quantity="",efermi=0):
        if self.grid is None:
            raise RuntimeError("the data should be on a grid before generating FermiSurfer files. use to_grid() method")
        if self.gridorder!='C':
            raise RuntimeError("the data should be on a 'C'-ordered grid for generating FermiSurfer files")
        FSfile=" {0}  {1}  {2} \n".format(self.grid[0],self.grid[1],self.grid[2])
        FSfile+="1 \n"  # so far only this option of Fermisurfer is implemented
        FSfile+="{} \n".format(self.nband)
        FSfile+="".join( ["  ".join("{:14.8f}".format(x) for x in v) + "\n" for v in self.basis] )
        for iband in range(self.nband):
            FSfile+="".join("{0:.8f}\n".format(x) for x in self.Enk[:,iband]-efermi )
        
        
        if quantity=="":
            return FSfile

        quantity,comp=quantity.split("_")
        if quantity not in self.results:
            raise RuntimeError("requested quantity '{}' was not calculated".format(quantity))
        
        xyz={"x":0,"y":1,"z":2}
    
        X=self.results[quantity]
        if comp  in "xyz":
            Xnk = X[:,:,xyz[comp]]
        elif comp=='n':
            Xnk =  np.linalg.norm(X,axis=-1)
        elif comp=='s':
            Xnk = np.linalg.norm(X,axis=-1)**2
                
        for iband in range(self.nband):
            FSfile+="".join("{0:.8f}\n".format(x) for x in Xnk[:,iband] )

        return FSfile


    def max(self):
        raise NotImplementedError("adaptive refinement cannot be used for tabulating")


