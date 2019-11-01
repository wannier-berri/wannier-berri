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
import parent
from copy import deepcopy
from berry import calcImf


def tabEnk(data,ibands):
    Enk=data.E_K_only
    kpoints=data.kpoints_all
    return TABresult(kpoints=kpoints,Enk=Enk[:,ibands],basis=data.recip_lattice )


def tabXnk(data,quantities="",ibands=None):
    if quantities=="":
       return tabEnk(data,ibands)
       
    quantities=quantities.lower()
    if ibands is None:
        ibands=np.arange(data.nbands)

    Enk=data.E_K[:,ibands]
    if "v" in quantities:
        dEnk=data.delE_K[:,ibands]
    else:
        dEnk=None
        
    if "o" in quantities:
        berry=calcImf(data)[:,ibands]
    else:
        berry=None

    kpoints=data.kpoints_all
    return TABresult(kpoints=kpoints,Enk=Enk,dEnk=dEnk,berry=berry,basis=data.recip_lattice )

def tabEVnk(data,ibands=None):
    Enk=data.E_K
    dEnk=data.delE_K
    if ibands is None:
        ibands=np.arange(Enk.shape[1])
#    print ("dEnk={}".format(dEnk))
    dkx,dky,dkz=1./data.NKFFT
    kpoints=data.kpoints_all
    return TABresult(kpoints=kpoints,basis=data.recip_lattice,Enk=Enk[:,ibands],dEnk=dEnk[:,ibands] )



class TABresult(parent.Result):

    def __init__(self,kpoints,basis,Enk=None,dEnk=None,berry=None):
        self.nband=None
        self.grid=None
        self.gridorder=None
        self.basis=basis
        self.kpoints=np.array(kpoints,dtype=float)%1
        if berry is not None:
            assert len(kpoints)==berry.shape[0]
            self.berry=berry
            self.nband=berry.shape[1]
        else: 
            self.berry=None

        if Enk is not None:
            assert len(kpoints)==Enk.shape[0]
            self.Enk=Enk
            if self.nband is not None:
                assert(self.nband==Enk.shape[1])
            else:
                self.nband=Enk.shape[1]
        else: 
            self.Enk=None

        if dEnk is not None:
            assert len(kpoints)==dEnk.shape[0]
            self.dEnk=dEnk
            if self.nband is not None:
                assert(self.nband==dEnk.shape[1])
            else:
                self.nband=dEnk.shape[1]
        else: 
            self.dEnk=None

        
    def __mul__(self,other):
    #k-point factors do not play arole in tabulating quantities
        return self
    
    def __add__(self,other):
        if other == 0:
            return self
        if self.nband!=other.nband:
            raise RuntimeError ("Adding results with different number of bands {} and {} - not allowed".format(
                self.nband,other.nband) )

        if not ((self.berry is None) or (other.berry is None)):
            Berry=np.vstack( (self.berry,other.berry) )
        else:
            Berry=None

        if not ((self.Enk is None) or (other.Enk is None)):
            Enk=np.vstack( (self.Enk,other.Enk) )
        else:
            Enk=None

        if not ((self.dEnk is None) or (other.dEnk is None)):
            dEnk=np.vstack( (self.dEnk,other.dEnk) )
        else:
            dEnk=None


        return TABresult(np.vstack( (self.kpoints,other.kpoints) ), basis=self.basis,berry=Berry,Enk=Enk,dEnk=dEnk) 


    def write(self,name):
        return   # do nothing so far
        raise NotImplementedError()
        open(name,"w").write(
           "    ".join("{0:^15s}".format(s) for s in ["EF",]+
                [b for b in ("x","y","z")*2])+"\n"+
          "\n".join(
           "    ".join("{0:15.6f}".format(x) for x in [ef]+[x for x in ahc]) 
                      for ef,ahc in zip (self.Efermi,np.hstack( (self.AHC,self.AHCsmooth) ) ))
               +"\n") 

    def transform(self,sym):
        kpoints=[sym.transform_k_vector(k,self.basis) for k in self.kpoints]
        berry=None if self.berry is None else sym.transform_axial_vector(self.berry)
        dEnk =None if self.dEnk  is None else sym.transform_v_vector(self.dEnk)
        return TABresult(kpoints=kpoints,basis=self.basis,berry=berry,Enk=self.Enk,dEnk=dEnk)

    def to_grid(self,grid,order='C'):
        grid1=[np.linspace(0.,1.,g,False) for g in grid]
#        print ( np.meshgrid(grid1[0],grid1[1],grid1[2],indexing='ij') )
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

        
#        print (np.array(k_map))
#            print ("k_grid={}".format(k))
#            for ik in k_map[-1]:
#                print (self.Enk[ik])

#        weights=[]
#        print ("defining weights")
#        for ik,km in enumerate(k_map):
#            if len(km)==0: 
#                raise NotImplementedError(
#                   "Grid point {}=[{},{},{}] was not calculated. Interpolation needed, which is to be implemented".format(
#                     ik,ik//(grid[1]*grid[2])/grid[0], (ik//grid[2])%(grid[1])/grid[1] ,
#                        (ik%grid[2])/grid[2])        )
#            weights.append(np.array([1./len(km)]*len(km)))
        def __collect(Xnk):
            if Xnk is None:
                return None
            else:
                return   np.array( [sum(Xnk[ik] for ik in km)/len(km) 
                           for km in k_map])
        print ("collecting")
        res=TABresult( k_new,basis=self.basis,berry=__collect(self.berry) ,  
                    Enk=__collect(self.Enk),dEnk=__collect(self.dEnk))
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
#        print (self.Enk.shape,iband)
        for iband in range(self.nband):
            FSfile+="".join("{0:.8f}\n".format(x) for x in self.Enk[:,iband]-efermi )
        
#        return FSfile
        xyz={"x":0,"y":1,"z":2}
        
        if quantity=="":
            return FSfile

        def _getComp(X,i):
            if X is None:
               raise RuntimeError("requested quantity '{}' was not calculated".format(quantity))
            if i in "xyz":
               return X[:,:,xyz[i]]
            elif i=='n':
               return np.linalg.norm(X,axis=-1)
            elif i=='s':
               return np.linalg.norm(X,axis=-1)**2
            
        quantity=quantity.lower()
        if quantity[0] == "v":
            Xnk=_getComp(self.dEnk,quantity[1])
        elif quantity[0] == "o":
            Xnk=_getComp(self.berry,quantity[1])
        
        for iband in range(self.nband):
            FSfile+="".join("{0:.8f}\n".format(x) for x in Xnk[:,iband] )

        
        return FSfile


    def max(self):
        if self.berry is not None:
            return [self.berry.max()]
        else:
            return [0]


