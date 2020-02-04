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

import numpy as np
from scipy import constants as constants
from collections import Iterable,defaultdict
from copy import deepcopy

from .__utility import  print_my_name_start,print_my_name_end,voidsmoother
from . import __result as result
from . import  __berry as berry
from . import  __spin as spin
from . import  __symmetry  as symmetry

#If one whants to add  new quantities to tabulate, just modify the following dictionaries

#should be functions of only one parameter of class data_dk
calculators={ 
         'spin'       : spin.calcSpin_band_kn, 
         'V'          : berry.calcV_band_kn  , 
         'morb'       : berry.calcImgh_band_kn,
         'berry'      : berry.calcImf_band_kn ,
         'hall_spin'  : spin.calcHall_spin_kn,
         'hall_orb'   : spin.calcHall_orb_kn
         }


additional_parameters=defaultdict(lambda: defaultdict(lambda:None )   )
additional_parameters_description=defaultdict(lambda: defaultdict(lambda:"no description" )   )


descriptions=defaultdict(lambda:"no description")
descriptions['berry']="Berry curvature"
descriptions['V']="velocity"
descriptions['spin']="Spin"
descriptions['morb']="orbital magnetic moment"
descriptions['hall_spin']="spin contribution to low-field Hall effect"
descriptions['hall_orb']="orbital contribution to low-field Hall effect"



def tabXnk(data,quantities=[],degen_thresh=None,ibands=None,parameters={}):

    if degen_thresh is not None:
        data.set_degen(degen_thresh=degen_thresh)

    if ibands is None:
        ibands=np.arange(data.nbands)


    Enk=data.E_K[:,ibands]
       
    degen_thresh=1e-5
    A=[np.hstack( ([0],np.where(E[1:]-E[:1]>degen_thresh)[0]+1, [E.shape[0]]) ) for E in Enk ]
    deg= [[(ib1,ib2) for ib1,ib2 in zip(a,a[1:])] for a in A]

    results={'E':result.KBandResult(Enk,TRodd=False,Iodd=False)}
    for q in quantities:
        __parameters={}
        for param in additional_parameters[q]:
            if param in parameters:
                 __parameters[param]=parameters[param]
            else :
                 __parameters[param]=additional_parameters[q][param]
        results[q]=calculators[q](data,**__parameters).select_bands(ibands).average_deg(deg)

    kpoints=data.kpoints_all
    return TABresult( kpoints=kpoints,basis=data.recip_lattice,results=results )



class TABresult(result.Result):

    def __init__(self,kpoints,basis,results={}):
        self.nband=results['E'].nband
        self.grid=None
        self.gridorder=None
        self.basis=basis
        self.kpoints=np.array(kpoints,dtype=float)%1

        self.results=results
        for r in results:
#            print (r,self.nband,len(self.kpoints),results[r].shape)
            assert len(kpoints)==results[r].nk
            assert self.nband==results[r].nband
            
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
        results={r: self.results[r]+other.results[r] for r in self.results if r in other.results }
        return TABresult(np.vstack( (self.kpoints,other.kpoints) ), basis=self.basis,results=results) 

    def write(self,name):
        return   # do nothing so far

    def transform(self,sym):
        results={r:self.results[r].transform(sym)  for r in self.results}
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
        results={r:self.results[r].to_grid(k_map)  for r in self.results}
        res=TABresult( k_new,basis=self.basis,results=results)
        res.grid=np.copy(grid)
        res.gridorder=order
        return res
            
    
    def fermiSurfer(self,quantity=None,component=None,efermi=0):
        if not (quantity is None):
            Xnk=self.results[quantity].get_component(component)

        if self.grid is None:
            raise RuntimeError("the data should be on a grid before generating FermiSurfer files. use to_grid() method")
        if self.gridorder!='C':
            raise RuntimeError("the data should be on a 'C'-ordered grid for generating FermiSurfer files")
        FSfile=" {0}  {1}  {2} \n".format(self.grid[0],self.grid[1],self.grid[2])
        FSfile+="1 \n"  # so far only this option of Fermisurfer is implemented
        FSfile+="{} \n".format(self.nband)
        FSfile+="".join( ["  ".join("{:14.8f}".format(x) for x in v) + "\n" for v in self.basis] )
        for iband in range(self.nband):
            FSfile+="".join("{0:.8f}\n".format(x) for x in self.Enk.data[:,iband]-efermi )
        
        if quantity is None:
            return FSfile
        
        if quantity not in self.results:
            raise RuntimeError("requested quantity '{}' was not calculated".format(quantity))
            return FSfile
        
        for iband in range(self.nband):
            FSfile+="".join("{0:.8f}\n".format(x) for x in Xnk[:,iband] )
        return FSfile


    def max(self):
        raise NotImplementedError("adaptive refinement cannot be used for tabulating")


