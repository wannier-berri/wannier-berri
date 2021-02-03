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
from time import time
from io import StringIO
import  multiprocessing 
import functools
from collections import Iterable
from .__utility import  print_my_name_start,print_my_name_end
from . import __result as result
from . import  __berry as berry
from . import  symmetry

#If one whants to add  new quantities to tabulate, just modify the following dictionaries

#should be functions of only one parameter of class Data_K
calculators={ 
         'spin'       : berry.calcSpin_band_kn, 
         'V'          : berry.calcV_band_kn  , 
         'morb'       : berry.calcImgh_band_kn,
         'berry'      : berry.calcImf_band_kn ,
         'hall_spin'  : berry.calcHall_spin_kn,
         'hall_orb'   : berry.calcHall_orb_kn
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
    return TABresult( kpoints=kpoints,recip_lattice=data.recip_lattice,results=results )



class TABresult(result.Result):

    def __init__(self,kpoints,recip_lattice,results={}):
        self.nband=results['E'].nband
        self.grid=None
        self.gridorder=None
        self.recip_lattice=recip_lattice
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
    #K-point factors do not play arole in tabulating quantities
        return self
    
    def __add__(self,other):
        if other == 0:
            return self
        if self.nband!=other.nband:
            raise RuntimeError ("Adding results with different number of bands {} and {} - not allowed".format(
                self.nband,other.nband) )
        results={r: self.results[r]+other.results[r] for r in self.results if r in other.results }
        return TABresult(np.vstack( (self.kpoints,other.kpoints) ), recip_lattice=self.recip_lattice,results=results) 

    def write(self,name):
        return   # do nothing so far

    def transform(self,sym):
        results={r:self.results[r].transform(sym)  for r in self.results}
        kpoints=[sym.transform_reduced_vector(k,self.recip_lattice) for k in self.kpoints]
        return TABresult(kpoints=kpoints,recip_lattice=self.recip_lattice,results=results)

    def to_grid(self,grid,order='C'):
        print ("settinng the grid")
        grid1=[np.linspace(0.,1.,g,False) for g in grid]
        print ("setting new kpoints")
        k_new=np.array(np.meshgrid(grid1[0],grid1[1],grid1[2],indexing='ij')).reshape((3,-1),order=order).T
        k_map=[[] for i in range(np.prod(grid))]
        print ("finding equivalent kpoints")
        for ik,k in enumerate(self.kpoints):
            k1=k*grid
            ik1=np.array(k1.round(),dtype=int)
            if np.linalg.norm(ik1/grid-k)<1e-5 : 
                ik1=ik1%grid
                ik2=ik1[2]+grid[2]*(ik1[1] + grid[1]*ik1[0])
#                print (ik,k,ik1,ik2)
                k_map[ik1[2]+grid[2]*(ik1[1] + grid[1]*ik1[0])].append(ik)
            else:
                print ("WARNING: k-point {}={} is skipped".format(ik,k))

        t0=time()
        print ("collecting")
        results={r:self.results[r].to_grid(k_map)  for r in self.results}
        t1=time()
        print ("collecting: to_grid  : {}".format(t1-t0))
        res=TABresult( k_new,recip_lattice=self.recip_lattice,results=results)
        t2=time()
        print ("collecting: TABresult  : {}".format(t2-t1))
        res.grid=np.copy(grid)
        res.gridorder=order
        t3=time()
        print ("collecting - OK : {} ({})".format(t3-t0,t3-t2))
        return res


    def __get_data_grid(self,quantity,iband,component=None,efermi=None):
        if quantity=='E':
            return self.Enk.data[:,iband].reshape(self.grid)
        elif component==None:
            return self.results[quantity].data[:,iband].reshape(tuple(self.grid)+(3,)*self.results[quantity].rank)
        else:
            return self.results[quantity].get_component(component)[:,iband].reshape(self.grid)


    def __get_data_path(self,quantity,iband,component=None,efermi=None):
        if quantity=='E':
            return self.Enk.data[:,iband]
        elif component==None:
            return self.results[quantity].data[:,iband]
        else:
            return self.results[quantity].get_component(component)[:,iband]
 
    def get_data(self,quantity,iband,component=None,efermi=None):
        if self.grid is None:
            return self.__get_data_path(quantity,iband,component=component,efermi=efermi)
        else : 
            return self.__get_data_grid(quantity,iband,component=component,efermi=efermi)


    def fermiSurfer(self,quantity=None,component=None,efermi=0,npar=0,iband=None,frmsf_name=None):
        if iband is None:
            iband=np.arange(self.nband)
        elif isinstance(iband, int):
            iband=[iband]
        if not (quantity is None):
            Xnk=self.results[quantity].get_component(component)
        if self.grid is None:
            raise RuntimeError("the data should be on a grid before generating FermiSurfer files. use to_grid() method")
        if self.gridorder!='C':
            raise RuntimeError("the data should be on a 'C'-ordered grid for generating FermiSurfer files")
        FSfile=""
        FSfile+=(" {0}  {1}  {2} \n".format(self.grid[0],self.grid[1],self.grid[2]))
        FSfile+=("1 \n" ) # so far only this option of Fermisurfer is implemented
        FSfile+=("{} \n".format(len(iband)))
        FSfile+=("".join( ["  ".join("{:14.8f}".format(x) for x in v) + "\n" for v in self.recip_lattice] ))

        FSfile+=_savetxt(a=self.Enk.data[:,iband].flatten(order='F')-efermi,npar=npar)
#        for iband in range(self.nband):
#            np.savetxt(FSfile,self.Enk.data[:,iband]-efermi,fmt="%.8f")
#            FSfile+="".join("{0:.8f}\n".format(x) for x in self.Enk.data[:,iband]-efermi )
        
        if quantity is None:
            return FSfile

        if quantity not in self.results:
            raise RuntimeError("requested quantity '{}' was not calculated".format(quantity))
            return FSfile
        FSfile+=_savetxt(a=Xnk[:,iband].flatten(order='F'),npar=npar)
#        for iband in range(self.nband):
#            np.savetxt(FSfile,Xnk[:,iband]-efermi,fmt="%.8f")
#            FSfile+="".join("{0:.8f}\n".format(x) for x in Xnk[:,iband] )
        if frmsf_name is not None:
            if not (frmsf_name.endswith(".frmsf")):
                frmsf_name+=".frmsf"
            open(frmsf_name,"w").write(FSfile)
        return FSfile

    def plot_path_fat(self, 
                  path,
                  quantity=None,
                  component=None,
                  save_file=None,
                  Eshift=0,
                  Emin=None,  Emax=None,
                  iband=None,
                  mode="fatband",
                  fatfactor=20,
                  cut_k=True
                  ):
        """a routine to plot a result along the path"""

        import matplotlib.pyplot as plt
        if iband is None: 
            iband=np.arange(self.nband)
        elif isinstance(iband,int):
            iband=np.array([iband])
        elif isinstance(iband,Iterable):
            iband=np.array(iband)
            if iband.dtype!=int:
                raise ValueError("iband should be integer")
        else:
            raise ValueError("iband should be either an integer, or array of intergers, or None")


        kline=path.getKline()
        E=self.get_data(quantity='E',iband=iband)-Eshift
        print ("shape of E",E.shape)

        plt.ylabel(r"$E$, eV")
        if Emin is None:
            Emin=E.min()-0.5
        if Emax is None:
            Emax=E.max()+0.5

        klineall=[]
        for ib in iband:
            e=E[:,ib]
            selE=(e<=Emax)*(e>=Emin)
            klineselE=kline[selE]
            klineall.append(klineselE)
            plt.plot(klineselE,e[selE],color="gray")
        if cut_k:
            klineall=[k for kl in klineall for k in kl]
            kmin=min(klineall)
            kmax=max(klineall)
        else:
            kmin=kline.min()
            kmax=kline.max()

        if quantity is not None:
            data=self.get_data(quantity='berry',iband=iband,component=component)
            print ("shape of data",data.shape)
            if mode=="fatband" :
                for ib in iband:
                    e=E[:,ib]
                    selE=(e<=Emax)*(e>=Emin)
                    klineselE=kline[selE]
                    y=data[selE][:,ib]
                    e1=e[selE]
                    for col,sel in [("red",(y>0)),("blue",(y<0))]:
#                        print (col,ib,kline.shape,e.shape,kline[sel].shape,e[sel].shape)
                        plt.scatter(klineselE[sel],e1[sel],s=abs(y[sel])*fatfactor,color=col)
            else :
                raise ValueError("So far only fatband mode is implemented")



        x_ticks_labels    = []  
        x_ticks_positions = [] 
        for k,v in path.labels.items():
            x_ticks_labels.append(v) 
            x_ticks_positions.append(kline[k]) 
            plt.axvline(x=kline[k] )
        plt.xticks(x_ticks_positions, x_ticks_labels )
        plt.ylim([Emin,Emax])
        plt.xlim([kmin,kmax])

        if save_file is None:
           plt.show()
        else:
           plt.savefig(save_file)
        plt.close()


    def max(self):
        raise NotImplementedError("adaptive refinement cannot be used for tabulating")


def _savetxt(limits=None,a=None,fmt=".8f",npar=0):
    assert a.ndim==1 , "only 1D arrays are supported. found shape{}".format(a.shape)
    if npar<=0:
        if limits is None:
            limits=(0,a.shape[0])
        fmtstr="{0:"+fmt+"}\n"
        return "".join(fmtstr.format(x) for x in a[limits[0]:limits[1]] )
    else:
        if limits is not None: 
            raise ValueError("limits shpould not be used in parallel mode")
        nppproc=a.shape[0]//npar+(1 if a.shape[0]%npar>0 else 0)
        print ("using a pool of {} processes to write txt frmsf of {} points".format(npar,nppproc))
        asplit=[(i,i+nppproc) for i in range(0,a.shape[0],nppproc)]
        p=multiprocessing.Pool(npar)
        res= p.map(functools.partial(_savetxt,a=a,fmt=fmt,npar=0)  , asplit)
        p.close()
        return "".join(res)

