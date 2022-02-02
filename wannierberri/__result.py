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
#
#  The purpose of this module is to provide Result classes for  
#  different types of  calculations. 
#  child classes can be defined specifically in each module

import numpy as np
from lazy_property import LazyProperty as Lazy

from .__utility import VoidSmoother


## A class to contain results or a calculation:
## For any calculation there should be a class with the samemethods implemented

class Result():

    def __init__(self):
        raise NotImplementedError()

#  multiplication by a number 
    def __mul__(self,other):
        raise NotImplementedError()

# +
    def __add__(self,other):
        raise NotImplementedError()

# -
    def __sub__(self,other):
        raise NotImplementedError()

# writing to a file
    def savetxt(self,name):
        raise NotImplementedError()
        
# saving as binary
    def save(self,name):
        raise NotImplementedError()
    

#  how result transforms under symmetry operations
    def transform(self,sym):
        raise NotImplementedError()

# a list of numbers, by each of those the refinement points will be selected
    @property
    def max(self):
        raise NotImplementedError()


### these methods do no need re-implementation: 
    def __rmul__(self,other):
        return self*other

    def __radd__(self,other):
        return self+other
        
    def __truediv__(self,number):
        return self*(1./number)


# a class for data defined for a set of Fermi levels
#Data is stored in an array data, where first dimension indexes the Fermi level


class EnergyResult(Result):
    """A class to store data dependent on several energies, e.g. Efermi and Omega
      Energy may also be an empty list, then the quantity does not depend on any energy (does it work?)

    Parameters
    -----------
    Energies  : 1D array or list of 1D arrays
        |  The energies, on which the data depend
        |  Energy may also be an empty list, then the quantity does not depend on any energy (does it work?)
    data : array(float) or array(complex)
        | the data. The first dimensions should match the sizes of the Energies arrays. The rest should be equal to 3
    smoothers :  a list of :class:`~wannierberri._utility.Smoother`
        | smoothers, one per each energy variable (usually do not need to be set by the calculator function. 
        | but are set automaticaly for Fermi levels and Omega's , and during the further * and + operations
    TRodd : bool 
        | True if the symmetric part of the result is Odd under time-reversal operation (False if it is even) 
        | relevant if system has TimeReversal, either alone or in combination with spatial symmetyries 
    TRtrans : bool 
        | True if the result is should be transposed under time-reversal operation. (i.e. the symmetric and 
        | antisymmetric parts have different parity under TR)
        | allowed only for rank-2 tensors
        | relevant if system has TimeReversal, either alone or in combination with spatial symmetyries 
    Iodd : bool 
        | `True` if the result is Odd under spatial inversion (`False` if it is even) 
        | relevant if system has Inversion, either alone or as part of other symmetyries (e.g. Mx=C2x*I)
    rank : int 
        | of the tensor, usually no need, to specify, it is set automatically to the number of dimensions
        | of the `data` array minus number of energies
    E_titles : list of str
        | titles to be printed above the energy columns
    file_npz : str 
        | path to a np file (if provided, the parameters `Enegries`, `data`, `TRodd`, `Iodd`, `rank` and 
        | `E_titles` are neglected)

     """
                      
    def __init__(self,Energies=None,data=None, smoothers=None,
                      TRodd=False,Iodd=False,TRtrans=False,rank=None,E_titles=["Efermi","Omega"],
                      file_npz = None):
        if file_npz is not None:
            res = np.load(open(file_npz,"rb"))
            energ = [res[f'Energies_{i}'] for i,_ in enumerate(res['E_titles'])]  # in binary mode energies are just two arrays
            try:
                TRtrans = res['TRtrans']
            except KeyError:
                pass
            self.__init__(Energies=energ,data=res['data'],smoothers=smoothers,TRodd=res['TRodd'],Iodd=res['Iodd'],rank=res['rank'],E_titles=list(res['E_titles']),TRtrans=TRtrans)
        else:
            if not isinstance (Energies,(list,tuple)) : 
                Energies=[Energies]
            if not isinstance (E_titles,(list,tuple)) :
                E_titles=[E_titles]
            E_titles=list(E_titles)
    
            self.N_energies=len(Energies)
            if self.N_energies<=len(E_titles):
               self.E_titles=E_titles[:self.N_energies]
            else:
               self.E_titles=E_titles+["???"]*(self.N_energies-len(E_titles))
            self.rank=data.ndim-self.N_energies if rank is None else rank
            if self.rank>0:
                shape=data.shape[-self.rank:]
                assert np.all(np.array(shape)==3), "data.shape={}".format(data.shape)
            for i in range(self.N_energies):
                assert (Energies[i].shape[0]==data.shape[i]) , "dimension of Energy[{}] = {} does not match do dimension of data {}".format(i,Energy[i].shape[0],data.shape[i])
            self.Energies=Energies
            self.data=data
            self.set_smoother(smoothers)
            self.TRodd=TRodd
            self.TRtrans=TRtrans
            self.Iodd=Iodd
            if self.TRtrans :
                assert self.rank == 2
        
    
    def set_smoother(self, smoothers):
        if smoothers is None:
            smoothers = (None,)*self.N_energies
        if not isinstance (smoothers,(list,tuple)) :
            smoothers=[smoothers]
        assert len(smoothers)==self.N_energies
        self.smoothers = [(VoidSmoother() if s is None else s) for s in  smoothers]

    @Lazy
    def dataSmooth(self):
        data_tmp=self.data.copy()
        for i in range(self.N_energies-1,-1,-1):
            data_tmp=self.smoothers[i](self.data,axis=i)
        return data_tmp

    def mul_array(self,other,axes=None):
        if isinstance(axes,int): 
            axes=(axes,)
        if axes is None: 
            axes = tuple(range(other.ndim))
        for i,d in enumerate(other.shape):
            assert d==self.data.shape[axes[i]], "shapes  {} should match the axes {} of {}".format(other.shape,axes,self.data.shape)
        reshape=tuple((self.data.shape[i] if i in axes else 1) for i in range(self.data.ndim))
        return EnergyResult(Energies = self.Energies,data = self.data*other.reshape(reshape),smoothers = self.smoothers,TRodd = self.TRodd,TRtrans = self.TRtrans, Iodd = self.Iodd,rank = self.rank,E_titles = self.E_titles)


    def __mul__(self,other):
        if isinstance(other,int) or isinstance(other,float) :
            return EnergyResult(Energies = self.Energies,data = self.data*other,smoothers = self.smoothers,TRodd = self.TRodd, TRtrans = self.TRtrans, Iodd = self.Iodd,rank = self.rank,E_titles = self.E_titles)
        else:
            raise TypeError("result can only be multilied by a number")

    def __add__(self,other):
        assert self.TRodd == other.TRodd
        assert self.TRtrans == other.TRtrans
        assert self.Iodd  == other.Iodd
        if other == 0:
            return self
        for i in range(self.N_energies):
            if np.linalg.norm(self.Energies[i]-other.Energies[i])>1e-8:
                raise RuntimeError ("Adding results with different energies {} ({}) - not allowed".format(i,self.E_titles[i]))
            if self.smoothers[i] != other.smoothers[i]:
                raise RuntimeError ("Adding results with different smoothers [i]: {} and {}".format(i,self.smoothers[i],other.smoothers[i]))
        return EnergyResult(Energies = self.Energies, data = self.data+other.data, smoothers = self.smoothers, TRodd = self.TRodd, TRtrans = self.TRtrans, Iodd = self.Iodd,rank = self.rank, E_titles = self.E_titles)

    def __sub__(self,other):
        return self+(-1)*other

    def __write(self,data,datasm,i):
        if i>self.N_energies:
            raise ValueError("not allowed value i={} > {}".format(i,self.N_energies))
        elif i==self.N_energies:
            data_tmp=list(data.reshape(-1))+list(datasm.reshape(-1))
            if data.dtype == complex:
                return ["    "+"    ".join("{0:15.6e} {1:15.6e}".format(x.real,x.imag) for x in data_tmp )]
            elif data.dtype == float:
                return ["    "+"    ".join("{0:15.6e}".format(x) for x in  data_tmp )  ]
        else:
            return ["{0:15.6e}    {1:s}".format(E,s) for j,E in enumerate(self.Energies[i]) for s in self.__write(data[j],datasm[j],i+1) ]

    def savetxt(self,name):
        frmt="{0:^31s}" if self.data.dtype == complex else "{0:^15s}"
        def getHead(n):
            if n<=0:
                return ['  ']
            else:
                return [a+b for a in 'xyz' for b in getHead(n-1)]

        head="#"+"    ".join("{0:^15s}".format(s) for s in self.E_titles)+" "*8+"    ".join(frmt.format(b) for b in getHead(self.rank)*2)+"\n"
        name = name.format('')

        open(name,"w").write(head+"\n".join(self.__write(self.data,self.dataSmooth,i=0)))


    def save(self,name):
        """
        writes a dictionary-like objectto file called `name`  with the folloing keys:
        - 'E_titles' : list of str - titles of the energies on which the result depends
        - 'Energies_0', ['Energies_1', ... ] - corresponding arrays of energies
        - data : array of shape (len(Energies_0), [ len(Energies_1), ...] , [3  ,[ 3, ... ]] )
        """
        name = name.format('')
        energ = {f'Energies_{i}':E for i,E in enumerate(self.Energies)}
        with open(name+".npz","wb") as f:
            np.savez_compressed(f,E_titles=self.E_titles,data=self.data,rank=self.rank,TRodd=self.TRodd,TRtrans = self.TRtrans,Iodd=self.Iodd,**energ)

    @property
    def _maxval(self):
        return np.abs(self.dataSmooth).max()

    @property
    def _norm(self):
        return np.linalg.norm(self.dataSmooth)

    @property
    def _normder(self):
        return np.linalg.norm(self.dataSmooth[1:]-self.dataSmooth[:-1])
    
    @property
    def max(self):
        return np.array([self._maxval,self._norm,self._normder])

    def transform(self,sym):
        return EnergyResult(Energies = self.Energies ,
                            data     = sym.transform_tensor(self.data,self.rank,TRodd=self.TRodd,Iodd=self.Iodd,TRtrans = self.TRtrans),
                            smoothers = self.smoothers, 
                            TRodd = self.TRodd, TRtrans = self.TRtrans, Iodd = self.Iodd,
                            rank = self.rank, E_titles = self.E_titles)




class EnergyResultDict(EnergyResult):
    '''Stores a dictionary of instances of the class Result.'''
    
    def __init__(self, results):
        '''
        Initialize instance with a dictionary of results with string keys and values of type Result.
        '''
        self.results = results
        
    def set_smoother(self, smoother):
        for v in self.results.values():
            v.set_smoother(smoother)

    #  multiplication by a number 
    def __mul__(self, other):
        return EnergyResultDict({ k : v*other for k,v in self.results.items() })

    # +
    def __add__(self, other):
        if other == 0:
            return self
        results = { k : self.results[k] + other.results[k] for k in self.results if k in other.results }
        return EnergyResultDict(results) 

    # -
    def __sub__(self, other):
        return self + (-1)*other

    # writing to a text file
    def savetxt(self, name):
        for k,v in self.results.items():
            v.savetxt(name.format('-'+k+'{}')) # TODO: check formatting

    # writing to a binary file
    def save(self, name):
        for k,v in self.results.items():
            v.save(name.format('-'+k+'{}'))
            

    #  how result transforms under symmetry operations
    def transform(self, sym):
        results = { k : self.results[k].transform(sym)  for k in self.results}
        return EnergyResultDict(results)

    # a list of numbers, by each of those the refinement points will be selected
    @property
    def max(self):
        return np.array([x for v in self.results.values() for x in v.max])


class EnergyResultScalar(EnergyResult):
    def __init__(self,Energy,data,smoother=VoidSmoother()):
         super().__init__(Energy,data,smoother,TRodd=False,Iodd=False,rank=0)

class EnergyResultAxialV(EnergyResult):
    def __init__(self,Energy,data,smoother=VoidSmoother()):
         super().__init__(Energy,data,smoother,TRodd=True,Iodd=False,rank=1)

class EnergyResultPolarV(EnergyResult):
    def __init__(self,Energy,data,smoother=VoidSmoother()):
         super().__init__(Energy,data,smoother,TRodd=False,Iodd=True,rank=1)

class NoComponentError(RuntimeError):

    def __init__(self, comp,dim):
        # Call the base class constructor with the parameters it needs
        super().__init__("component {} does not exist for tensor with dimension {}".format(comp,dim))


class KBandResult(Result):

    def __init__(self,data,TRodd,Iodd):
        if isinstance(data,list):
            self.data_list=data
        else:
            self.data_list=[data]
        self.TRodd=TRodd
        self.Iodd=Iodd
        
    def fit(self,other):
        for var in ['TRodd','Iodd','rank','nband']:
            if getattr(self,var)!=getattr(other,var):
                return False
        return True

    @property
    def data(self):
        if len(self.data_list)>1:
            self.data_list=[np.vstack(self.data_list)]
        return self.data_list[0]

    
    @property
    def rank(self):
       return len(self.data_list[0].shape)-2

    @property
    def nband(self):
       return self.data_list[0].shape[1]

    @property
    def nk(self):
       return sum(data.shape[0] for data in   self.data_list)

    def __add__(self,other):
        assert self.fit(other)
        return KBandResult(self.data_list+other.data_list,self.TRodd,self.Iodd) 

    def to_grid(self,k_map):
        dataall=self.data
        data=np.array( [sum(dataall[ik] for ik in km)/len(km)   for km in k_map])
        return KBandResult(data,self.TRodd,self.Iodd) 


    def select_bands(self,ibands):
        return KBandResult(self.data[:,ibands],self.TRodd,self.Iodd)


    def average_deg(self,deg):
        for i,D in enumerate(deg):
           for ib1,ib2 in D:
             for j in range(len(self.data_list)):
              self.data_list[j][i,ib1:ib2]=self.data_list[j][i,ib1:ib2].mean(axis=0)
        return self


    def transform(self,sym):
        data=[sym.transform_tensor(data,rank=self.rank,TRodd=self.TRodd,Iodd=self.Iodd) for data in self.data_list]
        return KBandResult(data,self.TRodd,self.Iodd)


    def get_component(self,component=None):
        xyz={"x":0,"y":1,"z":2}
        dim=self.data.shape[2:]

        if True:
            if not  np.all(np.array(dim)==3):
                raise RuntimeError("dimensions of all components should be 3, found {}".format(dim))
                
            dim=len(dim)
            if component is not None:
                component=component.lower()
            if component == "" :
                component = None
            if dim==0:
                if component is None:
                    return self.data
                else:
                    raise NoComponentError(component,0) 
            elif dim==1:
                if component  in ["x","y","z"]:
                    return self.data[:,:,xyz[component]]
                elif component=='norm':
                    return np.linalg.norm(self.data,axis=-1)
                elif component=='sq':
                    return np.linalg.norm(self.data,axis=-1)**2
                else:
                    raise NoComponentError(component,1) 
            elif dim==2:
                if component=="trace":
                    return sum([self.data[:,:,i,i] for i in range(3)])
                else:
                    try :
                        return self.data[:,:,xyz[component[0]],xyz[component[1]]]
                    except IndexError:
                        raise NoComponentError(component,2) 
            elif dim==3:
                if component=="trace":
                    return sum([self.data[:,:,i,i,i] for i in range(3)])
                else:
                    try :
                        return self.data[:,:,xyz[component[0]],xyz[component[1]],xyz[component[2]]]
                    except IndexError:
                        raise NoComponentError(component,3) 
            elif dim==4:
                if component=="trace":
                    return sum([self.data[:,:,i,i,i,i] for i in range(3)])
                else:
                    try :
                        return self.data[:,:,xyz[component[0]],xyz[component[1]],xyz[component[2]],xyz[component[3]]]
                    except IndexError:
#                        raise RuntimeError("Unknown component {} for rank-4  tensors".format(component))
                        raise NoComponentError(component,4) 
            else: 
                raise RuntimeError("writing tensors with rank >4 is not implemented. But easy to do")

