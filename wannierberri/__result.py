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
from copy import deepcopy

from .__utility import voidsmoother


## A class to contain results or a calculation:
## For any calculation there should be a class with the samemethods implemented

class Result():

    def __init__(self):
        raise NotImprementedError()

#  multiplication by a number 
    def __mul__(self,other):
        raise NotImprementedError()

# +
    def __add__(self,other):
        raise NotImprementedError()

# -
    def __sub__(self,other):
        raise NotImprementedError()

# writing to a file
    def write(self,name):
        raise NotImprementedError()

#  how result transforms under symmetry operations
    def transform(self,sym):
        raise NotImprementedError()

# a list of numbers, by each of those the refinement points will be selected
    @property
    def max(self):
        raise NotImprementedError()


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


    def __init__(self,Energy,data,smoother=voidsmoother,TRodd=False,Iodd=False,rank=None):
        self.rank=len(data.shape[1:]) if rank is None else rank
        if self.rank>0:
            shape=data.shape[-self.rank:]
            assert np.all(np.array(shape)==3)
        assert (Energy.shape[0]==data.shape[0])
        self.Energy=Energy
        self.data=data
        self.smoother=smoother
        self.TRodd=TRodd
        self.Iodd=Iodd
    
    def set_smoother(self, smoother):
        self.smoother = smoother

    @Lazy
    def dataSmooth(self):
        return self.smoother(self.data)

    def mul_array(self,other):
#        print ('multiplying result by array',other)
        assert np.all(other.shape==self.data.shape[:len(other.shape)]), "shapes should match {} and {}".format(other.shape,self.data.shape)
        reshape=other.shape+(1,)*(len(self.data.shape)-len(other.data.shape))
        return EnergyResult(self.Energy,self.data*other.reshape(reshape),self.smoother,self.TRodd,self.Iodd,self.rank)


    def __mul__(self,other):
        if isinstance(other,int) or isinstance(other,float) :
#            print ('multiplying result by number',other)
            return EnergyResult(self.Energy,self.data*other,self.smoother,self.TRodd,self.Iodd,self.rank)
        else:
            raise TypeError("result can only be multilied by a number")

    def __add__(self,other):
        assert self.TRodd == other.TRodd
        assert self.Iodd  == other.Iodd
        if other == 0:
            return self
        if np.linalg.norm(self.Energy-other.Energy)>1e-8:
            raise RuntimeError ("Adding results with different Fermi energies - not allowed")
        if self.smoother != other.smoother:
            raise RuntimeError ("Adding results with different smoothers ={} and {}".format(self.smoother,other.smoother))
        return EnergyResult(self.Energy,self.data+other.data,self.smoother,self.TRodd,self.Iodd,self.rank)

    def __sub__(self,other):
        return self+(-1)*other

    def _write_complex(self, name):
        '''Writes the result if data has complex entries.'''
        # assume that the dimensions starting from first are cartesian coordinates       
        def getHead(n):
            if n<=0:
                return ['  ']
            else:
                return [a+b for a in 'xyz' for b in getHead(n-1)]
        rank=len(self.data.shape[1:])

        open(name,"w").write(
           "    ".join("{0:^30s}".format(s) for s in ["# EF",]+
                [b for b in getHead(rank)*2])+"\n"+
          "\n".join(
           "    ".join("{0:15.6e}{1:15.6e}".format(np.real(x),np.imag(x)) for x in [ef]+[x for x in data.reshape(-1)]+[x for x in datasm.reshape(-1)]) 
                      for ef,data,datasm in zip (self.Energy,self.data,self.dataSmooth)  )
               +"\n") 

    def write(self,name):
        name = name.format('')
        if (self.data.dtype == np.dtype('complex')):
            self._write_complex(name)
        else:
            # assume that the dimensions starting from first are cartesian coordinates       
            def getHead(n):
               if n<=0:
                  return ['  ']
               else:
                  return [a+b for a in 'xyz' for b in getHead(n-1)]
            rank=len(self.data.shape[1:])

            open(name,"w").write(
               "    ".join("{0:^15s}".format(s) for s in ["# EF",]+
                    [b for b in getHead(rank)*2])+"\n"+
              "\n".join(
               "    ".join("{0:15.6e}".format(x) for x in [ef]+[x for x in data.reshape(-1)]+[x for x in datasm.reshape(-1)]) 
                          for ef,data,datasm in zip (self.Energy,self.data,self.dataSmooth)  )
                   +"\n") 

    @property
    def _maxval(self):
        if self.dataSmooth.dtype == np.dtype('complex'):
            return np.maximum(np.real(self.dataSmooth).max(), np.imag(self.dataSmooth).max())
        else:
            return self.dataSmooth.max() 

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
        return EnergyResult(self.Energy,sym.transform_tensor(self.data,self.rank,TRodd=self.TRodd,Iodd=self.Iodd),self.smoother,self.TRodd,self.Iodd,self.rank)


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

    # writing to a file
    def write(self, name):
        for k,v in self.results.items():
            v.write(name.format('-'+k+'{}')) # TODO: check formatting

    #  how result transforms under symmetry operations
    def transform(self, sym):
        results = { k : self.results[k].transform(sym)  for k in self.results}
        return EnergyResultDict(results)

    # a list of numbers, by each of those the refinement points will be selected
    @property
    def max(self):
        return np.array([x for v in self.results.values() for x in v.max])


class EnergyResultScalar(EnergyResult):
    def __init__(self,Energy,data,smoother=voidsmoother):
         super(EnergyResultScalar,self).__init__(Energy,data,smoother,TRodd=False,Iodd=False,rank=0)

class EnergyResultAxialV(EnergyResult):
    def __init__(self,Energy,data,smoother=voidsmoother):
         super(EnergyResultAxialV,self).__init__(Energy,data,smoother,TRodd=True,Iodd=False,rank=1)

class EnergyResultPolarV(EnergyResult):
    def __init__(self,Energy,data,smoother=voidsmoother):
         super(EnergyResultpolarV,self).__init__(Energy,data,smoother,TRodd=False,Iodd=True,rank=1)

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
            component=component.lower()
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

