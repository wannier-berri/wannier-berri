from . import fermiocean
from . import covariant_formulak as frml
from . import __result as result
from .__tabulate import TABresult
import numpy as np

class Calculator():
    def __init__(self,degen_thresh=1e-4,degen_Kramers=False,save_mode="bin+txt"):
        self.degen_thresh = degen_thresh
        self.degen_Kramers = degen_Kramers
        self.save_mode = save_mode

class AHC(Calculator):
    def __init__(self,Efermi,tetra=False,kwargs_formula={},**kwargs):
        self.Efermi = Efermi
        self.tetra  = tetra
        self.kwargs_formula = kwargs_formula
        super().__init__(**kwargs)
    
    def __call__(self,data_K):
        res =  fermiocean.AHC(data_K,self.Efermi,tetra=self.tetra,degen_thresh=self.degen_thresh,degen_Kramers=self.degen_Kramers,**self.kwargs_formula)
        res.set_save_mode(self.save_mode)
        return res


class  Tabulator(Calculator):

    def __init__(self ,  Formula, kwargs_formula={},**kwargs):
        self.Formula = Formula
        self.ibands = None
        self.kwargs_formula = kwargs_formula
        super().__init__(**kwargs)



    def __call__(self,data_K):
        formula = self.Formula(data_K,**self.kwargs_formula)
        nk=data_K.nk
        NB=data_K.num_wann
        ibands = self.ibands
        if ibands is None:
            ibands = np.arange(NB)
        band_groups=data_K.get_bands_in_range_groups(-np.Inf,np.Inf,degen_thresh=self.degen_thresh,degen_Kramers=self.degen_Kramers,sea=False)
        # bands_groups  is a digtionary (ib1,ib2):E
        # now select only the needed groups
        band_groups = [  [ n    for n in groups.keys() if np.any(  (ibands>=n[0])*(ibands<n[1]) )  ]
              for groups in band_groups ]    # select only the needed groups
        group = [[] for ik in range(nk)]
        for ik in range(nk):
            for ib in ibands:
                for n in band_groups[ik]:
                    if ib<n[1] and ib >=n[0]:
                        group[ik].append(n)
                        break

        rslt = np.zeros( (nk,len(ibands))+(3,)*formula.ndim )
        for ik in range(nk):
            values={}
            for n in band_groups[ik]:
                inn = np.arange(n[0],n[1])
                out = np.concatenate( (np.arange(0,n[0]), np.arange(n[1],NB) ) )
                values[n] = formula.trace(ik,inn,out)/(n[1]-n[0])
            for ib,b in enumerate(ibands): 
                rslt[ik,ib] = values[group[ik][ib]]
        return result.KBandResult(rslt,TRodd=formula.TRodd,Iodd=formula.Iodd)


class BerryCurvature(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml.Omega,**kwargs)

class Energy(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml.Eavln,**kwargs)


class TabulatorAll(Calculator):

    def __init__(self,tabulators,ibands=None):
        """ tabulators - dict 'key':tabulator
        one of them should be "Energy" """
        self.tabulators = tabulators
        if "Energy" not in self.tabulators.keys():
            raise ValueError("Energy is not included in tabulators")
        if ibands is not None:
            ibands = np.array(ibands)
        for k,v in self.tabulators.items():
            if v.ibands is None:
                v.ibands = ibands
            else:
                assert v.ibands == ibands
        
    def __call__(self,data_K):
        return TABresult( kpoints       = data_K.kpoints_all,
                      recip_lattice = data_K.system.recip_lattice,
                      results       =  {k:v(data_K) for k,v in self.tabulators.items()}
                          )

