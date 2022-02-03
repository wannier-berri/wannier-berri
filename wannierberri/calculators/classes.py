from wannierberri import fermiocean
from wannierberri import __result as result
from wannierberri.__tabulate import TABresult
import numpy as np
import abc,functools
from wannierberri.__kubo import Gaussian, Lorentzian


class Calculator():

    def __init__(self,degen_thresh=1e-4,degen_Kramers=False,save_mode="bin+txt"):
        self.degen_thresh = degen_thresh
        self.degen_Kramers = degen_Kramers
        self.save_mode = save_mode

##################################################
######                                     #######
######         integration (Efermi-only)   #######
######                                     #######
##################################################

class StaticCalculator(Calculator):

    def __init__(self,Efermi,tetra=False,kwargs_formula={},**kwargs):
        self.Efermi = Efermi
        self.tetra  = tetra
        self.kwargs_formula = kwargs_formula
        assert hasattr(self,'factor')
        assert hasattr(self,'fder')
        assert hasattr(self,'Formula') , "Formula should be a Formula or a list of Formula (in the latter case a product is taken"
        super().__init__(**kwargs)

    def  __call__(self,data_K):
        res = fermiocean.FermiOcean(self.Formula(data_K,**self.kwargs_formula),data_K,self.Efermi,self.tetra,fder=self.fder,degen_thresh=self.degen_thresh,degen_Kramers=self.degen_Kramers)()
        res*= self.factor
        res.set_save_mode(self.save_mode)
        return res


##################################################
######                                     #######
######      integration (Efermi-omega)     #######
######                                     #######
##################################################

def FermiDirac(E, mu, kBT):
    "here E is a number, mu is an array"
    if kBT == 0:
        return 1.0*(E <= mu)
    else:
        res = np.zeros_like(mu)
        res[mu>E+30*kBT] = 1.0
        res[mu<E-30*kBT] = 0.0
        sel = abs(mu-E)<=30*kBT
        res[sel]=1.0/(np.exp((E-mu[sel])/kBT) + 1)
        return res



class DynamicCalculator(Calculator,abc.ABC):

    def __init__(self,Efermi=None,omega=None,  kBT=0, smr_fixed_width=0.1, smr_type='Lorentzian', adpt_smr=False,
                adpt_smr_fac=np.sqrt(2), adpt_smr_max=0.1, adpt_smr_min=1e-15 , sep_sym_asym = False, **kwargs):

        for k,v in locals().items(): # is it safe to do so?
            if k not in ['self','kwargs']:
                vars(self)[k]=v
        super().__init__(**kwargs)

        self.formula_kwargs = {}
        self.Formula  = None 
        self.final_factor = 1.
        self.dtype = complex
        self.EFmin=self.Efermi.min()
        self.EFmax=self.Efermi.max()
        self.omegamin=self.omega.min()
        self.omegamax=self.omega.max()
        
        if self.smr_type == 'Lorentzian':
            self.smear = functools.partial(Lorentzian,width = self.smr_fixed_width)
        elif self.smr_type == 'Gaussian':
            self.smear = functools.partial(Gaussian,width = self.smr_fixed_width,adpt_smr = False)
        else:
            raise ValueError("Invalid smearing type {self.smr_type}")
        self.FermiDirac = functools.partial(FermiDirac,mu = self.Efermi,kBT = self.kBT) 
        


    @abc.abstractmethod
    def energy_factor(self,E1,E2):
        pass

    def nonzero(self,E1,E2):
        emin = self.EFmin-30*self.kBT
        if E1<emin and E2<emin:
            return False
        emax = self.EFmax+30*self.kBT
        if E1>emax and E2>emax:
            return False
        return True


    def __call__(self,data_K):
        formula  = self.Formula(data_K,**self.formula_kwargs)
        restot_shape = (len(self.Efermi),len(self.omega))+(3,)*formula.ndim
        restot  = np.zeros(restot_shape,self.dtype)
    
        for ik in range(data_K.nk):
            degen_groups = data_K.get_bands_in_range_groups_ik(ik,-np.Inf,np.Inf,degen_thresh=self.degen_thresh,degen_Kramers=self.degen_Kramers)
            #now find needed pairs:
            # as a dictionary {((ibm1,ibm2),(ibn1,ibn2)):(Em,En)} 
            degen_group_pairs= { (ibm,ibn):(Em,En) 
                                     for ibm,Em in degen_groups.items()
                                         for ibn,En in degen_groups.items()
                                             if self.nonzero(Em,En) }
#        matrix_elements = {(inn1,inn2):self.formula.trace_ln(ik,inn1,inn2) for (inn1,inn2) in self.energy_factor().keys()}
            for pair,EE in degen_group_pairs.items():
                factor = self.energy_factor(EE[0],EE[1])
                matrix_element = formula.trace_ln(ik,np.arange(*pair[0]),np.arange(*pair[1]))
#                restot+=np.einsum( "ew,...->ew...",factor,matrix_element )
                restot+=factor.reshape(factor.shape+(1,)*formula.ndim)*matrix_element[None,None]
        restot *= self.final_factor / (data_K.nk*data_K.cell_volume)
        return result.EnergyResult([self.Efermi,self.omega],restot, TRodd=formula.TRodd, Iodd=formula.Iodd, TRtrans = formula.TRtrans )



##################################################
######                                     #######
######      tabulating                     #######
######                                     #######
##################################################


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

