import numpy as np
from scipy import constants
from collections import defaultdict
from .__utility import  warning, TAU_UNIT, alpha_A,beta_A
from .__tetrahedron import weights_parallelepiped  as weights_tetra  
from . import __result as result
from . import __formulas_nonabelian as frml
from .__formula import FormulaProduct
from scipy.constants import Boltzmann, elementary_charge, hbar, electron_mass, physical_constants, angstrom
from math import ceil
bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom
fac_ahc = -1e8 * elementary_charge ** 2 / hbar

degen_thresh=1e-5

def AHC(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    fac_ahc  = -1.0e8*elementary_charge**2/hbar
    return Omega_tot(data_K,Efermi,kpart=kpart,tetra=tetra,degen_thresh=degen_thresh)*fac_ahc

def cumdos(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    return iterate_kpart(frml.Identity,data_K,Efermi,kpart,tetra,degen_thresh=degen_thresh)*data_K.cell_volume

def dos(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    return iterate_kpart(frml.Identity,data_K,Efermi,kpart,tetra,degen_thresh=degen_thresh,fder=1)*data_K.cell_volume

def berry_dipole(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    res = iterate_kpart(frml.derOmega,data_K,Efermi,kpart,tetra,degen_thresh=degen_thresh)
    res.data= np.swapaxes(res.data,1,2)  # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    return res

def Hplus_der(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    res = iterate_kpart(frml.derHplus,data_K,Efermi,kpart,tetra,degen_thresh=degen_thresh)
    res.data= np.swapaxes(res.data,1,2)  # swap axes to be consistent with the eq. (30) of DOI:10.1038/s41524-021-00498-5
    return res

def tensor_K(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    Hp = Hplus_der(data_K,Efermi,kpart=kpart,tetra=tetra,degen_thresh=-1).data
    D = berry_dipole(data_K,Efermi,kpart=kpart,tetra=tetra,degen_thresh=-1).data
    tensor_K = - elementary_charge**2/(2*hbar)*(Hp - 2*Efermi[:,None,None]*D  )
    return result.EnergyResult(Efermi,tensor_K,TRodd=False,Iodd=True)

def Morb(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    fac_morb =  -eV_au/bohr**2
    return fac_morb*(iterate_kpart(frml.Hplusminus,data_K,Efermi,kpart,tetra,degen_thresh=degen_thresh)
            - 2*Omega_tot(data_K,Efermi,kpart,tetra,degen_thresh=degen_thresh).mul_array(Efermi) )*data_K.cell_volume
#    return fac_morb*data_K.cell_volume*iterate_kpart(frml.Hplusminus,data_K,Efermi,kpart,tetra,degen_thresh=degen_thresh)
  #  return fac_morb*data_K.cell_volume*-2*Omega_tot(data_K,Efermi,kpart,tetra,degen_thresh=degen_thresh).mul_array(Efermi)

def Omega_tot(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    return iterate_kpart(frml.Omega,data_K,Efermi,kpart,tetra,degen_thresh=degen_thresh)

factor_ohmic=(elementary_charge/Ang_SI/hbar**2  # first, transform to SI, not forgeting hbar in velocities - now in  1/(kg*m^3)
                 *elementary_charge**2*TAU_UNIT  # multiply by a dimensional factor - now in A^2*s^2/(kg*m^3*tau_unit) = S/(m*tau_unit)
                   * 1e-2  ) # now in  S/(cm*tau_unit)

def ohmic(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    return iterate_kpart(frml.InverseMass,data_K,Efermi,kpart,tetra,degen_thresh=degen_thresh)*factor_ohmic

def ohmic_fsurf(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    def _formula(datak,op,ed):
        _velocity=  frml.Velocity(datak,op,ed)
        return  FormulaProduct ( [_velocity,_velocity], name='vel-vel')
    return iterate_kpart(_formula,data_K,Efermi,kpart,tetra,fder=1,degen_thresh=degen_thresh)*factor_ohmic


# _general yields integral(V*V*V'*(-f0')) in units eV^2*Ang
# we want in S/(cm*T)/tau_unit^2
# S/T=A^3*s^5/(kg^2*m^2))
factor_Hall_classic=elementary_charge**2*Ang_SI/hbar**3  # first, transform to SI, not forgeting hbar in velocities - now in  m/(J*s^3)
factor_Hall_classic*=elementary_charge**3/hbar*TAU_UNIT**2  # multiply by a dimensional factor - now in A^3*s^5*cm/(J^2*tau_unit^2) = S/(T*m*tau_unit^2)
factor_Hall_classic*=1e-2   #  finally transform to S/(T*cm*tau_unit^2)


def Hall_classic(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    def _formula(datak,op,ed):
        _velocity =  frml.Velocity   (datak,op,ed)
        _mass     =  frml.InverseMass(datak,op,ed)
        return  FormulaProduct ( [_velocity,_mass,_velocity], name='vel-mass-vel')
    res =  iterate_kpart(_formula,data_K,Efermi,kpart,tetra,fder=1,degen_thresh=degen_thresh)*factor_Hall_classic
    res.data=res.data[:,:,:,beta_A,alpha_A]-res.data[:,:,:,alpha_A,beta_A]
    res.data=-0.5*(res.data[:,alpha_A,beta_A,:]-res.data[:,beta_A,alpha_A,:])
    res.rank-=2
    return res


def Hall_classic_sea(data_K,Efermi,kpart=None,tetra=False,degen_thresh=-1):
    def _formula(datak,op,ed):
        _mass     =  frml.InverseMass(datak,op,ed)
        return  FormulaProduct ( [_mass,_mass], name='mass-mass')
    res =  iterate_kpart(_formula,data_K,Efermi,kpart,tetra,fder=0,degen_thresh=degen_thresh)*(factor_Hall_classic *-1)
    res.data=res.data.swapaxes(2,3)
    res.data=res.data[:,:,:,beta_A,alpha_A]-res.data[:,:,:,alpha_A,beta_A]
    res.data=-0.5*(res.data[:,alpha_A,beta_A,:]-res.data[:,beta_A,alpha_A,:])
    res.rank-=2
    return res



##################################
### The private part goes here  ##
##################################

def iterate_kpart(formula_fun,data_K,Efermi,kpart=None,tetra=False,fder=0,degen_thresh=1e-5,**parameters): 
    """ formula_fun should be callable returning a Formula object 
    with first three parameters as data_K,op,ed, and the rest
    and the rest will be arbitrary keyword arguments.
    fder derivative of fermi distribution . 0: fermi-sea, 1: fermi-surface 2: f''
    """

    if kpart is None : 
        kpart=data_K.NKFFT_tot
    n=data_K.NKFFT_tot//kpart
    if n>0 :
        kpart += ceil( (data_K.NKFFT_tot % kpart)/n )
    borders=list(range(0,data_K.NKFFT_tot,kpart))+[data_K.NKFFT_tot]
#    print ("processing the {} FFT grid points in {} portions of {},  last portion is {}".format(
#               data_K.NKFFT_tot,len(borders)-1,kpart,data_K.NKFFT_tot%kpart))
    f0=formula_fun(data_K,0,0,**parameters)  # just to get the basic properties
    res= sum (
            FermiOcean( formula_fun(data_K,op,ed,**parameters),
                    data_K,op,ed,
                    Efermi,
                    ndim=f0.ndim,
                    tetra=tetra,
                    fder=fder ,
                    degen_thresh=degen_thresh ) ()
                   for op,ed in zip(borders,borders[1:]) 
               ) / (data_K.NKFFT_tot * data_K.cell_volume)

    return result.EnergyResult(Efermi,res, TRodd=f0.TRodd, Iodd=f0.Iodd )


## Note - there is probalby no point to create an object and use it only once
## it is done for visual separation of "preparation" and "evaluation"


class  FermiOcean():

    def __init__(self , formula , data_K, op, ed, Efermi, ndim, tetra,fder,degen_thresh):

        Emin=Efermi[ 0]
        Emax=Efermi[-1]
        #print(Efermi)
        self.Efermi=Efermi
        self.fder=fder
        self.tetra=tetra
        self.nk=ed-op
        # get a list [{(ib1,ib2):W} for ik in op:ed]  
        if self.tetra:
            self.weights=data_K.tetraWeights.weights_all_band_groups(Efermi,op=op,ed=ed,der=self.fder,degen_thresh=degen_thresh)   # here W is array of shape Efermi
        else:
            self.weights=data_K.get_bands_in_range_groups(Emin,Emax,op,ed,degen_thresh=degen_thresh,sea=(self.fder==0)) # here W is energy
       # print(self.weights)
        self.__evaluate_traces(formula, self.weights, ndim)

    def __evaluate_traces(self,formula,bands, ndim):
        """formula  - TraceFormula to evaluate 
           bands = a list of lists of k-points for every 
        """
        self.shape = (3,)*ndim
        lambdadic= lambda: np.zeros(((3, ) * ndim), dtype=float)
        self.values = [defaultdict(lambdadic ) for ik in range(self.nk)]
        for ik,bnd in enumerate(bands):
            for n in bnd :
                self.values[ik][n] = formula(ik,ib_in_start=n[0],ib_in_end=n[1],trace=True)#,Emax=bands[ik][n] )

    def __call__(self) :
        result = np.zeros(self.Efermi.shape + self.shape )
        for ik,weights in enumerate(self.weights):
            resk = np.zeros(self.Efermi.shape + self.shape )
            values = self.values[ik]
            if self.tetra:
                for n,w in weights.items():
                    resk+=np.einsum( "e,...->e...",w,values[n] )
            else:
                resk = np.zeros(self.Efermi.shape + self.shape )
                if self.fder==0:
                #    print( sorted(weights.items()))
                    for n,E in sorted(weights.items()):
                    #for n,E in weights.items():
                        #print('n,E',n,E)
                        #print('values',values)
                    #   resk[self.Efermi >= E] = values[n]
                        resk[self.Efermi >= E] += values[n]
                else :
                    raise NotImplementedError("fermi-surface properties in fermi-ocean are implemented only with tetrahedron method so far")
            result += resk
        return result

