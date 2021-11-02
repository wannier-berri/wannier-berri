import numpy as np
from .__utility import  alpha_A,beta_A
import abc 
from scipy.constants import Boltzmann,elementary_charge
from .formula import Formula_ln, Matrix_ln , Matrix_GenDer_ln
from .data_K import _Dcov
#####################################################
#####################################################


class Identity(Formula_ln):
    def __init__(self):
        self.ndim=0
        self.TRodd=False
        self.Iodd=False

    def nn(self,ik,inn,out):
        return np.eye(len(inn))

    def ln(self,ik,inn,out):
        return np.zeros((len(out),len(inn)))



class Eavln(Matrix_ln):
    """ be careful : this is not a covariant matrix"""
    def __init__(self,data_K):
        super().__init__(
                       0.5* (data_K.E_K[:,:,None]+data_K.E_K[:,None,:])
                                )
        self.ndim=0
        self.TRodd=False
        self.Iodd=False


class DEinv_ln(Matrix_ln):

    def __init__(self,data_K):
        super(DEinv_ln,self).__init__(data_K.dEig_inv)

    def nn(self,ik,inn,out):
        raise NotImplementedError("dEinv_ln should not be called within inner states")



class DerDcov(_Dcov):

    def __init__(self,data_K):
        self.W=data_K.covariant('Ham',commader = 2)
        self.V=data_K.covariant('Ham',gender = 1)
        self.D=data_K.Dcov
        self.dEinv=DEinv_ln(data_K)

    def ln(self,ik,inn,out):
        summ=self.W.ln(ik,inn,out)
        tmp =  np.einsum( "lpb,pnd->lnbd" , self.V.ll(ik,inn,out) , self.D.ln(ik,inn,out) )
        summ+= tmp+tmp.swapaxes(2,3)
        tmp =  -np.einsum( "lmb,mnd->lnbd" , self.D.ln(ik,inn,out) , self.V.nn(ik,inn,out) )
        summ+= tmp+tmp.swapaxes(2,3)
        summ*=-self.dEinv.ln(ik,inn,out)[:,:,None,None]
        return summ


class InvMass(Matrix_GenDer_ln):
    r""" :math:`\overline{V}^{b:d}`"""
    def __init__(self,data_K):
        super().__init__(data_K.covariant('Ham',commader=1),data_K.covariant('Ham',commader=2),data_K.Dcov)
        self.TRodd=False
        self.Iodd=False

class DerWln(Matrix_GenDer_ln):
    r""" :math:`\overline{W}^{bc:d}`"""
    def __init__(self,data_K):
        super().__init__(data_K.covariant('Ham',2),data_K.covariant('Ham',3),data_K.Dcov)
        self.TRodd=False
        self.Iodd=False



##################################
###   Third derivative of  E  ####
##################################


class Der3E(Formula_ln):

    def __init__(self,data_K,**parameters):
        super().__init__(data_K,**parameters)
        self.V=data_K.covariant('Ham',commader=1)
        self.D=data_K.Dcov
        self.dV=InvMass(data_K)
        self.dD=DerDcov(data_K)
        self.dW=DerWln(data_K)
        self.ndim=3
        self.Iodd=True
        self.TRodd=True

    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3,3,3),dtype=complex )
        summ += 1 * self.dW.nn(ik,inn,out)
        summ += 1 * np.einsum("mlac,lnb->mnabc",self.dV.nl(ik,inn,out),self.D.ln(ik,inn,out) )
        summ += 1 * np.einsum("mla,lnbc->mnabc",self.V.nl(ik,inn,out),self.dD.ln(ik,inn,out) )
        summ+=  -1 * np.einsum("mlbc,lna->mnabc",self.dD.nl(ik,inn,out),self.V.ln(ik,inn,out) )
        summ+=  -1 * np.einsum("mlb,lnac->mnabc",self.D.nl(ik,inn,out),self.dV.ln(ik,inn,out) )

        # TODO: alternatively: add factor 0.5 to first term, remove 4th and 5th, and ad line below. 
        # Should give the same, I think, but needs to be tested
        # summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self,ik,inn,out):
        raise NotImplementedError()


#############################
###   Berry curvature    ####
#############################

class Omega(Formula_ln):

    def __init__(self,data_K,**parameters):
        super().__init__(data_K,**parameters)
        self.D=data_K.Dcov

        if self.external_terms:
            self.A=data_K.covariant('AA')
            self.O=data_K.covariant('OO')

        self.ndim=1
        self.Iodd=False
        self.TRodd=True

    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3),dtype=complex )

        if self.internal_terms:
            summ+= -1j*np.einsum("mlc,lnc->mnc",self.D.nl(ik,inn,out)[:,:,alpha_A],self.D.ln(ik,inn,out)[:,:,beta_A])

        if self.external_terms:
            summ += 0.5 * self.O.nn(ik,inn,out)
            summ +=  -1 * np.einsum("mlc,lnc->mnc",self.D.nl(ik,inn,out)[:,:,alpha_A],self.A.ln(ik,inn,out)[:,:,beta_A])
            summ +=  +1 * np.einsum("mlc,lnc->mnc",self.D.nl(ik,inn,out)[:,:,beta_A] ,self.A.ln(ik,inn,out)[:,:,alpha_A])
            summ+=  -1j * np.einsum("mlc,lnc->mnc",self.A.nn(ik,inn,out)[:,:,alpha_A],self.A.nn(ik,inn,out)[:,:,beta_A])

        summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self,ik,inn,out):
        raise NotImplementedError()


#############################
###   derivative of      ####
###   Berry curvature    ####
#############################


class DerOmega(Formula_ln):

    def __init__(self,data_K,**parameters):
        super().__init__(data_K,**parameters)
        self.dD = DerDcov(data_K)
        self.D  = data_K.Dcov

        if self.external_terms:
            self.A  = data_K.covariant('AA')
            self.dA = data_K.covariant('AA',gender=1)
            self.dO  = data_K.covariant('OO',gender=1)
        self.ndim=2
        self.Iodd=True
        self.TRodd=False

    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3,3),dtype=complex )
        if self.external_terms:
            summ += 0.5 * self.dO.nn(ik,inn,out)

        for s,a,b in (+1,alpha_A,beta_A),(-1,beta_A,alpha_A):
            if self.internal_terms:
                summ+= -1j*s*np.einsum("mlc,lncd->mncd",self.D.nl(ik,inn,out)[:,:,a],self.dD.ln(ik,inn,out)[:,:,b])
                pass

            if self.external_terms:
                summ +=  -1 *s* np.einsum("mlc,lncd->mncd",self.D.nl (ik,inn,out)[:,:,a]   , self.dA.ln(ik,inn,out)[:,:,b,:])
                summ +=  -1 *s* np.einsum("mlcd,lnc->mncd",self.dD.nl(ik,inn,out)[:,:,a,:] , self.A.ln (ik,inn,out)[:,:,b  ])
                summ+=  -1j *s* np.einsum("mlc,lncd->mncd",self.A.nn (ik,inn,out)[:,:,a]   , self.dA.nn(ik,inn,out)[:,:,b,:])
                pass

        summ+=summ.swapaxes(0,1).conj()
        return summ


    def ln(self,ik,inn,out):
        raise NotImplementedError()


#############################
###   orbital moment     ####
#############################

class Velocity(Matrix_ln):
    def __init__(self,data_K):
        v =  data_K.covariant('Ham',gender = 1)
        self.__dict__.update(v.__dict__)

class Spin(Matrix_ln):
    def __init__(self,data_K):
        s =  data_K.covariant('SS')
        self.__dict__.update(s.__dict__)

class DerSpin(Matrix_GenDer_ln):
    def __init__(self,data_K):
        s =  data_K.covariant('SS',gender=1)
        self.__dict__.update(s.__dict__)


class Morb_H(Formula_ln):
    def __init__(self,data_K,**parameters):
        r"""  :math:`\varepcilon_{abc} \langle \partial_a u | H | \partial_b \rangle` """
        super().__init__(data_K,**parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.B = data_K.covariant('BB')
            self.C = data_K.covariant('CC')
        self.D = data_K.Dcov
        self.E = data_K.E_K
        self.ndim=1
        self.Iodd=False
        self.TRodd=True


    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3),dtype=complex )

        if self.internal_terms:
            summ+= -1j*np.einsum(  "mlc,lnc->mnc",
                             self.D.nl(ik,inn,out)[:,:,alpha_A]*self.E[ik][out][None,:,None],
                             self.D.ln(ik,inn,out)[:,:,beta_A]  )

        if self.external_terms:
            summ +=  0.5  * self.C.nn(ik,inn,out)
            summ +=  -1  * np.einsum(  "mlc,lnc->mnc",self.D.nl(ik,inn,out)[:,:,alpha_A],self.B.ln(ik,inn,out)[:,:,beta_A])
            summ +=  +1  * np.einsum(  "mlc,lnc->mnc",self.D.nl(ik,inn,out)[:,:,beta_A] ,self.B.ln(ik,inn,out)[:,:,alpha_A])
            summ +=  -1j * np.einsum("mlc,lnc->mnc",
                                    self.A.nn(ik,inn,out)[:,:,alpha_A]*self.E[ik][inn][None,:,None] ,
                                    self.A.nn(ik,inn,out)[:,:,beta_A]  )
        summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self,ik,inn,out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


class Morb_Hpm(Formula_ln):
    def __init__(self,data_K,sign=+1,**parameters):
        r""" Morb_H  +- (En+Em)/2 * Omega """
        super().__init__(data_K,**parameters)
        self.H = Morb_H(data_K,**parameters)
        self.sign = sign
        if self.sign!=0:
            self.O = Omega (data_K,**parameters)
            self.Eav = Eavln ( data_K )
        self.ndim=1
        self.Iodd=False
        self.TRodd=True

    @property
    def additive(self):
        return False

    def nn(self,ik,inn,out):
        res = self.H.nn(ik,inn,out)
        if self.sign!=0:
            res+= self.sign*self.Eav.nn(ik,inn,out)[:,:,None]*self.O.nn(ik,inn,out)
        return res

    def ln(self,ik,inn,out):
        raise NotImplementedError()



#############################
###   derivative of      ####
###   orbital moment     ####
#############################



class DerMorb(Formula_ln):
    def __init__(self,data_K,**parameters):
        super().__init__(data_K,**parameters)
        self.dD = DerDcov(data_K)
        self.D  = data_K.Dcov
        self.V = data_K.covariant('Ham',commader=1)
        self.E = data_K.E_K
        self.dO  = DerOmega(data_K,**parameters)
        self.Omega = Omega(data_K,**parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA',gender = 1)
            self.B = data_K.covariant('BB')
            self.dB = data_K.covariant('BB',gender = 1)
            self.dH  = data_K.covariant('CC',gender = 1)
        self.ndim=2
        self.Iodd=True
        self.TRodd=False

    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3,3),dtype=complex )
        if self.internal_terms:
            summ += -2j * np.einsum("mpc,pld,lnc->mncd",self.D.nl(ik,inn,out)[:,:,alpha_A],self.V.ll(ik,inn,out),self.D.ln(ik,inn,out)[:,:,beta_A] )
            for s,a,b in (+1,alpha_A,beta_A),(-1,beta_A,alpha_A):
                summ+=  -2j *s* np.einsum("mlc,lncd->mncd",self.D.nl(ik,inn,out)[:,:,a],
                    self.E[ik][out][:,None,None,None]*self.dD.ln(ik,inn,out)[:,:,b])
        if self.external_terms:
            summ += 1 * self.dH.nn(ik,inn,out)
            summ += -2j * np.einsum("mpc,pld,lnc->mncd",self.A.nn(ik,inn,out)[:,:,alpha_A],self.V.nn(ik,inn,out),self.A.nn(ik,inn,out)[:,:,beta_A] )
            for s,a,b in (+1,alpha_A,beta_A),(-1,beta_A,alpha_A):
                summ+=  -2j *s* np.einsum("mlc,lncd->mncd",self.A.nn(ik,inn,out)[:,:,a]*self.E[ik][inn][None,:,None],self.dA.nn(ik,inn,out)[:,:,b,:])
                summ +=  -2 *s* np.einsum("mlc,lncd->mncd",self.D.nl (ik,inn,out)[:,:,a], self.dB.ln(ik,inn,out)[:,:,b,:])
                summ +=  -2 *s* np.einsum("mlc,lncd->mncd",(self.B.ln(ik,inn,out)[:,:,a]).transpose(1,0,2).conj() , self.dD.ln (ik,inn,out)[:,:,b,:])

        summ += 1 * np.einsum("mlc,lnd->mncd",self.Omega.nn(ik,inn,out),self.V.nn(ik,inn,out) )
        summ += 1 * self.E[ik][inn][:,None,None,None]*self.dO.nn(ik,inn,out)

        # Stepan: Shopuldn't we use the line below? 
        # TODO: check this formula
        #summ+=summ.swapaxes(0,1).conj()
        return summ


    def ln(self,ik,inn,out):
        raise NotImplementedError()


    @property
    def additive(self):
        return False

