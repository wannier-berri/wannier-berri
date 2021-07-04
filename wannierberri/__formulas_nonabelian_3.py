import numpy as np
from .__utility import  alpha_A,beta_A
import abc 

from .__formula_3 import Formula_ln, Matrix_ln , Matrix_GenDer_ln
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


class Vln(Matrix_ln):
    def __init__(self,data_K):
        super(Vln,self).__init__(data_K.V_H)
        self.TRodd = True
        self.Iodd  = True

class Aln(Matrix_ln):
    def __init__(self,data_K):
        super(Aln,self).__init__(data_K.A_Hbar)

class dAln(Matrix_ln):
    def __init__(self,data_K):
        super(dAln,self).__init__(data_K.A_Hbar_der)

class Wln(Matrix_ln):
    def __init__(self,data_K):
        super(Wln,self).__init__(data_K.del2E_H)

class Oln(Matrix_ln):
    def __init__(self,data_K):
        super(Oln,self).__init__(data_K.Omega_Hbar)

class dOln(Matrix_ln):
    def __init__(self,data_K):
        super(dOln,self).__init__(data_K.Omega_bar_der)


class Dln(Matrix_ln):

    def __init__(self,data_K):
        super(Dln,self).__init__(data_K.D_H)

    def nn(self,ik,inn,out):
        raise NotImplementedError("Dln should not be called within inner states")


class DEinv_ln(Matrix_ln):

    def __init__(self,data_K):
        super(DEinv_ln,self).__init__(data_K.dEig_inv)

    def nn(self,ik,inn,out):
        raise NotImplementedError("dEinv_ln should not be called within inner states")


class DerDln(Dln):

    def __init__(self,data_K):
        self.W=Wln(data_K)
        self.V=Vln(data_K)
        self.D=Dln(data_K)
        self.dEinv=DEinv_ln(data_K)

    def ln(self,ik,inn,out):
        summ=self.W.ln(ik,inn,out)
        tmp =  np.einsum( "lpb,pnd->lnbd" , self.V.ll(ik,inn,out) , self.D.ln(ik,inn,out) )
        summ+= tmp+tmp.swapaxes(2,3)
        tmp =  -np.einsum( "lmb,mnd->lnbd" , self.D.ln(ik,inn,out) , self.V.nn(ik,inn,out) )
        summ+= tmp+tmp.swapaxes(2,3)
        summ*=-self.dEinv.ln(ik,inn,out)[:,:,None,None]
        return summ


class DerOmega_Hbar_ln(Matrix_GenDer_ln):
    r""" :math:`\overline{\Omega}^{b:d}`"""
    def __init__(self,data_K):
        super(DerOmega_Hbar_ln,self).__init__(Oln(data_K),dOln(data_K),Dln(data_K))


class DerA_Hbar_ln(Matrix_GenDer_ln):
    r""" :math:`\overline{A}^{b:d}`"""
    def __init__(self,data_K):
        super(DerA_Hbar_ln,self).__init__(Aln(data_K),dAln(data_K),Dln(data_K))


class InvMass(Matrix_GenDer_ln):
    r""" :math:`\overline{V}^{b:d}`"""
    def __init__(self,data_K):
        super(InvMass,self).__init__(Vln(data_K),Wln(data_K),Dln(data_K))
        self.TRodd=False
        self.Iodd=False


class DerOmega(Formula_ln):

    def __init__(self,data_K,**parameters):
        super(DerOmega,self).__init__(data_K,**parameters)
        self.dD = DerDln(data_K)
        self.D  = Dln(data_K)

        print (f"derOmega evaluating: internal({self.internal_terms}) and external({self.external_terms})")

        if self.external_terms:
            self.A  = Aln(data_K)
            self.dA = DerA_Hbar_ln(data_K)
            self.dO  = DerOmega_Hbar_ln(data_K)
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



class Omega(Formula_ln):

    def __init__(self,data_K,**parameters):
        super(Omega,self).__init__(data_K,**parameters)
        self.A=Aln(data_K)
        self.V=Vln(data_K)
        self.D=Dln(data_K)
        self.O=Oln(data_K)
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


