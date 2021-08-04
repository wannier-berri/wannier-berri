import numpy as np
from .__utility import  alpha_A,beta_A
import abc 
from scipy.constants import Boltzmann,elementary_charge
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


class Eavln(Matrix_ln):
    def __init__(self,data_K):
        super(Eavln,self).__init__(
                       0.5* (data_K.E_K[:,:,None]+data_K.E_K[:,None,:])
                                )
        self.ndim=0
        self.TRodd=False
        self.Iodd=False


class Vln(Matrix_ln):
    def __init__(self,data_K):
        super(Vln,self).__init__(data_K.V_H)
        self.TRodd = True
        self.Iodd  = True

class Aln(Matrix_ln):
    def __init__(self,data_K):
        super(Aln,self).__init__(data_K.A_Hbar)

class Sln(Matrix_ln):
    def __init__(self,data_K):
        super(Sln,self).__init__(data_K.S_H)
        self.TRodd=True
        self.Iodd=False

class dSln(Matrix_ln):
    def __init__(self,data_K):
        super(dSln,self).__init__(data_K.delS_H)

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


class DerSln(Matrix_GenDer_ln):
    r""" :math:`\overline{S}^{b:d}`"""
    def __init__(self,data_K):
        super(DerSln,self).__init__(Sln(data_K),dSln(data_K),Dln(data_K))
        self.TRodd=True
        self.Iodd=False


class InvMass(Matrix_GenDer_ln):
    r""" :math:`\overline{V}^{b:d}`"""
    def __init__(self,data_K):
        super(InvMass,self).__init__(Vln(data_K),Wln(data_K),Dln(data_K))
        self.TRodd=False
        self.Iodd=False

class DerWln(Matrix_GenDer_ln):
    r""" :math:`\overline{W}^{bc:d}`"""
    def __init__(self,data_K):
        super(DerWln,self).__init__(Wln(data_K),del3E_H(data_K),Dln(data_K))
        self.TRodd=False
        self.Iodd=False


class del3E_H(Matrix_ln):
    def __init__(self,data_K):
        super(del3E_H,self).__init__(data_K.del3E_H)

##################################
###   Third derivative of  E  ####
##################################


class Der3E(Formula_ln):

    def __init__(self,data_K,**parameters):
        super(Der3E,self).__init__(data_K,**parameters)
        self.V=Vln(data_K)
        self.D=Dln(data_K)
        self.dV=InvMass(data_K)
        self.dD=DerDln(data_K)
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

        #summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self,ik,inn,out):
        raise NotImplementedError()


#############################
###   Berry curvature    ####
#############################

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

     #   if self.internal_terms:
        summ+= -1j*np.einsum("mlc,lnc->mnc",self.D.nl(ik,inn,out)[:,:,alpha_A],self.D.ln(ik,inn,out)[:,:,beta_A])

     #   if self.external_terms:
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


#############################
###   orbital moment     ####
#############################


class Bln(Matrix_ln):
    def __init__(self,data_K):
        super(Bln,self).__init__(data_K.B_Hbar)


class Cln(Matrix_ln):
    def __init__(self,data_K):
        super(Cln,self).__init__(data_K.Morb_Hbar)

class Morb_H(Formula_ln):
    def __init__(self,data_K,**parameters):
        r"""  :math:`\varepcilon_{abc} \langle \partial_a u | H | \partial_b \rangle` """
        super(Morb_H,self).__init__(data_K,**parameters)
        if self.external_terms:
            self.A = Aln(data_K)
            self.B = Bln(data_K)
            self.C = Cln(data_K)
        self.D = Dln(data_K)
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
            summ += 0.5  * self.C.nn(ik,inn,out)
            summ +=  -1  * np.einsum(  "mlc,lnc->mnc",self.D.nl(ik,inn,out)[:,:,alpha_A],self.B.ln(ik,inn,out)[:,:,beta_A])
            summ +=  +1  * np.einsum(  "mlc,lnc->mnc",self.D.nl(ik,inn,out)[:,:,beta_A] ,self.B.ln(ik,inn,out)[:,:,alpha_A])
            summ +=  -1j * np.einsum("mlc,lnc->mnc",
                                    self.A.nn(ik,inn,out)[:,:,alpha_A]*self.E[ik][inn][None,:,None] ,
                                    self.A.nn(ik,inn,out)[:,:,beta_A]  
                                    )
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
        super(Morb_Hpm,self).__init__(data_K,**parameters)
        self.H = Morb_H(data_K,**parameters)
        self.O = Omega (data_K,**parameters)
        self.Eav = Eavln ( data_K )
        self.s = sign
        self.ndim=1
        self.Iodd=False
        self.TRodd=True

    @property
    def additive(self):
        return False

    def nn(self,ik,inn,out):
        return  self.H.nn(ik,inn,out)+self.s*self.Eav.nn(ik,inn,out)[:,:,None]*self.O.nn(ik,inn,out)

    def ln(self,ik,inn,out):
        raise NotImplementedError()



#############################
###   derivative of      ####
###   orbital moment     ####
#############################

class dHln(Matrix_ln):
    def __init__(self,data_K):
        super(dHln,self).__init__(data_K.Morb_Hbar_der)

class DerMorb_Hbar_ln(Matrix_GenDer_ln):
    r""" :math:`\overline{H}^{bc:d}`"""
    def __init__(self,data_K):
        super(DerMorb_Hbar_ln,self).__init__(Cln(data_K),dHln(data_K),Dln(data_K))

class dBln(Matrix_ln):
    def __init__(self,data_K):
        super(dBln,self).__init__(data_K.B_Hbar_der)

class DerB_Hbar_ln(Matrix_GenDer_ln):
    r""" :math:`\overline{B}^{b:d}`"""
    def __init__(self,data_K):
        super(DerB_Hbar_ln,self).__init__(Bln(data_K),dBln(data_K),Dln(data_K))


class DerMorb(Formula_ln):
    def __init__(self,data_K,**parameters):
        super(DerMorb,self).__init__(data_K,**parameters)
        self.dD = DerDln(data_K)
        self.D  = Dln(data_K)
        self.V = Vln(data_K)
        self.A = Aln(data_K)
        self.dA = DerA_Hbar_ln(data_K)
        self.B = Bln(data_K)
        self.E = data_K.E_K
        self.dB = DerB_Hbar_ln(data_K)
      #  self.dO  = DerOmega_Hbar_ln(data_K)
        self.dO  = DerOmega(data_K,**parameters)
        self.dH  = DerMorb_Hbar_ln(data_K)
      #  self.Omega = Oln(data_K)
        self.Omega = Omega(data_K)
        self.ndim=2
        self.Iodd=True
        self.TRodd=False

    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3,3),dtype=complex )
        summ += 1 * self.dH.nn(ik,inn,out)
        summ += 1 * self.E[ik][inn][:,None,None,None]*self.dO.nn(ik,inn,out)
        summ += 1 * np.einsum("mlc,lnd->mncd",self.Omega.nn(ik,inn,out),self.V.nn(ik,inn,out) )
        for s,a,b in (+1,alpha_A,beta_A),(-1,beta_A,alpha_A):
            summ += -1j *s* np.einsum("mpc,pld,lnc->mncd",self.A.nn(ik,inn,out)[:,:,a],self.V.nn(ik,inn,out),self.A.nn(ik,inn,out)[:,:,b] )
            summ += -1j *s* np.einsum("mpc,pld,lnc->mncd",self.D.nl(ik,inn,out)[:,:,a],self.V.ll(ik,inn,out),self.D.ln(ik,inn,out)[:,:,b] )
        
            summ+=  -2j *s* np.einsum("mlc,lncd->mncd",self.A.nn(ik,inn,out)[:,:,a]*self.E[ik][inn][None,:,None],self.dA.nn(ik,inn,out)[:,:,b,:])
          
            summ +=  -2 *s* np.einsum("mlc,lncd->mncd",self.D.nl (ik,inn,out)[:,:,a], self.dB.ln(ik,inn,out)[:,:,b,:])
            summ +=  -2 *s* np.einsum("lmc,lncd->mncd",(self.B.ln(ik,inn,out)[:,:,a]).conj() , self.dD.ln (ik,inn,out)[:,:,b,:])
        
            summ+=  -2j *s* np.einsum("mlc,lncd->mncd",self.D.nl(ik,inn,out)[:,:,a],
                    self.E[ik][out][:,None,None,None]*self.dD.ln(ik,inn,out)[:,:,b])
            

        #summ+=summ.swapaxes(0,1).conj()
        return summ


    def ln(self,ik,inn,out):
        raise NotImplementedError()
