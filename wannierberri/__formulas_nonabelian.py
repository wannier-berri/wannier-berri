import numpy as np
from .__utility import  alpha_A,beta_A
from .__formula import Formula
from functools import partial
#####################################################
#####################################################

# Here we write some functions, that take a argument Data_K object, op and ed, and return a Formula

def Identity(data_K,op,ed):
        "an identity operator (to compute DOS)"
        # first give our matrices short names
        NB= data_K.nbands
        # This is the formula to be implemented:
        formula =  Formula ( TRodd=False,Iodd=False,ndim=0,name='Identity' )
        mat = np.zeros((ed-op,NB,NB),dtype=complex)
        for i in range(NB): 
           mat[:,i,i]=1.
        formula.add_term ('mn', mat) 
        return formula


def Velocity(data_K,op,ed):
        "inverse effective mass"
        # first give our matrices short names
        V   = data_K.V_H[op:ed]
        # This is the formula to be implemented:
        formula =  Formula ( TRodd=True,Iodd=True,ndim=1,name="vbelocity" )
        formula.add_term ( 'mn'   ,  V ,  1. )
        return formula



def InverseMass(data_K,op,ed):
        "inverse effective mass"
        # first give our matrices short names
        d2E = data_K.del2E_H[op:ed]
        D   = data_K.D_H[op:ed]
        V   = data_K.V_H[op:ed]
        # This is the formula to be implemented:
        formula =  Formula ( TRodd=False,Iodd=False,ndim=2,name="Inverse Mass" )
        formula.add_term ( 'mn'   ,  d2E                           ,  1. )
        formula.add_term ( 'mL,Ln',(V[:,:,:,:,None], D[:,:,:,None,:] ) ,  1. )
        formula.add_term ( 'mL,Ln',(D[:,:,:,:,None], V[:,:,:,None,:] ) , -1. )
        return formula


def Omega(data_K,op=None,ed=None,onlytrace=False):
        "an attempt for a faster implementation"
        # first give our matrices short names
        NB= data_K.nbands
        A = data_K.A_Hbar[op:ed]
        D = data_K.D_H[op:ed]
        O = data_K.Omega_Hbar[op:ed]
        # now define the "alpha" and "beta" components
        A_,D_={},{}
        for var in 'A','D':
            for c in 'alpha','beta':
                locals()[var+"_"][c]=locals()[var][:,:,:,globals()[c+'_A']]
        # This is the formula to be implemented:
        formula =  Formula ( ndim=1,TRodd=True,Iodd=False, name="Berry Curvature")
        formula.add_term( 'mn', (O, ) )
        formula.add_term( 'mL,Ln',(D_['alpha'], D_['beta' ] ) ,-2j )
        if onlytrace:
            formula.add_term( 'mL,Ln',(D_['alpha'], A_['beta' ] ) , -4 )
        else:
            formula.add_term( 'mL,Ln',(D_['alpha'], A_['beta' ] ) , -2 )
            formula.add_term( 'mL,Ln',(D_['beta' ], A_['alpha'] ) ,  2 )
        return formula

Omega_onlytrace=partial(Omega,onlytrace=True)


def derOmega(data_K,op=None,ed=None):
        "an attempt for a faster implementation"
        # first give our matrices short name
        print ("using kpoint [{}:{}]".format(op,ed))
        A  = data_K.A_Hbar[op:ed]
        dA = data_K.A_Hbar_der[op:ed]
#        print ("dA=",dA)
        _D = data_K.D_H[op:ed]
        _V = data_K.V_H[op:ed]
        O  = data_K.Omega_Hbar[op:ed,:,:,:,None]
        dO = data_K.Omega_bar_der[op:ed]
        W  = data_K.del2E_H[op:ed]

        Acal= (-(A+1j*_D)*data_K.dEig_inv[op:ed,:,:,None])[:,:,:,:,None]
        A  =  A[:,:,:,:,None]
        D  = _D[:,:,:,:,None]
        Dd = _D[:,:,:,None,:]
        V  = _V[:,:,:,:,None]
        Vd = _V[:,:,:,None,:]

        del _D,_V

        # now define the "alpha" and "beta" components
        A_,D_,W_,V_,Acal_,dA_={},{},{},{},{},{}
        for var in 'A','D','Acal','W','V','dA':
            for c in 'alpha','beta':
#                print (var,c,locals()[var].shape)
                locals()[var+"_"][c]=locals()[var][:,:,:,globals()[c+'_A']]
        # This is the formula to be implemented:
        # orange terms
        formula =  Formula ( ndim=2,TRodd=False,Iodd=True, name="derivative of Berry curvature")
        formula.add_term  ('mn',(dO, ) )
        formula.add_term  ( 'mL,Ln',(Dd, O ), -2. )
        for s,a,b in ( +1.,'alpha','beta'),(-1.,'beta','alpha'):
            #  blue terms
            formula.add_term( 'mL,Ln',    (Acal_ [a] , W_[b]         ),  2*s )
            formula.add_term( 'mL,LP,Pn', (Acal_ [a] , V_[b] , Dd    ),  2*s )
            formula.add_term( 'mL,LP,Pn', (Acal_ [a] , Vd    , D_[b] ),  2*s )
            formula.add_term( 'mL,Ll,ln', (Acal_ [a] , D_[b] , Vd    ), -2*s )
            formula.add_term( 'mL,Ll,ln', (Acal_ [a] , Dd    , V_[b] ), -2*s )
            #  green terms
            formula.add_term( 'mL,Ln',     (  D_ [a] , dA_[b]        ) , -2*s)
            formula.add_term( 'mL,LP,Pn',  (  D_ [a] , A_[b] , Dd    ) , -2*s)
            formula.add_term( 'mL,Ll,ln',  (  D_ [a] , Dd    , A_[b] ) ,  2*s)
        return formula


