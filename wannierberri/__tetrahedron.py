#------------------------------------------------------------#
# This file is distributed as part of the WannierBerri code  # 
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
#------------------------------------------------------------#

from collections import defaultdict


import functools
import numpy as np
from numba import njit,jit

## TODO : optimize to process many tetrahedra in one shot

@njit
def weights_tetra(efall,e0,e1,e2,e3,der=0):
    e1,e2,e3,e4=sorted([e0,e1,e2,e3])
    nEF=len(efall)
#    efall2=efall * efall
#    efall3=efall2* efall
    occ=np.zeros((nEF))
    denom3 = 1./ ((e4 - e1) * (e4 - e2) * (e4 - e3) )
    denom2 = 1./ ((e3 - e1) * (e4 - e1) * (e3 - e2) * (e4-e2) )
    denom1 = 1./ ((e2 - e1) * (e3 - e1) * (e4 - e1) )

    if der==0:
        c10 = -e1**3  *denom1
        c30 = -e4**3  *denom3 +1.
        c20 = ( e1**2 *  (e3 - e2) * (e4 - e2)  - (e2**2*e4 ) *    (e1 - e3) - e1*e2*e3  *    (e2 - e4)        ) *denom2
    if der <=3:
        c13 =          denom1
        c33 =          denom3
        c23 = denom2 * ( e1+e2-e3-e4  )
    if der <=2:
        c12 = -3*e1   *denom1
        c22 = (( (e3 - e2) * (e4 - e2) ) - (e1 - e3) * ( 2*e2 + e4 ) - ( e3 + e1 + e2 ) * ( e2 - e4 )          ) * denom2
        c32 = -3*e4   *denom3
    if der <=1:
        c11 = 3*e1**2 *denom1
        c21 = ( -2*e1* ( (e3 - e2) * (e4 - e2) ) + (2*e2*e4+e2**2 )  *(e1 - e3) + (e1*e2+e2*e3+e1*e3) *(e2-e4) )*denom2
        c31 = 3*e4**2 *denom3

    if der==0:
        for i in range(nEF):
            ef  = efall [i]
            if ef>=e4:
                occ[i]= 1.
            elif ef<e1:
                occ[i]=0.
            elif  ef>=e3:# c3
                occ[i] = c30+ef*(c31+ef*(c32+c33*ef))
            elif ef>=e2:   # c2
                occ[i] = c20+ef*(c21+ef*(c22+c23*ef))
            else :  #c1
                occ[i] = c10+ef*(c11+ef*(c12+c13*ef))
    elif der==1:
        for i in range(nEF):
            ef  = efall [i]
            if ef>=e4:
                occ[i] = 0.
            elif ef<e1:
                occ[i] = 0.
            elif  ef>=e3:# c3
                occ[i] = c31+ef*(2*c32+3*c33*ef)
            elif ef>=e2:   # c2
                occ[i] = c21+ef*(2*c22+3*c23*ef)
            else :  #c1
                occ[i] = c11+ef*(2*c12+3*c13*ef)
    elif der==2:
        for i in range(nEF):
            ef  = efall [i]
            if ef>=e4:
                occ[i] = 0.
            elif ef<e1:
                occ[i] = 0.
            elif  ef>=e3:# c3
                occ[i] = 2*c32+6*c33*ef
            elif ef>=e2:   # c2
                occ[i] = 2*c22+6*c23*ef
            else :  #c1
                occ[i] = 2*c12+6*c13*ef
    elif der==3:
        for i in range(nEF):
            ef  = efall[i]
            if ef>=e4:
                occ[i] = 0.
            elif ef<e1:
                occ[i] = 0.
            elif  ef>=e3:# c3
                occ[i] = 6*c33
            elif ef>=e2:   # c2
                occ[i] = 6*c23
            else :  #c1
                occ[i] = 6*c13
    return occ


def weights_parallelepiped(efermi,Ecenter,Ecorner,der=0):
    occ=np.zeros((efermi.shape))
    Ecorner=np.reshape(Ecorner,(2,2,2))
    triang1=np.array([[True,True],[True,False]])
    triang2=np.array([[False,True],[True,True]])
    for iface in 0,1:
        for Eface in Ecorner[iface,:,:],Ecorner[:,iface,:],Ecorner[ :,:,iface]:
            occ += weights_tetra(efermi,Ecenter,Eface[0,0],Eface[0,1],Eface[1,1],der=der)
            occ += weights_tetra(efermi,Ecenter,Eface[0,0],Eface[1,0],Eface[1,1],der=der)
    return occ/12.



def average_degen(E,weights):
    # make sure that degenerate bands have same weights
    borders=np.hstack( ( [0], np.where( (E[1:]-E[:-1])>1e-5)[0]+1, [len(E)]) )
    degengroups=[ (b1,b2) for b1,b2 in zip(borders,borders[1:]) if b2-b1>1]
    for b1,b2 in degengroups:
       weights[b1:b2]=weights[b1:b2].mean()


class TetraWeights():
    """the idea is to make a lazy evaluation, i.e. the weights are evaluated only once for a particular ik,ib
       the Fermi level list remains the same throughout calculation"""
    def __init__(self,eCenter,eCorners):
        self.nk, self.nb = eCenter.shape
        assert eCorners.shape==(self.nk,2,2,2,self.nb)
        self.eCenter=eCenter
        self.eCorners=eCorners
        self.eFermis=[]
        self.weights=defaultdict(lambda : defaultdict(lambda : {}))
        Eall=np.concatenate( (self.eCenter[:,None,:] , self.eCorners.reshape(self.nk,8,self.nb) ),axis=1)
        self.Emin=Eall.min(axis=1)
        self.Emax=Eall.max(axis=1)
        self.eFermi=None


    def __get_bands_in_range(self,emin,emax):
        return [np.where((self.Emax[ik]>=emin)*(self.Emin[ik]<=emax))[0] for ik in range(self.nk)]

    def __weight_1b(self,ik,ib,der):
#        print (ib,ik,der)
        if ib not in self.weights[der][ik]:
            self.weights[der][ik][ib]=weights_parallelepiped(self.eFermi,self.eCenter[ik,ib],self.eCorners[ik,:,:,:,ib],der=der)
        return self.weights[der][ik][ib]


    def weights_allbands(self,eFermi,der):
        if self.eFermi is None:
            self.eFermi=eFermi
        else :
            assert self.eFermi is eFermi


        bands_in_range=self.__get_bands_in_range(eFermi[0],eFermi[-1])

        return [{ib:self.__weight_1b(ik,ib,der)  for ib in ibrg } for ik,ibrg in enumerate(bands_in_range)]



if __name__ == '__main__' : 
    import matplotlib.pyplot as plt


    Efermi=np.linspace(-1,1,1001)
    E=np.random.random(9)-0.5
    Ecenter=E[0]
    Ecorner=E[1:].reshape(2,2,2)
    Ecorn=E[1:4]
#    occ=weights_1band_parallelepiped(Efermi,Ecenter,Ecorner)
    from time import time

    Ncycl=100

    t00=time()
    weights_tetra (Efermi,Ecenter,Ecorn[0],Ecorn[1],Ecorn[2])
    t0=time()
    [weights_tetra (Efermi,Ecenter,Ecorn[0],Ecorn[1],Ecorn[2])  for i in range(Ncycl) ]
    t1=time()
    print ("time for {} is {} ms (compilation:{})".format("tetra",(t1-t0)/Ncycl*1000,(t0-t00)*1000))

    t00=time()
    weights_parallelepiped  (Efermi,Ecenter,Ecorner)
    t0=time()
    [weights_parallelepiped (Efermi,Ecenter,Ecorner)  for i in range(Ncycl) ]
    t1=time()
    print ("time for {} is {} ms (compilation:{})".format("paral",(t1-t0)/Ncycl*1000,(t0-t00)*1000))


    t1=time()
    print ("time for {} is {} ms".format("vec_sea",(t1-t0)/Ncycl*1000*12))
    
    print (Ecenter,Ecorner)
#    occ_sea_2    = weights_1band_vec_sea_2    (Efermi,Efermi2,Efermi3,Ecenter,Ecorn)
    occ_sea      = weights_tetra (Efermi,Ecenter,Ecorn[0],Ecorn[1],Ecorn[2],der=0)
    occ_surf     = weights_tetra (Efermi,Ecenter,Ecorn[0],Ecorn[1],Ecorn[2],der=1)
    occ_surf_der = weights_tetra (Efermi,Ecenter,Ecorn[0],Ecorn[1],Ecorn[2],der=2)

    occ_surf_fd=occ_sea*0
    occ_surf_fd[1:-1]=(occ_sea[2:]-occ_sea[:-2])/(Efermi[2]-Efermi[0])

    occ_surf_der_fd=occ_sea*0
    occ_surf_der_fd[2:-2]=(occ_sea[4:]+occ_sea[:-4]-2*occ_sea[2:-2])/(Efermi[2]-Efermi[0])**2
    
    plt.plot(Efermi,occ_sea   ,c='blue')
#    plt.plot(Efermi,occ_sea_2 ,c='red')
#    plt.plot(Efermi,(occ_sea_2-occ_sea) , c='green' )

    plt.scatter(Efermi,occ_surf_fd ,c='green')
    plt.scatter(Efermi,occ_surf_der_fd ,c='red')
    plt.plot(Efermi,occ_surf , c='yellow')
    plt.plot(Efermi,occ_surf_der , c='cyan')

    for x in E[1:4]:
        plt.axvline(x,c='blue')
    plt.axvline(Ecenter,c='red')
    plt.xlim(-0.6,0.6)
    plt.show()
    exit()
