#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# part of this file is based on                              #
# the corresponding Fortran90 code from                      #
#                                 Quantum Espresso  project  #
#                                                            #                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                                                            #
# The webpage of the QuantumEspresso  code is                #
#            https://www.quantum-espresso.org/               #
#------------------------------------------------------------#
#                                                            #
#  Translated to python and adapted for wannier19 project by #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#


#! Copyright message from Quantum Espresso:
#! Copyright (C) 2016 Quantum ESPRESSO Foundation
#! This file is distributed under the terms of the
#! GNU General Public License. See the file `License'
#! in the root directory of the present distribution,
#! or http://www.gnu.org/copyleft/gpl.txt .
#!


import numpy as np

## TODO : optimize to process many tetrahedra in one shot
def weights_1band_vec_sea(efall,e0,etetra):
    # energies will be sorted, remember which is at the corner of interest
    etetra=np.hstack([[e0],etetra])
    e1,e2,e3,e4=np.sort(etetra)
    occ=np.zeros(len(efall),dtype=float)
    occ[efall>=e4]=1.
 
    select= (efall>=e3)*(efall<e4)
    ef=efall[select]
    occ[select]=1. - (e4 - ef) **3 / ((e4 - e1) * (e4 - e2) * (e4 - e3))

    select= (efall>=e1)*(efall<e2)
    ef=efall[select]
    occ[select]= (ef - e1) **3 / ((e2 - e1) * (e3 - e1) * (e4 - e1))

    select= (efall>=e2)*(efall<e3)
    ef=efall[select]  
    occ[select]= (
                    (ef-e1)**2  *             ( (e3-e2) * (e4-e2) ) 
                  + (ef-e2)**2  * (e4-ef)   * ( (e3-e1)           ) 
#                  - (ef-e2)*(ef-e1)*(ef-e3) * (    (e4-e2)        )
                   - (  ef**3  -ef**2*(e3+e1+e2) +ef*(e1*e2+e2*e3+e1*e3) - e1*e2*e3) * (    (e4-e2)        )
                   )    / ( (e3-e1) * (e4-e1) * (e3-e2) * (e4-e2) )
    return occ


def weights_1band_vec_surf(efall,e0,etetra):
    # energies will be sorted, remember which is at the corner of interest
    etetra=np.hstack([[e0],etetra])
    e1,e2,e3,e4=np.sort(etetra)
    occ=np.zeros(len(efall),dtype=float)
 
    select= (efall>=e3)*(efall<e4)
    ef=efall[select]
    occ[select]=3*(e4 - ef) **2 / ((e4 - e1) * (e4 - e2) * (e4 - e3))

    select= (efall>=e1)*(efall<e2)
    ef=efall[select]
    occ[select]= 3* (ef - e1) **2 / ((e2 - e1) * (e3 - e1) * (e4 - e1))

    select= (efall>=e2)*(efall<e3)
    ef=efall[select]  
    occ[select]= ( 
                    2*(ef-e1)  *   ( (e3-e2) * (e4-e2) ) 
                    + (ef-e2)* (2*e4+e2-3*ef )  * ( (e3-e1)  ) 
                  -(ef*(3*ef-2*(e1+e2+e3))+(e1*e2+e2*e3+e1*e3)   )*         (    (e4-e2)        )
                   )    / ( (e3-e1) * (e4-e1) * (e3-e2) * (e4-e2) )
    return occ




def weights_1band_parallelepiped(efermi,Ecenter,Ecorner):
    occ=np.zeros(efermi.shape,dtype=float)
    Ecorner=np.reshape(Ecorner,(2,2,2))
    triang1=np.array([[True,True],[True,False]])
    triang2=np.array([[False,True],[True,True]])
    for iface in 0,1:
        for _Eface in Ecorner[iface,:,:],Ecorner[:,iface,:],Ecorner[:,:,iface]:
            for eface in _Eface[triang1],_Eface[triang2]:
                occ   += weights_1band_vec_sea(efermi,Ecenter,eface)
    return occ/12.





if __name__ == '__main__' : 
    import matplotlib.pyplot as plt

    Efermi=np.linspace(-1,1,1001)
    E=np.random.random(9)-0.5
    Ecenter=E[0]
    Ecorner=E[1:].reshape(2,2,2)
    Ecorn=E[1:4]
#    occ=weights_1band_parallelepiped(Efermi,Ecenter,Ecorner)
    print (Ecenter,Ecorner)
    occ=weights_1band_vec_surf(Efermi,Ecenter,Ecorn)
    occ2=weights_1band_vec_sea(Efermi,Ecenter,Ecorn)
    occ3=occ2*0
    occ3[1:-1]=(occ2[2:]-occ2[:-2])/(Efermi[2]-Efermi[0])
    print (occ)
    
    
    plt.scatter(Efermi,occ3 ,c='green')
    plt.plot(Efermi,occ , c='blue')
    for x in E[1:4]:
        plt.axvline(x,c='blue')
    plt.axvline(Ecenter,c='red')
    plt.xlim(-0.6,0.6)
    plt.show()
    exit()


#    for x in Ecorn:
#        plt.axvline(x,col='blue')
#    plt.axvline(Ecenter,col='red')




def weights_all_bands_parallelepiped(efermi,Ecenter,Ecorner):
#    occ=np.array([weights_1band_parallelepiped(etetra,Ef)
    return np.array([weights_1band(etetra,Ef) for etetra in Etetra])




def average_degen(E,weights):
    # make sure that degenerate bands have same weights
    borders=np.hstack( ( [0], np.where( (E[1:]-E[:-1])>1e-5)[0]+1, [len(E)]) )
    degengroups=[ (b1,b2) for b1,b2 in zip(borders,borders[1:]) if b2-b1>1]
    for b1,b2 in degengroups:
       weights[b1:b2]=weights[b1:b2].mean()

def weights_all_bands_1tetra(Etetra,Ef):
    return np.array([weights_1band(etetra,Ef) for etetra in Etetra])

def get_occ(E,E_neigh,Ef):
#  E_neigh is a dict (i,j,k):E 
# where i,j,k = -1,0,+1 - are coordinates of a k-point, relative to the reference point
    num_wann=E.shape[0]
    occ=np.zeros(num_wann)
    weights=np.zeros(num_wann)
    Etetra=np.zeros( (num_wann,4),dtype=float)
    Etetra[:,0]=E
    for tetra in __TETRA_NEIGHBOURS:
       for i,p in enumerate(tetra):
          Etetra[:,i+1]=E_neigh[tuple(p)]
          weights+=weights_all_bands_1tetra(Etetra,Ef)
    
    average_degen(E,weights)
    return weights/24.

    