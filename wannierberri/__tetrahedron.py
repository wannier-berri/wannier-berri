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



def weights_1band_vec_sea(ef,e0,etetra):
    # energies will be sorted, remember which is at the corner of interest
    dosef=0
    etetra=np.hstack([[e0],etetra])
    ivertex=np.sum(e0>etetra[1:])
    e1,e2,e3,e4=np.sort(etetra)
    occ=np.zeros(len(ef),dtype=float)
    occ[ef>=e4]=1.
 
    select= (ef>=e3)*(ef<e4)
    efsel=ef[select]
    c4 =  (e4 - efsel) **3 / ((e4 - e1) * (e4 - e2) * (e4 - e3))
#    dosef = 0.3 * (e4 - efsel) **2 /((e4 - e1)* (e4 - e2) * (e4 - e3))*(e1 + e2 + e3 + e4 - 4. * e0 )
    if   ivertex in (0,1,2) :
        occ[select] =  1.  - c4 * (e4 - efsel) / (e4 - e0) + dosef 
    elif  ivertex==3: 
        occ[select] =  1.  - c4 * (4. - (e4 - efsel) * (1. / (e4 - e1) + 1. / (e4 - e2) 
                   + 1. / (e4 - e3) ) ) + dosef 

    select= (ef>=e2)*(ef<e3)
    efsel=ef[select]
    c1 = (efsel - e1) **2 / ((e4 - e1) * (e3 - e1))
    c2 = (efsel - e1) * (efsel - e2) * (e3 - efsel)  / ((e4 - e1) * (e3 - e2) * (e3 - e1))
    c3 = (efsel - e2) **2 * (e4 - efsel) /( (e4 - e2)  * (e3 - e2) * (e4 - e1))
#    dosef = 0.1 / (e3 - e1) / (e4 - e1) * (3. * 
#               (e2 - e1) + 6. * (efsel - e2) - 3. * (e3 - e1 + e4 - e2) 
#               * (efsel - e2) **2 / (e3 - e2) / (e4 - e2) )* (e1 + e2 +  e3 + e4 - 4. * e0 ) 
    if ivertex==0:
        occ[select] =   c1 + (c1 + c2) * (e3 - efsel) / (e3 - e1) + (c1 + c2 + c3) * (e4 - efsel) / (e4 - e1) + dosef
    elif ivertex==1:
        occ[select] =    c1 + c2 + c3 + (c2 + c3)  * (e3 - efsel) / (e3 - e2) + c3 * (e4 - efsel) / (e4 - e2) + dosef 
    elif ivertex==2:
        occ[select] =    (c1 + c2) * (efsel - e1) / (e3 - e1) + (c2 + c3) * (efsel - e2) / (e3 - e2) + dosef 
    elif ivertex==3:
        occ[select] =    (c1 + c2 + c3) * (efsel - e1)  / (e4 - e1) + c3 * (efsel - e2) / (e4 - e2) + dosef 


    select= (ef>=e1)*(ef<e2)
    efsel=ef[select]
    c4 = (efsel - e1) **3 / (e2 - e1) / (e3 - e1) / (e4 - e1)
#    dosef = 0.3 * (efsel - e1) **2 / (e2 - e1) / (e3 - e1) / (e4 - e1) * (e1 + e2 + e3 + e4 - 4. * e0 ) 
    if   ivertex==0:
        occ[select] =   c4 * (4. - (efsel - e1) * (1. / (e2 - e1) + 1. / (e3 - e1) + 1. / (e4 - e1) ) )   + dosef
    elif ivertex in (1,2,3):
        occ[select] =   c4 * (efsel - e1) / (e0 - e1)  + dosef

    return occ




def weights_1band_parallelepiped(efermi,Ecenter,Ecorner):
    occ=np.zeros(efermi.shape,dtype=float)
    Ecorner=np.reshape(Ecorner,(2,2,2))
    for iface in 0,1:
        for _Eface in Ecorner[iface,:,:],Ecorner[:,iface,:],Ecorner[:,:,iface]:
            Eface=np.reshape(_Eface,-1)
            for j in range(4):
                eface =  np.roll(Eface,j)[:-1]
#                print (Eface.shape,eface.shape)
                occ   += weights_1band_vec_sea(efermi,Ecenter,eface)
    return occ/24.



if __name__ == '__main__' : 
    import matplotlib.pyplot as plt

    Efermi=np.linspace(-1,1,101)
    E=np.random.random(9)-0.5
    Ecenter=E[0]
    Ecorner=E[1:].reshape(2,2,2)
    #Ecorn=E[1:4]
    occ=weights_1band_parallelepiped(Efermi,Ecenter,Ecorner)
    print (Ecenter,Ecorner)
    #occ=weights_1band_vec_sea(Efermi,Ecenter,Ecorn)
    print (occ)
    
    
    plt.plot(Efermi,occ)
    for x in E[1:]:
        plt.axvline(x,c='blue')
    plt.axvline(Ecenter,c='red')
    plt.xlim(-0.6,0.6)
    plt.show()
    exit()


    for x in Ecorner:
        plt.axvline(x,col='blue')
    plt.axvline(Ecenter,col='red')




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

    