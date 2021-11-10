"""Here we define some models, that can be used to test the code, or just to play around"""

import numpy as np

def Haldane_tbm(
    delta=0.2,
    t=-1.0,
    hop2=0.15,
    phi = np.pi/2.
    ):
    """
    Defines a Haldane model within tbmodels

    Parameters
    -----------
    delta : float
        difference between the on-site potentials of the two atoms
    t : float
        nearest-neighbour hopping
    hop2 : float
        magnitude of next nearest-neighbour hopping 
    phi : float
        phase of next nearest-neighbour hopping 

    """
    import tbmodels

    t2 =hop2*np.exp(1.j*phi)
    t2c=t2.conjugate()
    my_model = tbmodels.Model(
            on_site=[delta, -delta],uc = [[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]], dim=2, occ=1, pos=[[1./3.,1./3.],[2./3.,2./3.]]
            )
    my_model.add_hop(t, 0, 1, [ 0, 0])
    my_model.add_hop(t, 1, 0, [ 1, 0])
    my_model.add_hop(t, 1, 0, [ 0, 1])
    my_model.add_hop(t2 , 0, 0, [ 1, 0])
    my_model.add_hop(t2 , 1, 1, [ 1,-1])
    my_model.add_hop(t2 , 1, 1, [ 0, 1])
    my_model.add_hop(t2c, 1, 1, [ 1, 0])
    my_model.add_hop(t2c, 0, 0, [ 1,-1])
    my_model.add_hop(t2c, 0, 0, [ 0, 1])
    
    return my_model


def Haldane_ptb(
    delta=0.2,
    t=-1.0,
    hop2=0.15,
    phi = np.pi/2.):
    """same as Haldane_tbm, but uses pythTB"""
    import pythtb 
    lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
    orb=[[1./3.,1./3.],[2./3.,2./3.]]

    my_model=pythtb.tb_model(2,2,lat,orb)

    delta=0.2
    t=-1.0
    t2 =hop2*np.exp(1.j*phi)
    t2c=t2.conjugate()

    my_model.set_onsite([-delta,delta])
    my_model.set_hop(t, 0, 1, [ 0, 0])
    my_model.set_hop(t, 1, 0, [ 1, 0])
    my_model.set_hop(t, 1, 0, [ 0, 1])
    my_model.set_hop(t2 , 0, 0, [ 1, 0])
    my_model.set_hop(t2 , 1, 1, [ 1,-1])
    my_model.set_hop(t2 , 1, 1, [ 0, 1])
    my_model.set_hop(t2c, 1, 1, [ 1, 0])
    my_model.set_hop(t2c, 0, 0, [ 1,-1])
    my_model.set_hop(t2c, 0, 0, [ 0, 1])
    
    return my_model


def Chiral(
    delta=2,
    t=1,
    hop2=1./3,
    phi=np.pi/10,
    hopz=0.2
    ):
    """Create a chiral model  - a chirally stacked haldane model.
       Following the article by 
       `Yoda,Yokoyama & Murakami 2018 <https://doi.org/10.1021/acs.nanolett.7b04300>`__
       this model breaks  time-reversal and inversion, so it 
       can be used to test almost any quantity. 
       Has a symmetry C3z

    Parameters
    -----------
    delta : float
        difference between the on-site potentials of the two atoms
    t : float
        nearest-neighbour in-plane hopping
    hop2 : float
        magnitude of next nearest-neighbour in-plane hopping 
    phi : float
        phase of next nearest-neighbour in-plane hopping 
    hopz : float or complex
        chiral (real) hopping in the z direction

    """

    import pythtb 

    lat=[[1.0,0.0,0.0],[0.5,np.sqrt(3.0)/2.0,0.0],[0.0,0.0,1.0]]
    # define coordinates of orbitals
    orb=[[1./3.,1./3.,0.0],[2./3.,2./3.,0.0]]

    # make tree dimensional (stacked) tight-binding Haldene model
    my_model=pythtb.tb_model(3,3,lat,orb)

    # set model parameters
    t2=hop2*np.exp(1.j*phi)

    # set on-site energies
    my_model.set_onsite([-delta,delta])
    # set hoppings (one for each connected pair of orbitals)
    # from j in R to i in 0
    # (amplitude, i, j, [lattice vector to cell containing j])
    my_model.set_hop(t, 0, 1, [ 0, 0,0])
    my_model.set_hop(t, 1, 0, [ 1, 0,0])
    my_model.set_hop(t, 1, 0, [ 0, 1,0])
    # add second neighbour complex hoppings
    my_model.set_hop(t2 , 0, 0, [  0,-1,0])
    my_model.set_hop(t2 , 0, 0, [  1, 0,0])
    my_model.set_hop(t2 , 0, 0, [ -1, 1,0])
    my_model.set_hop(t2 , 1, 1, [ -1, 0,0])
    my_model.set_hop(t2 , 1, 1, [  1,-1,0])
    my_model.set_hop(t2 , 1, 1, [  0, 1,0])

    # add chiral hoppings
    my_model.set_hop(hopz  , 0, 0, [  0,-1,1])
    my_model.set_hop(hopz  , 0, 0, [  1, 0,1])
    my_model.set_hop(hopz  , 0, 0, [ -1, 1,1])
    my_model.set_hop(hopz  , 1, 1, [ -1, 0,1])
    my_model.set_hop(hopz  , 1, 1, [  1,-1,1])
    my_model.set_hop(hopz  , 1, 1, [  0, 1,1])
    return my_model
