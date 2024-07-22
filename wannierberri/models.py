"""Here we define some models, that can be used to test the code, or just to play around"""

import numpy as np


def Haldane_tbm(delta=0.2, hop1=-1.0, hop2=0.15, phi=np.pi / 2):
    """
    Defines a Haldane model within `TBmodels <https://tbmodels.greschd.ch>`__

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

    Notes
    -----
    TBmodels  should be installed to use this (`pip install tbmodels`)

    """
    import tbmodels

    t2 = hop2 * np.exp(1.j * phi)
    t2c = t2.conjugate()
    my_model = tbmodels.Model(
        on_site=[-delta, delta],
        uc=[[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]],
        dim=2,
        occ=1,
        pos=[[1. / 3., 1. / 3.], [2. / 3., 2. / 3.]])
    my_model.add_hop(hop1, 0, 1, [0, 0])
    my_model.add_hop(hop1, 1, 0, [1, 0])
    my_model.add_hop(hop1, 1, 0, [0, 1])
    my_model.add_hop(t2, 0, 0, [1, 0])
    my_model.add_hop(t2, 1, 1, [1, -1])
    my_model.add_hop(t2, 1, 1, [0, 1])
    my_model.add_hop(t2c, 1, 1, [1, 0])
    my_model.add_hop(t2c, 0, 0, [1, -1])
    my_model.add_hop(t2c, 0, 0, [0, 1])

    return my_model


def Haldane_ptb(delta=0.2, hop1=-1.0, hop2=0.15, phi=np.pi / 2):
    """same as :func:`~wannierberri.models.Haldane_tbm`, but uses `PythTB <http://www.physics.rutgers.edu/pythtb/>`__

    Notes
    -----
    PythTB should be installed to use this (`pip install pythtb`)
    """
    import pythtb
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1. / 3., 1. / 3.], [2. / 3., 2. / 3.]]

    my_model = pythtb.tb_model(2, 2, lat, orb)

    delta = 0.2
    t2 = hop2 * np.exp(1.j * phi)
    t2c = t2.conjugate()

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(hop1, 0, 1, [0, 0])
    my_model.set_hop(hop1, 1, 0, [1, 0])
    my_model.set_hop(hop1, 1, 0, [0, 1])
    my_model.set_hop(t2, 0, 0, [1, 0])
    my_model.set_hop(t2, 1, 1, [1, -1])
    my_model.set_hop(t2, 1, 1, [0, 1])
    my_model.set_hop(t2c, 1, 1, [1, 0])
    my_model.set_hop(t2c, 0, 0, [1, -1])
    my_model.set_hop(t2c, 0, 0, [0, 1])

    return my_model


def Chiral(delta=2, hop1=1, hop2=1 / 3, phi=np.pi / 10, hopz_right=0.0, hopz_left=0.2, hopz_vert=0.0):
    """Create a chiral model  - a chirally stacked haldane model -
       using `PythTB <http://www.physics.rutgers.edu/pythtb/>`__
       Following the article by
       `Yoda,Yokoyama & Murakami 2018 <https://doi.org/10.1021/acs.nanolett.7b04300>`__
       this model breaks  time-reversal and inversion, so it
       can be used to test almost any quantity.
       Has a symmetry C3z

    Parameters
    -----------
    delta : float
        difference between the on-site potentials of the two atoms
    hop1 : float
        nearest-neighbour in-plane hopping
    hop2 : float
        magnitude of next nearest-neighbour in-plane hopping
    phi : float
        phase of next nearest-neighbour in-plane hopping
    hopz_vert : float or complex
        interlayer vertical hopping
    hopz_right : float or complex
        chiral right-handed  hopping in the z direction
    hopz_left : float or complex
        chiral left-handed  hopping in the z direction

    Notes
    -----
    PythTB should be installed to use this (`pip install pythtb`)

    """

    import pythtb

    lat = [[1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 1.0]]
    # define coordinates of orbitals
    orb = [[1. / 3., 1. / 3., 0.0], [2. / 3., 2. / 3., 0.0]]

    # make tree dimensional (stacked) tight-binding Haldene model
    my_model = pythtb.tb_model(3, 3, lat, orb)

    # set model parameters
    t2 = hop2 * np.exp(1.j * phi)

    # set on-site energies
    my_model.set_onsite([-delta, delta])
    # set hoppings (one for each connected pair of orbitals)
    # from j in R to i in 0
    # (amplitude, i, j, [lattice vector to cell containing j])
    my_model.set_hop(hop1, 0, 1, [0, 0, 0])
    my_model.set_hop(hop1, 1, 0, [1, 0, 0])
    my_model.set_hop(hop1, 1, 0, [0, 1, 0])
    # add second neighbour complex hoppings
    my_model.set_hop(t2, 0, 0, [0, -1, 0])
    my_model.set_hop(t2, 0, 0, [1, 0, 0])
    my_model.set_hop(t2, 0, 0, [-1, 1, 0])
    my_model.set_hop(t2, 1, 1, [-1, 0, 0])
    my_model.set_hop(t2, 1, 1, [1, -1, 0])
    my_model.set_hop(t2, 1, 1, [0, 1, 0])

    # add vertical hoppings
    my_model.set_hop(hopz_vert, 0, 0, [0, 0, 1])
    my_model.set_hop(hopz_vert, 1, 1, [0, 0, 1])

    # add chiral hoppings (left-handed)
    my_model.set_hop(hopz_left, 0, 0, [0, -1, 1])
    my_model.set_hop(hopz_left, 0, 0, [1, 0, 1])
    my_model.set_hop(hopz_left, 0, 0, [-1, 1, 1])
    my_model.set_hop(hopz_left, 1, 1, [-1, 0, 1])
    my_model.set_hop(hopz_left, 1, 1, [1, -1, 1])
    my_model.set_hop(hopz_left, 1, 1, [0, 1, 1])

    # add chiral hoppings (right-handed)
    my_model.set_hop(hopz_right, 0, 0, [0, -1, -1])
    my_model.set_hop(hopz_right, 0, 0, [1, 0, -1])
    my_model.set_hop(hopz_right, 0, 0, [-1, 1, -1])
    my_model.set_hop(hopz_right, 1, 1, [-1, 0, -1])
    my_model.set_hop(hopz_right, 1, 1, [1, -1, -1])
    my_model.set_hop(hopz_right, 1, 1, [0, 1, -1])

    return my_model


def CuMnAs_2d(
    nx=0,
    ny=1,
    nz=0,
    hop1=1,
    hop2=0.08,
    l=0.8,
    J=0.6,
    dt=0.0,
):
    """Create a 2D model of antiferromagnetic CuMnAs
       using `PythTB <http://www.physics.rutgers.edu/pythtb/>`__
       Following the article by
       `Šmejkal, Železný, Sinova & Jungwirth 2017 <https://doi.org/10.1103/PhysRevLett.118.106402>`__
       this model breaks time-reversal and inversion, but it
       preserves the combination of both (P*T).

    Parameters
    -----------
    nx : float
        x-component of the Néel vector
    ny : float
        y-component of the Néel vector
    nz : float
        z-component of the Néel vector
    hop1 : float
        magnitude of nearest-neighbour hopping (A-A)
    hop2 : float
        magnitude of next nearest-neighbour hopping (A-B)
    l : float
        second nearest-neighbor spin-orbit coupling
    J : float
        magnitude of the Néel vector
    dt : float
        PT-breaking term making the A-A and B-B hoppings different

    Notes
    -----
    PythTB should be installed to use this (`pip install pythtb`)

    """

    import pythtb

    # initialize the CuMnAs model
    lat = [[1.0, 0.0], [0.0, 1.0]]
    orb = [[0., 0.], [0., 0.], [0.5, 0.5], [0.5, 0.5]]

    # Neel vector orientation
    nabs = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = nx / nabs
    ny = ny / nabs
    nz = nz / nabs

    my_model = pythtb.tb_model(2, 2, lat, orb)

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    # nearest-neighbor (A-B hopping)
    my_model.set_hop(-0.5 * hop1, 0, 2, [0, 0])
    my_model.set_hop(-0.5 * hop1, 0, 2, [-1, 0])
    my_model.set_hop(-0.5 * hop1, 0, 2, [0, -1])
    my_model.set_hop(-0.5 * hop1, 0, 2, [-1, -1])
    my_model.set_hop(-0.5 * hop1, 1, 3, [0, 0])
    my_model.set_hop(-0.5 * hop1, 1, 3, [-1, 0])
    my_model.set_hop(-0.5 * hop1, 1, 3, [0, -1])
    my_model.set_hop(-0.5 * hop1, 1, 3, [-1, -1])
    # second nearest-neighbor (A-A hopping)
    my_model.set_hop(-0.5 * hop2, 0, 0, [1, 0])
    my_model.set_hop(-0.5 * hop2, 0, 0, [0, 1])
    my_model.set_hop(-0.5 * hop2, 1, 1, [1, 0])
    my_model.set_hop(-0.5 * hop2, 1, 1, [0, 1])
    my_model.set_hop(-0.5 * hop2, 2, 2, [1, 0])
    my_model.set_hop(-0.5 * hop2, 2, 2, [0, 1])
    my_model.set_hop(-0.5 * hop2, 3, 3, [1, 0])
    my_model.set_hop(-0.5 * hop2, 3, 3, [0, 1])
    # second neighbor SOC
    my_model.set_hop(-0.5 * l, 0, 1, [1, 0])
    my_model.set_hop(0.5 * l, 0, 1, [-1, 0])
    my_model.set_hop(0.5 * l * 1j, 0, 1, [0, 1])
    my_model.set_hop(-0.5 * l * 1j, 0, 1, [0, -1])
    my_model.set_hop(0.5 * l, 2, 3, [1, 0])
    my_model.set_hop(-0.5 * l, 2, 3, [-1, 0])
    my_model.set_hop(-0.5 * l * 1j, 2, 3, [0, 1])
    my_model.set_hop(0.5 * l * 1j, 2, 3, [0, -1])
    # AF exchange coupling
    my_model.set_hop(J * (nx - 1j * ny), 0, 1, [0, 0])
    my_model.set_hop(J * (-nx + 1j * ny), 2, 3, [0, 0])
    # set on-site energies
    my_model.set_onsite([J * nz, -J * nz, -J * nz, J * nz])

    # break PT by making sublattices different
    my_model.set_hop(-0.5 * dt, 0, 0, [1, 0], mode='add')
    my_model.set_hop(-0.5 * dt, 0, 0, [0, 1], mode='add')
    my_model.set_hop(-0.5 * dt, 1, 1, [1, 0], mode='add')
    my_model.set_hop(-0.5 * dt, 1, 1, [0, 1], mode='add')
    my_model.set_hop(0.5 * dt, 2, 2, [1, 0], mode='add')
    my_model.set_hop(0.5 * dt, 2, 2, [0, 1], mode='add')
    my_model.set_hop(0.5 * dt, 3, 3, [1, 0], mode='add')
    my_model.set_hop(0.5 * dt, 3, 3, [0, 1], mode='add')

    return my_model


def KaneMele_ptb(topological):
    """Return a Kane-Mele model in the normal or topological phase.
      example taken from `PythTB web page  <https://www.physics.rutgers.edu/pythtb/examples.html#kane-mele-model-using-spinor-features>`__
      topological : `even` or `odd`
      """
    import pythtb
    # define lattice vectors
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    # define coordinates of orbitals
    orb = [[1. / 3., 1. / 3.], [2. / 3., 2. / 3.]]

    # make two dimensional tight-binding Kane-Mele model
    ret_model = pythtb.tb_model(2, 2, lat, orb, nspin=2)

    # set model parameters depending on whether you are in the topological
    # phase or not
    if topological == "even":
        esite = 2.5
    elif topological == "odd":
        esite = 1.0
    # set other parameters of the model
    thop = 1.0
    spin_orb = 0.6 * thop * 0.5
    rashba = 0.25 * thop

    # set on-site energies
    ret_model.set_onsite([esite, (-1.0) * esite])

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])

    # useful definitions
    sigma_x = np.array([0., 1., 0., 0])
    sigma_y = np.array([0., 0., 1., 0])
    sigma_z = np.array([0., 0., 0., 1])

    # spin-independent first-neighbor hoppings
    ret_model.set_hop(thop, 0, 1, [0, 0])
    ret_model.set_hop(thop, 0, 1, [0, -1])
    ret_model.set_hop(thop, 0, 1, [-1, 0])

    # second-neighbour spin-orbit hoppings (s_z)
    ret_model.set_hop(-1.j * spin_orb * sigma_z, 0, 0, [0, 1])
    ret_model.set_hop(1.j * spin_orb * sigma_z, 0, 0, [1, 0])
    ret_model.set_hop(-1.j * spin_orb * sigma_z, 0, 0, [1, -1])
    ret_model.set_hop(1.j * spin_orb * sigma_z, 1, 1, [0, 1])
    ret_model.set_hop(-1.j * spin_orb * sigma_z, 1, 1, [1, 0])
    ret_model.set_hop(1.j * spin_orb * sigma_z, 1, 1, [1, -1])

    # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)
    r3h = np.sqrt(3.0) / 2.0
    # bond unit vectors are (r3h,half) then (0,-1) then (-r3h,half)
    ret_model.set_hop(1.j * rashba * (0.5 * sigma_x - r3h * sigma_y), 0, 1, [0, 0], mode="add")
    ret_model.set_hop(1.j * rashba * (-1.0 * sigma_x), 0, 1, [0, -1], mode="add")
    ret_model.set_hop(1.j * rashba * (0.5 * sigma_x + r3h * sigma_y), 0, 1, [-1, 0], mode="add")

    return ret_model


def Chiral_OSD():
    """ Chiral model to trest OSD
    """

    from pythtb import tb_model
    # define lattice vectors
    a = 1.0
    c = 1.0
    a1 = [np.sqrt(3.0) * a, 0.0, 0.0]
    a2 = [np.sqrt(3.0) * a / 2.0, 3.0 * a / 2.0, 0.0]
    a3 = [0.0, 0.0, c]
    lat = [a1, a2, a3]

    # define coordinates of orbitals
    orb = [[0.0, 0.0, 0.0],
           [1 / 3, 1 / 3, 0.0]]

    # make three dimensional tight-binding model
    my_model = tb_model(3, 3, lat, orb, nspin=2)

    # set model parameters
    t1 = 1.0
    Delta = 0.5 * t1
    l1 = -0.06 * t1
    l2 = 0.05 * t1

    # useful definitions
    # Pauli matrices
    P0 = np.array([1.0, 0.0, 0.0, 0.0])
    P1 = np.array([0.0, 1.0, 0.0, 0.0])
    P2 = np.array([0.0, 0.0, 1.0, 0.0])
    P3 = np.array([0.0, 0.0, 0.0, 1.0])
    # nearest-neighbor vectors
    d1 = [0.0, a, 0.0]
    d2 = [np.sqrt(3.0) * a / 2.0, -a / 2.0, 0.0]
    d3 = [-np.sqrt(3.0) * a / 2.0, -a / 2.0, 0.0]

    # set on-site energies
    my_model.set_onsite([-Delta * P0, Delta * P0])
    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    # nearest-neighbor hoppings (spin-independent and -dependent):
    my_model.set_hop(1.j * t1 * P0, 1, 0, [0, 0, 0])
    my_model.set_hop(1.j * t1 * P0, 1, 0, [1, 0, 0])
    my_model.set_hop(1.j * t1 * P0, 1, 0, [0, 1, 0])

    my_model.set_hop((1.j * l1 / a) * (d1[0] * P1 + d1[1] * P2), 1, 0, [0, 1, 0], mode='add')
    my_model.set_hop((1.j * l1 / a) * (d2[0] * P1 + d2[1] * P2), 1, 0, [1, 0, 0], mode='add')
    my_model.set_hop((1.j * l1 / a) * (d3[0] * P1 + d3[1] * P2), 1, 0, [0, 0, 0], mode='add')

    # next-to-nearest neighbor hoppings (helical):
    my_model.set_hop((1.j * l2 / a) * (a1[0] * P1 + a1[1] * P2 + a3[2] * P3), 0, 0, [1, 0, 1])
    my_model.set_hop((1.j * l2 / a) * (-a2[0] * P1 - a2[1] * P2 + a3[2] * P3), 0, 0, [0, -1, 1])
    my_model.set_hop((1.j * l2 / a) * ((-a1[0] + a2[0]) * P1 + (-a1[1] + a2[1]) * P2 + a3[2] * P3), 0, 0, [-1, 1, 1])

    my_model.set_hop((1.j * l2 / a) * (a2[0] * P1 + a2[1] * P2 + a3[2] * P3), 1, 1, [0, 1, 1])
    my_model.set_hop((1.j * l2 / a) * (-a1[0] * P1 - a1[1] * P2 + a3[2] * P3), 1, 1, [-1, 0, 1])
    my_model.set_hop((1.j * l2 / a) * ((a1[0] - a2[0]) * P1 + (a1[1] - a2[1]) * P2 + a3[2] * P3), 1, 1, [1, -1, 1])
    return my_model
