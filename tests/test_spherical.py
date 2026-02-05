import pytest
from scipy.special import spherical_jn
from wannierberri.symmetry.orbitals import (
    Bessel_j_radial_int, radial_function_tilde, SphericalHarmonics,
    Projector, bohr_radius_angstrom)
import numpy as np


@pytest.mark.parametrize("a0factor", [0.5, 1.0, 2.0])
def test_radial_functions(a0factor):
    """
    check orthonormality of radial functions
    """
    from wannierberri.symmetry.orbitals import radial_function_tilde
    import numpy as np
    r = np.linspace(0, 100, 10000)
    dr = r[1] - r[0]
    a0 = 0.529 * a0factor
    y0 = radial_function_tilde(1, r / a0) * a0**(-3 / 2)
    y1 = radial_function_tilde(2, r / (2 * a0)) * (a0)**(-3 / 2)
    y2 = radial_function_tilde(3, r / (3 * a0)) * (a0)**(-3 / 2)
    y0[0] /= np.sqrt(2)
    y1[0] /= np.sqrt(2)
    y2[0] /= np.sqrt(2)

    Y = np.array([y0, y1, y2])
    integral = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            integral[i, j] = np.sum(Y[i] * Y[j] * r**2) * dr
    assert np.allclose(integral, np.eye(3), atol=1e-5), f"Integral matrix:\n{integral}"


@pytest.mark.parametrize("l", [0, 1, 2])
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("prec", ["default", "high"])
def test_Bessel_int(l, n, prec):
    if prec == "high":
        if n >= 3:
            pytest.skip("high precision test for n>=3 is too slow")
        bessel = Bessel_j_radial_int(k0=5, kmax=200, dk=0.005, dtk=0.1, kmin=1e-4,
                    x0=10, xmax=100, dx=0.001, dtx=0.01,)
        tol = 1e-5 if n == 1 else 5e-5
    elif prec == "default":
        bessel = Bessel_j_radial_int()
        tol = 1e-3

    nr = 10000
    r = np.linspace(0, 100, nr)
    dr = r[1] - r[0]
    bessel = Bessel_j_radial_int()
    k_list = np.linspace(0, 5, 100)
    I = bessel(l=l, k=k_list, n=n)
    # print(f'Integral for l={l}, n=1:')
    # print(I)
    I2 = np.zeros_like(I)
    for i, k in enumerate(k_list):
        j_l = spherical_jn(l, k * r * n)
        Rn = radial_function_tilde(n=n, r=r)
        I2[i] = n**3 * np.sum(j_l * Rn * r**2) * dr
    diff = np.max(np.abs(I - I2))
    assert diff < tol, f"Max difference in Bessel integral for l={l}, n={n} is {diff}, which is larger than tol={tol} for {prec} precision.]"


def test_SphericalHarmonics():
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    costheta = np.cos(theta)
    costheta, phi = np.meshgrid(costheta, phi, indexing='ij')
    costheta = costheta.flatten()
    phi = phi.flatten()
    print(f"costheta shape: {costheta.shape}, phi shape: {phi.shape}")
    sintheta = np.sqrt(1 - costheta**2)
    sph = SphericalHarmonics(costheta=costheta, phi=phi)
    harmonics = {orb: sph(orb) for orb in ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2']}
    # check normalization
    dAngle = (theta[1] - theta[0]) * (phi[1] - phi[0])
    for i, Y in harmonics.items():
        for j, Y2 in harmonics.items():
            integral = np.sum(Y * Y2 * sintheta) * dAngle
            print(f"Integral of {i} with {j} is {integral}")
            if i == j:
                assert np.isclose(integral, 1, atol=1e-3), f"Integral of {i}  with itself is {integral}, expected 1"
            else:
                assert np.isclose(integral, 0, atol=1e-3), f"Integral of {i} with {j} is {integral}, expected 0"


def test_projector():
    bessel = Bessel_j_radial_int()
    dk = 0.05
    kmax = 5.0
    kx = np.arange(-kmax, kmax + dk, dk)
    nk = len(kx)
    n = 3
    ddk = (dk / (2 * np.pi))**3

    k_grid = np.array(np.meshgrid(kx, kx, kx, indexing='ij')).reshape(3, -1).T
    projector = Projector(k_grid, bessel, a0=n * bohr_radius_angstrom)
    proj_dict = {}
    for orb in ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2']:
        proj = projector(orb)
        proj = proj.reshape((nk, nk, nk))
        proj_dict[orb] = proj

    for k1, p1 in proj_dict.items():
        for k2, p2 in proj_dict.items():
            overlap = np.sum(np.conj(p1) * p2) * ddk
            if k1 == k2:
                print(f"Overlap <{k1}|{k2}> = {overlap}")
                assert np.isclose(overlap, 1.0, atol=0.1), f"Overlap for {k1} is not 1: {overlap}"
            else:
                assert np.isclose(overlap, 0.0, atol=1e-8), f"Overlap between {k1} and {k2} is not zero: {overlap}"
