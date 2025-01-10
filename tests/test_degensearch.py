
import numpy as np
from wannierberri.calculators.degensearch import DegenSearcher, DegenSearcherKP
import wannierberri as wb


def test_degensearch_simple():
    ndim = 3
    nbands = 4


    H0 = np.random.random((nbands, nbands)) + 1j * np.random.random((nbands, nbands))
    H1 = np.random.random((nbands, nbands, ndim)) + 1j * np.random.random((nbands, nbands, ndim))
    H2 = np.random.random((nbands, nbands, ndim, ndim)) + 1j * np.random.random((nbands, nbands, ndim, ndim))

    H0 = np.zeros((nbands, nbands))
    H1 = np.zeros((nbands, nbands, ndim))
    H2 = np.zeros((nbands, nbands, ndim, ndim))
    H0[0, 0] = -1
    H0[1, 1] = -1
    H0[2, 2] = 1
    H0[3, 3] = 1
    for i in range(ndim):
        H2[1, 1, i, i] = 5
        H2[2, 2, i, i] = -2
    k0 = np.sqrt(2 / 7)
    E0 = -1 + k0**2 * 5

    searcher = DegenSearcherKP(H0=H0, H1=H1, H2=H2, degen_thresh=1e-8, kmax=1, kstep_max=0.1, iband=1)
    degen = searcher.find_degen_points(searcher.start_random(num_start_points=100, include_zero=True),
                                       max_num_iter=1000)
    assert len(degen) > 10, "too few degenerate points found"
    maxde = np.max(abs(degen[:, 4]))
    assert maxde < 1e-8, f"gap is {maxde}>=1e-8"
    maxe = np.max(abs(degen[:, 3] - E0))
    assert maxe < 1e-8, f"the energy deviates from {E0} by {maxe}>=1e-8"
    dk = abs(np.linalg.norm(degen[:, :3], axis=1) - k0).max()
    assert dk < 1e-8, f"the k  deviates from {k0} by {dk}>=1e-8"


def test_degensearch_calculator(system_Chiral_left):
    system = system_Chiral_left
    grid = wb.Grid(system, NKFFT=3, NKdiv=2)
    calc = DegenSearcher(iband=0, thresh=1e-8, gap=1, kmax=1, kstep_max=0.1)
    wb.run(system=system,
           grid=grid,
           calculators={"degen": calc})
