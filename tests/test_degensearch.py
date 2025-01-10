
import pickle
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
    degen = searcher.find_degen_points(searcher.start_random(num_start_points=50, include_zero=True),
                                       max_num_iter=1000)
    assert len(degen) > 10, "too few degenerate points found"
    maxde = np.max(abs(degen[:, 4]))
    assert maxde < 1e-8, f"gap is {maxde}>=1e-8"
    maxe = np.max(abs(degen[:, 3] - E0))
    assert maxe < 1e-8, f"the energy deviates from {E0} by {maxe}>=1e-8"
    dk = abs(np.linalg.norm(degen[:, :3], axis=1) - k0).max()
    assert dk < 1e-8, f"the k  deviates from {k0} by {dk}>=1e-8"


def test_degensearch_calculator(parallel_ray):
    chiral = wb.models.Chiral(delta=1,hop1=1,hop2=1)
    system = wb.System_PythTB(chiral)
    grid = wb.Grid(system, length=20, NKFFT=8)
    print (f"reciprocal lattice {system.recip_lattice}")
    print (f"real lattice {system.real_lattice}")
    calculators={}
    calculators["degen"] = DegenSearcher(iband=0, thresh=1e-9, gap=1, kmax=1.1, 
                         kstep_max=0.1, resolution=1e-5)
    # degen_searcher = calculators["degen"]
    # try:
    #     serialized = pickle.dumps(degen_searcher)
    #     deserialized = pickle.loads(serialized)
    #     print("DegenSearcher is serializable")
    # except Exception as e:
    #     print(f"DegenSearcher is not serializable: {e}")
    Efermi = np.arange(-10, 10, 0.01)
    calculators["dos"] = wb.calculators.static.DOS(tetra=True,
                                                   Efermi=Efermi)
    results = wb.run(system=system,
           grid=grid,
           calculators=calculators,
           use_irred_kpt=False,
           print_progress_step=1,
           parallel=parallel_ray)
    
    if "dos" in results.results:
        import matplotlib.pyplot as plt
        plt.plot(Efermi, results.results["dos"].data)
        plt.show()
    
    kp = results.results["degen"].dic[0][:,:3]
    for k in kp:
        r=wb.evaluate_k(system, k=k, quantities=["energy"])
        print (k,r, r[1]-r[0])