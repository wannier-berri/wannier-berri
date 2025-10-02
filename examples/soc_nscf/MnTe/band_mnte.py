from gpaw import GPAW

calc = GPAW("scf_norelax.gpw").fixed_density(
    symmetry="off",
    kpts={"path": "KGAKHML", "npoints": 240},
    nbands=60,
)
calc.write("band.gpw", mode="all")
