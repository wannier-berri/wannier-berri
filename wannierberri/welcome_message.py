_need_for_symmetry_str = "symmetry-related functionality (SAWF, symmetrization, projections, …)"
_needed_packages = {
    "irrep": "symmetry related ",
    "spglib": _need_for_symmetry_str,
    "numpy": "ESSENTIAL",
    "scipy": "ESSENTIAL",
    "spgrep": "projections searcher",
    "numba": "tetrahedron integration",
    "pyfftw": "fast Fourier transforms (optional, otherwise uses numpy's FFT)",
    "seekpath": "automatic generation of k-point paths",
    "matplotlib": "plotting",
    "sympy": _need_for_symmetry_str,
    "fortio": "reading Fortran unformatted files (uHu, chk, spn, unk, …)",
    "gpaw": "interface with GPAW",
    "ase": "interface with ASE and GPAW",
    "pythtb": "interface with PythTB (optional)",
    "xmltodict": "reading QuantumEspresso dynamical matrices for phonons",
}


def welcome():
    from importlib.metadata import version as get_package_version, PackageNotFoundError
    from termcolor import cprint
    from . import __version__ as wb_version
    # originally obtained by pyfiglet, font='cosmic'
    # with small modifications
    logo = """
.::    .   .::: .:::::::.  :::.    :::.:::.    :::. :::.,::::::  :::::::..       :::::::.  .,::::::  :::::::..   :::::::..   :::
';;,  ;;  ;;;' '  ;;`;;  ` `;;;;,  `;;;`;;;;,  `;;; ;;;;;;;''''  ;;;;``;;;;       ;;;'';;' ;;;;''''  ;;;;``;;;;  ;;;;``;;;;  ;;;
 '[[, [[, [['    ,[[ '[[,    [[[[[. '[[  [[[[[. '[[ [[[ [[cccc    [[[,/[[['       [[[__[[\\. [[cccc    [[[,/[[['   [[[,/[[['  [[[
   Y$c$$$c$P    c$$$cc$$$c   $$$ "Y$c$$  $$$ "Y$c$$ $$$ $$\"\"\"\"    $$$$$$c         $$\"\"\"\"Y$$ $$\"\"\"\"    $$$$$$c     $$$$$$c    $$$
    "88"888      888   888,  888    Y88  888    Y88 888 888oo,__  888b "88bo,    _88o,,od8P 888oo,__  888b "88bo, 888b "88bo,888
     "M "M"      YMM   ""`   MMM     YM  MMM     YM MMM \"\"\"\"YUMMM MMMM   "W"     ""YUMMMP"  \"\"\"\"YUMMM MMMM   "W"  MMMM   "W" MMM
"""
    cprint(logo, 'green')
    cprint(f"Version: {wb_version}\n", 'cyan', attrs=['bold'])
    cprint("""\n   HTTP://WANNIER-BERRI.ORG  \n""", 'yellow')

    print("Checking dependencies …")
    versions = {}
    for package in _needed_packages.keys():
        try:
            mod = __import__(package)
            # Try module's __version__ first, then fall back to package metadata
            version = getattr(mod, '__version__', None)
            if version is None:
                try:
                    version = get_package_version(package)
                except PackageNotFoundError:
                    version = 'unknown'
            cprint(f"{package} : {version}", 'cyan')
            versions[package] = version
        except ImportError:
            nfor = _needed_packages.get(package, "No description available")
            if nfor == "ESSENTIAL":
                cprint(f"{package} : not found. {nfor}. Please install it to use wannierberri.", 'red')
            else:
                cprint(f"{package} : not found. {nfor}", 'yellow')
            versions[package] = None
    print("#")
    return versions
