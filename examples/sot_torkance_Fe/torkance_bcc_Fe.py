import numpy as np
import wannierberri as wberri
from pathlib import Path
from gpaw import GPAW

from wannierberri.system import System_R
from wannierberri.system.system_soc import SystemSOC
from wannierberri.w90files.soc import SOC
from wannierberri.w90files.chk import CheckPoint as CHK
from wannierberri.calculators.static import TorkanceEven, TorkanceOdd

def run_fe_sot_example_and_generate_refs():
    HERE = Path(__file__).parent.resolve()
    DATA_DIR = HERE / "../../tests/data/Fe_gpaw"
    
    print("Constructing SystemSOC from collinear components...")
    system_up = System_R.from_npz(str(DATA_DIR / "system_up"))
    system_dw = System_R.from_npz(str(DATA_DIR / "system_dw"))
    
    cell = dict(positions=[[0, 0, 0]], typat=[1], magmoms_on_axis=[1])
    system_soc = SystemSOC(system_up=system_up, system_down=system_dw, cell=cell, silent=True)

    print("Extracting SOC matrix from GPAW wavefunctions...")
    calc_nscf = GPAW(str(DATA_DIR / 'Fe-nscf.gpw'), txt=None)
    soc = SOC.from_gpaw(calc_nscf)
    
    selected_bands_up = np.load(DATA_DIR / "system_up.chk.npz")['selected_bands']
    selected_bands_down = np.load(DATA_DIR / "system_dw.chk.npz")['selected_bands']
    soc.select_bands(selected_bands_up=selected_bands_up, selected_bands_down=selected_bands_down)

    chk_up = CHK.from_npz(str(DATA_DIR / "system_up.chk.npz"))
    chk_down = CHK.from_npz(str(DATA_DIR / "system_dw.chk.npz"))

    print("Integrating SOC into Wannier basis...")
    system_soc.set_soc_R(soc=soc, chk_up=chk_up, chk_down=chk_down)
    
    theta_deg, phi_deg = 49, 33
    system_soc.set_soc_axis(theta=theta_deg, phi=phi_deg, units="degrees", alpha_soc=1.0)
    
    print(f"Setting SOT operators for theta={theta_deg}, phi={phi_deg}...")
    system_soc.set_torque_operators_R(theta=theta_deg, phi=phi_deg, units="degrees")
    
    # ---------------------------------------------------------
    # CALCULATOR EVALUATION
    # ---------------------------------------------------------
    print("Setting up grid and calculators...")
    grid = wberri.Grid(system_soc, length=20)
    Efermi = np.linspace(-2.0, 2.0, 5)

    calculators = {
        "even": TorkanceEven(Efermi=Efermi, Emin=-2.0, Emax=2.0),
        "odd": TorkanceOdd(Efermi=Efermi, Emin=-2.0, Emax=2.0)
    }

    print("Running WannierBerri BZ integration...")
    results = wberri.run(
        system_soc, 
        grid=grid, 
        calculators=calculators,
        use_irred_kpt=False,  
        symmetrize=False      
    ).results
    
    # Print output
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    for component in ["even", "odd"]:
        print(f"\n================ Torkance {component.upper()} ================")
        res = results[component]
        for i, e in enumerate(res.Energies[0]):
            print(f"--- Fermi Energy: {e:.4f} eV ---")
            print(res.data[i])
            
    # ---------------------------------------------------------
    # REFERENCE DATA EXPORT (for pytest regression tests, not needed for normal example usage)
    # ---------------------------------------------------------
    """
    print("\nSaving reference arrays to current directory...")
    
    # 1. Save the real-space SOT operator matrix
    # Important: test_system.py expects compressed npz with the default 'arr_0' key
    SOT_R = system_soc.get_R_mat('SOT')
    np.savez_compressed("SOT.npz", SOT_R)
    
    # 2. Save the integrated calculator tensors
    np.save("ref_torkance_even.npy", results["even"].data)
    np.save("ref_torkance_odd.npy", results["odd"].data)
    
    print("Done! You can now move SOT.npz to tests/reference/systems/Fe_gpaw_soc_theta49.00_phi33.00_alpha1.00_symmetrized/")
    print("And move the .npy files to tests/reference/integrate_files/")
    """

if __name__ == "__main__":
    run_fe_sot_example_and_generate_refs()