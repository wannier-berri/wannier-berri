import numpy as np
import wannierberri as wberri
from pathlib import Path
from wannierberri.system.system_soc import SystemSOC
from wannierberri.calculators.static import Torkance

"""
Example use of wannierberri to compute the spin-orbit torque (SOT) torkance for bcc Fe 
using precomputed data from GPAW.
Gives 0.0 for the torkance due to inversion symmetry.
"""

def run_fe_sot_example():
    HERE = Path(__file__).parent.resolve()
    DATA_DIR = HERE / "../../tests/data/Fe_system_soc"
    system_soc = SystemSOC.from_npz(str(DATA_DIR / "system_soc"))
    
    # Torkance Calculation
    grid = wberri.Grid(system_soc, length=20)
    Efermi = np.linspace(-2.0, 2.0, 5)

    torkance_calc = Torkance(Efermi=Efermi, Emin=-2.0, Emax=2.0)

    # Run
    results = wberri.run(
        system_soc, 
        grid=grid, 
        calculators={"torkance": torkance_calc},
        use_irred_kpt=False,  
        symmetrize=False      
    ).results["torkance"]
    
    # Printing
    print("Torkance Results:\n")
    E_f = results.Energies[0]
    torkance_tensor = results.data  # Shape: (len(E_f), 3, 3)
    # Optional: clean up numpy's complex number formatting for terminal output
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    for i, e in enumerate(E_f):
        print(f"--- Fermi Energy: {e:.4f} eV ---")
        print("Full 3x3 Tensor:")
        print(torkance_tensor[i])        
        print()

    np.save("ref_torkance.npy", results.data)

if __name__ == "__main__":
    run_fe_sot_example()