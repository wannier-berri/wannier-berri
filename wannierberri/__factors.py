from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom  #, Boltzmann

TAU_UNIT = 1E-9  # tau in nanoseconds
TAU_UNIT_TXT = "ns"

bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom

#fac_morb_Z = elementary_charge/2/hbar * Ang_SI**2 # change unit of m_orb*B to (eV).
#fac_spin_Z = elementary_charge * hbar / (2 * electron_mass) / Ang_SI**2# change unit of m_spin*B to (eV).

factor_gme = -(elementary_charge / Ang_SI**2  # with tau^0 E^0 B^1
                * elementary_charge / hbar) # change velocity unit from eV*m to m/s
factor_ahc = -(elementary_charge**2 / hbar / Ang_SI) /100.   # with tau^0 E^1 B^0
factor_ohmic = (elementary_charge**2 / hbar / Ang_SI * TAU_UNIT /100.   # with tau^1 E^1 B^1
                * elementary_charge / hbar) # change velocity unit from eV*m to m/s
factor_nlahc = elementary_charge**3 /hbar**2 * TAU_UNIT    # with tau^1 E^2 B^0
factor_hall_classic = -(elementary_charge**3 /hbar**2 * Ang_SI * TAU_UNIT**2 /100.  # with tau^2 E^1 B^1
                * elementary_charge**2 / hbar**2) # change velocity unit from eV*m to m/s
factor_nldrude = -(elementary_charge**3 /hbar**2 * TAU_UNIT**2  # with tau^2 E^2 B^0
                * elementary_charge / hbar) # change velocity unit from eV*m to m/s

