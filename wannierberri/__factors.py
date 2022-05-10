from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom  #, Boltzmann

TAU_UNIT = 1E-9  # tau in nanoseconds
TAU_UNIT_TXT = "ns"

bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom

fac_ahc = -1e8 * elementary_charge**2 / hbar
fac_spin_hall = fac_ahc * -0.5
factor_ohmic = (
    elementary_charge / Ang_SI
    / hbar**2  # first, transform to SI, not forgeting hbar in velocities - now in  1/(kg*m^3)
    * elementary_charge**2
    * TAU_UNIT  # multiply by a dimensional factor - now in A^2*s^2/(kg*m^3*tau_unit) = S/(m*tau_unit)
    * 1e-2)  # now in  S/(cm*tau_unit)
factor_Hall_classic = elementary_charge**2 * Ang_SI / hbar**3  # first, transform to SI, not forgeting hbar in velocities - now in  m/(J*s^3)
factor_Hall_classic *= elementary_charge**3 / hbar * TAU_UNIT**2  # multiply by a dimensional factor - now in A^3*s^5*cm/(J^2*tau_unit^2) = S/(T*m*tau_unit^2)
factor_Hall_classic *= 1e-2  #  finally transform to S/(T*cm*tau_unit^2)
