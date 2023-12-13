from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom, pi  # , Boltzmann

TAU_UNIT = 1E-15  # tau in nanoseconds
TAU_UNIT_TXT = "fs"

bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ry_eV = physical_constants['Rydberg constant times hc in eV'][0]  # =13.605693122994
Ang_SI = angstrom

fac_orb_Z = elementary_charge / 2 / hbar * Ang_SI**2  # change unit of m_orb*B to (eV).
fac_spin_Z = hbar / (2 * electron_mass)  # change unit of m_spin*B to (eV).

factor_gme = -(elementary_charge / Ang_SI**2)  # for gme tensor
factor_ahc = -(elementary_charge**2 / hbar / Ang_SI)  # with tau^0 E^1 B^0
factor_ohmic = (elementary_charge**2 / hbar / Ang_SI * TAU_UNIT *  # with tau^1 E^1 B^1
                elementary_charge / hbar)  # change velocity unit from eV*m to m/s
factor_nlahc = elementary_charge**3 / hbar**2 * TAU_UNIT  # with tau^1 E^2 B^0
factor_hall_classic = -(elementary_charge**3 / hbar**2 * Ang_SI * TAU_UNIT**2 *  # with tau^2 E^1 B^1
                elementary_charge**2 / hbar**2)  # change velocity unit from eV*m to m/s
factor_nldrude = -(elementary_charge**3 / hbar**2 * TAU_UNIT**2 *  # with tau^2 E^2 B^0
                elementary_charge / hbar)  # change velocity unit from eV*m to m/s

factor_opt = -factor_ahc
factor_shc = -factor_ahc
#####################
# for old_API
fac_ahc = factor_ahc
factor_Hall_classic = elementary_charge**2 * Ang_SI / hbar**3  # first, transform to SI, not forgeting hbar in velocities - now in  m/(J*s^3)
factor_Hall_classic *= elementary_charge**3 / hbar * TAU_UNIT**2  # multiply by a dimensional factor - now in A^3*s^5*cm/(J^2*tau_unit^2) = S/(T*m*tau_unit^2)
fac_spin_hall = factor_ahc * -0.5
factor_shift_current = 1j * hbar / elementary_charge * pi * elementary_charge**3 / (4 * hbar**2)
factor_injection_current = - pi * elementary_charge**3 / (hbar**2) * TAU_UNIT
