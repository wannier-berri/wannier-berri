from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom, pi

TAU_UNIT = 1E-15  # tau in nanoseconds
TAU_UNIT_TXT = "fs"

bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ry_eV = physical_constants['Rydberg constant times hc in eV'][0]  # =13.605693122994
electron_g_factor = physical_constants['electron g factor'][0]

factor_morb_evA2_to_muB = -(elementary_charge * angstrom ** 2) * elementary_charge / (2 * hbar) / bohr_magneton  # change unit of morb from eV*angstrom^2 to mu_B
factor_morb_evA2_to_muB_2 = -elementary_charge * electron_mass * angstrom ** 2 / hbar ** 2
assert abs(factor_morb_evA2_to_muB - factor_morb_evA2_to_muB_2) < 1E-8, "factor_morb_evA2_to_muB and factor_morb_evA2_to_muB_2 should be the same"

fac_orb_Z = elementary_charge / 2 / hbar * angstrom ** 2  # change unit of m_orb*B to (eV).
fac_spin_Z = hbar / (2 * electron_mass)  # change unit of m_spin*B to (eV).

factor_gme = -(elementary_charge / angstrom ** 2)  # for gme tensor
factor_gme_orb = factor_morb_evA2_to_muB * bohr_magneton / angstrom ** 2  # for orbital part of gme tensor
assert abs(factor_gme * fac_orb_Z - factor_gme_orb) < 1E-8, f"factor_gme and factor_gme_orb should be the same, the ratio is {factor_gme * fac_orb_Z / factor_gme_orb}"
factor_gme_spin = -bohr_magneton / angstrom ** 2
assert abs(factor_gme * fac_spin_Z - factor_gme_spin) < 1E-8, f"factor_gme and factor_gme_spin should be the same, the ratio is {factor_gme * fac_spin_Z / factor_gme_spin}"

factor_ahc = -(elementary_charge ** 2 / hbar / angstrom)  # with tau^0 E^1 B^0
factor_ohmic = (elementary_charge ** 2 / hbar / angstrom * TAU_UNIT *  # with tau^1 E^1 B^1
                elementary_charge / hbar)  # change velocity unit from eV*m to m/s
factor_nlahc = elementary_charge ** 3 / hbar ** 2 * TAU_UNIT  # with tau^1 E^2 B^0
factor_hall_classic = -(elementary_charge ** 3 / hbar ** 2 * angstrom * TAU_UNIT ** 2 *  # with tau^2 E^1 B^1
                        elementary_charge ** 2 / hbar ** 2)  # change velocity unit from eV*m to m/s
factor_nldrude = -(elementary_charge ** 3 / hbar ** 2 * TAU_UNIT ** 2 *  # with tau^2 E^2 B^0
                   elementary_charge / hbar)  # change velocity unit from eV*m to m/s
factor_emcha = -(elementary_charge**4 / hbar**3 * angstrom**2 * TAU_UNIT**2 *  # with tau^2 E^2 B^1
                elementary_charge / hbar)  # change velocity unit from eV*m to m/s

factor_opt = -factor_ahc
factor_shc = -factor_ahc

#########
# Oscar #
###############################################################################
factor_SDCT = elementary_charge ** 2 / hbar
###############################################################################

#####################
factor_Hall_classic = elementary_charge ** 2 * angstrom / hbar ** 3  # first, transform to SI, not forgeting hbar in velocities - now in  m/(J*s^3)
factor_Hall_classic *= elementary_charge ** 3 / hbar * TAU_UNIT ** 2  # multiply by a dimensional factor - now in A^3*s^5*cm/(J^2*tau_unit^2) = S/(T*m*tau_unit^2)

# (stepan) I removed the 1j factor for shift current and took the -Im part of I_mn
factor_shift_current = hbar / elementary_charge * pi * elementary_charge ** 3 / (4 * hbar ** 2)
factor_injection_current = - pi * elementary_charge ** 3 / (hbar ** 2) * TAU_UNIT

## for SDCT
m_spin_prefactor_SI = -0.5 * electron_g_factor * hbar / electron_mass
#  "m_spin_prefactor_SI" converts to SI units, which are units of
# hbar/m, i.e. m^2/s

# While the B_M1 is computed with internal units (and without the hbar factor),
# which would be eV*Ang^2
# So, to bring that quantity to SI units we would need to multiply by
# elementary_charge*1e-20 (convert to J*m^2) and divide by hbar (in J*s). So, to
# bring the spin contributions to the same units as the orbital part, we need to
# adjust it to a factor

# hbar*1e20/elementary_charge ~= 65821.19569
#
m_spin_prefactor = m_spin_prefactor_SI * (hbar * 1e20 / elementary_charge)
