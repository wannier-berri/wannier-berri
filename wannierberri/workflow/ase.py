import warnings
from ase.utils import writer
from ase.constraints import FixAtoms, FixCartesian
from ase.io.espresso import format_atom_position, kspacing_to_grid, kpts2sizeandoffsets, kpts2ndarray, ibrav_error_message
from ase.io.espresso_namelist.namelist import Namelist
import numpy as np


@writer
def write_espresso_in(fd, atoms, input_data=None, pseudopotentials=None,
                      kspacing=None, kpts=None, koffset=(0, 0, 0), kpoints_array=None,
                      crystal_coordinates=True, additional_cards=None,
                      **kwargs):
    """
    Taken from ASE with some modifications
    Create an input file for pw.x.

    Use set_initial_magnetic_moments to turn on spin, if nspin is set to 2
    with no magnetic moments, they will all be set to 0.0. Magnetic moments
    will be converted to the QE units (fraction of valence electrons) using
    any pseudopotential files found, or a best guess for the number of
    valence electrons.

    Units are not converted for any other input data, so use Quantum ESPRESSO
    units (Usually Ry or atomic units).

    Keys with a dimension (e.g. Hubbard_U(1)) will be incorporated as-is
    so the `i` should be made to match the output.

    Implemented features:

    - Conversion of :class:`ase.constraints.FixAtoms` and
      :class:`ase.constraints.FixCartesian`.
    - ``starting_magnetization`` derived from the ``magmoms`` and
      pseudopotentials (searches default paths for pseudo files.)
    - Automatic assignment of options to their correct sections.

    Not implemented:

    - Non-zero values of ibrav
    - Lists of k-points
    - Other constraints
    - Hubbard parameters
    - Validation of the argument types for input
    - Validation of required options

    Parameters
    ----------
    fd: file | str
        A file to which the input is written.
    atoms: Atoms
        A single atomistic configuration to write to ``fd``.
    input_data: dict
        A flat or nested dictionary with input parameters for pw.x
    pseudopotentials: dict
        A filename for each atomic species, e.g.
        {'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}.
        A dummy name will be used if none are given.
    kspacing: float
        Generate a grid of k-points with this as the minimum distance,
        in A^-1 between them in reciprocal space. If set to None, kpts
        will be used instead.
    kpts: (int, int, int) or dict
        If kpts is a tuple (or list) of 3 integers, it is interpreted
        as the dimensions of a Monkhorst-Pack grid.
        If ``kpts`` is set to ``None``, only the Γ-point will be included
        and QE will use routines optimized for Γ-point-only calculations.
        Compared to Γ-point-only calculations without this optimization
        (i.e. with ``kpts=(1, 1, 1)``), the memory and CPU requirements
        are typically reduced by half.
        If kpts is a dict, it will either be interpreted as a path
        in the Brillouin zone (*) if it contains the 'path' keyword,
        otherwise it is converted to a Monkhorst-Pack grid (**).
        (*) see ase.dft.kpoints.bandpath
        (**) see ase.calculators.calculator.kpts2sizeandoffsets
    koffset: (int, int, int)
        Offset of kpoints in each direction. Must be 0 (no offset) or
        1 (half grid offset). Setting to True is equivalent to (1, 1, 1).
    crystal_coordinates: bool
        Whether the atomic positions should be written to the QE input file in
        absolute (False, default) or relative (crystal) coordinates (True).

    """

    # Convert to a namelist to make working with parameters much easier
    # Note that the name ``input_data`` is chosen to prevent clash with
    # ``parameters`` in Calculator objects
    input_parameters = Namelist(input_data)
    input_parameters.to_nested('pw', **kwargs)

    # Convert ase constraints to QE constraints
    # Nx3 array of force multipliers matches what QE uses
    # Do this early so it is available when constructing the atoms card
    moved = np.ones((len(atoms), 3), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            moved[constraint.index] = False
        elif isinstance(constraint, FixCartesian):
            moved[constraint.index] = ~constraint.mask
        else:
            warnings.warn(f'Ignored unknown constraint {constraint}')
    masks = []
    for atom in atoms:
        # only inclued mask if something is fixed
        if not all(moved[atom.index]):
            mask = ' {:d} {:d} {:d}'.format(*moved[atom.index])
        else:
            mask = ''
        masks.append(mask)

    # Species info holds the information on the pseudopotential and
    # associated for each element
    if pseudopotentials is None:
        pseudopotentials = {}
    species_info = {}
    for species in set(atoms.get_chemical_symbols()):
        # Look in all possible locations for the pseudos and try to figure
        # out the number of valence electrons
        pseudo = pseudopotentials[species]
        species_info[species] = {'pseudo': pseudo}

    # Convert atoms into species.
    # Each different magnetic moment needs to be a separate type even with
    # the same pseudopotential (e.g. an up and a down for AFM).
    # if any magmom are > 0 or nspin == 2 then use species labels.
    # Rememeber: magnetisation uses 1 based indexes
    atomic_species = {}
    atomic_species_str = []
    atomic_positions_str = []

    nspin = input_parameters['system'].get('nspin', 1)  # 1 is the default
    noncolin = input_parameters['system'].get('noncolin', False)
    rescale_magmom_fac = kwargs.get('rescale_magmom_fac', 1.0)
    if any(atoms.get_initial_magnetic_moments()):
        if nspin == 1 and not noncolin:
            # Force spin on
            input_parameters['system']['nspin'] = 2
            nspin = 2

    if nspin == 2 or noncolin:
        # Magnetic calculation on
        for atom, mask, magmom in zip(
                atoms, masks, atoms.get_initial_magnetic_moments()):
            if (atom.symbol, magmom) not in atomic_species:
                # for qe version 7.2 or older magmon must be rescale by
                # about a factor 10 to assume sensible values
                # since qe-v7.3 magmom values will be provided unscaled
                fspin = float(magmom) / rescale_magmom_fac
                # Index in the atomic species list
                sidx = len(atomic_species) + 1
                # Index for that atom type; no index for first one
                tidx = sum(atom.symbol == x[0] for x in atomic_species) or ' '
                atomic_species[(atom.symbol, magmom)] = (sidx, tidx)
                # Add magnetization to the input file
                mag_str = f"starting_magnetization({sidx})"
                input_parameters['system'][mag_str] = fspin
                species_pseudo = species_info[atom.symbol]['pseudo']
                atomic_species_str.append(
                    f"{atom.symbol}{tidx} {atom.mass} {species_pseudo}\n")
            # lookup tidx to append to name
            sidx, tidx = atomic_species[(atom.symbol, magmom)]
            # construct line for atomic positions
            atomic_positions_str.append(
                format_atom_position(
                    atom, crystal_coordinates, mask=mask, tidx=tidx)
            )
    else:
        # Do nothing about magnetisation
        for atom, mask in zip(atoms, masks):
            if atom.symbol not in atomic_species:
                atomic_species[atom.symbol] = True  # just a placeholder
                species_pseudo = species_info[atom.symbol]['pseudo']
                atomic_species_str.append(
                    f"{atom.symbol} {atom.mass} {species_pseudo}\n")
            # construct line for atomic positions
            atomic_positions_str.append(
                format_atom_position(atom, crystal_coordinates, mask=mask)
            )

    # Add computed parameters
    # different magnetisms means different types
    input_parameters['system']['ntyp'] = len(atomic_species)
    input_parameters['system']['nat'] = len(atoms)

    # Use cell as given or fit to a specific ibrav
    if 'ibrav' in input_parameters['system']:
        ibrav = input_parameters['system']['ibrav']
        if ibrav != 0:
            raise ValueError(ibrav_error_message)
    else:
        # Just use standard cell block
        input_parameters['system']['ibrav'] = 0

    # Construct input file into this
    pwi = input_parameters.to_string(list_form=True)

    # Pseudopotentials
    pwi.append('ATOMIC_SPECIES\n')
    pwi.extend(atomic_species_str)
    pwi.append('\n')

    if kpoints_array is not None:
        pwi.append('K_POINTS crystal\n')
        pwi.append(f"{kpoints_array.shape[0]}\n")
        kpoints_array = np.array(kpoints_array)
        if kpoints_array.shape[1] == 3:
            kpoints_array = np.hstack((kpoints_array, np.ones((len(kpoints_array), 1)) / kpoints_array.shape[0]))
        for k in kpoints_array:
            pwi.append(f"{k[0]:.14f} {k[1]:.14f} {k[2]:.14f} {k[3]:.14f}\n")
        pwi.append('\n')
    else:
        # KPOINTS - add a MP grid as required
        if kspacing is not None:
            kgrid = kspacing_to_grid(atoms, kspacing)
        elif kpts is not None:
            if isinstance(kpts, dict) and 'path' not in kpts:
                kgrid, shift = kpts2sizeandoffsets(atoms=atoms, **kpts)
                koffset = []
                for i, x in enumerate(shift):
                    assert x == 0 or abs(x * kgrid[i] - 0.5) < 1e-14
                    koffset.append(0 if x == 0 else 1)
            else:
                kgrid = kpts
        else:
            kgrid = "gamma"

        # True and False work here and will get converted by ':d' format
        if isinstance(koffset, int):
            koffset = (koffset, ) * 3

        # BandPath object or bandpath-as-dictionary:
        if isinstance(kgrid, dict) or hasattr(kgrid, 'kpts'):
            pwi.append('K_POINTS crystal_b\n')
            assert hasattr(kgrid, 'path') or 'path' in kgrid
            kgrid = kpts2ndarray(kgrid, atoms=atoms)
            pwi.append(f'{len(kgrid)}\n')
            for k in kgrid:
                pwi.append(f"{k[0]:.14f} {k[1]:.14f} {k[2]:.14f} 0\n")
            pwi.append('\n')
        elif isinstance(kgrid, str) and (kgrid == "gamma"):
            pwi.append('K_POINTS gamma\n')
            pwi.append('\n')
        else:
            pwi.append('K_POINTS automatic\n')
            pwi.append(f"{kgrid[0]} {kgrid[1]} {kgrid[2]} "
                    f" {koffset[0]:d} {koffset[1]:d} {koffset[2]:d}\n")
            pwi.append('\n')

    # CELL block, if required
    if input_parameters['SYSTEM']['ibrav'] == 0:
        pwi.append('CELL_PARAMETERS angstrom\n')
        pwi.append('{cell[0][0]:.14f} {cell[0][1]:.14f} {cell[0][2]:.14f}\n'
                   '{cell[1][0]:.14f} {cell[1][1]:.14f} {cell[1][2]:.14f}\n'
                   '{cell[2][0]:.14f} {cell[2][1]:.14f} {cell[2][2]:.14f}\n'
                   ''.format(cell=atoms.cell))
        pwi.append('\n')

    # Positions - already constructed, but must appear after namelist
    if crystal_coordinates:
        pwi.append('ATOMIC_POSITIONS crystal\n')
    else:
        pwi.append('ATOMIC_POSITIONS angstrom\n')
    pwi.extend(atomic_positions_str)
    pwi.append('\n')

    # DONE!
    fd.write(''.join(pwi))

    if additional_cards:
        if isinstance(additional_cards, list):
            additional_cards = "\n".join(additional_cards)
            additional_cards += "\n"

        fd.write(additional_cards)
