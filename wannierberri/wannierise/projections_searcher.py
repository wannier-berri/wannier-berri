import numpy as np
from termcolor import cprint
from ..__utility import UniqueList
from ..w90files.Dwann import Dwann
from .utility import select_window_degen

from .projections import ORBITALS, ProjectionsSet


class EBRsearcher:
    """
    A class  to search for EBRs that cover the frozen window, but fit into the outer window

    Parameters
    ----------
    spacegroup : irrep.SpaceGroup
        A space group object.
    win : wannierberri.w90files.WIN
        The object containing the win file (just for k-points for now)
    eig : wannierberri.w90files.EIG
        The file of eigenvalues
    dmn : wannierberri.w90files.DMN
        The file of symmetry properties (only the band part is used, the wann part may be anything or empty)
    froz_min, froz_max : float
        The minimum and maximum energy of the frozen window
    outer_min, outer_max : float
        The minimum and maximum energy of the outer window
    degen_thresh : float
        The threshold to consider bands as degenerate
    trial_positions : list of np.ndarray(shape=(3,), dtype=float)
        The trial projections to be tested
    trial_orbitals : list of str
        The orbitals of the trial projections, e.g. 's', 'p', 'sp3', etc.
    trial_projections : list of Projection
        alternaitvely, may be specified instead as a list of Projection objects
        appended to trial_positions and trial_orbitals (if they are not empty)
    debug : bool
        If True, print debug information

    Attributes
    ----------
    eig : np.ndarray(shape=(NKirr,NB), dtype=float)
        The eigenvalues of the bands at irreducible k-points
    Dwann_list : list of np.array(shape=(NKirr,nsym,NWi,NWi), dtype=complex)
        The list of D matriced transforming the Wannier-gauge Bloch functiuons
    nsym_little : list of int (length=NKirr)
        The number of little group symmetries for each irreducible k-point

    Notes
    -----
    * So far it does not work with time-reversal symmetry, and was not tested with spinor wavefuncitons.
      It is recommended to search for projections on a scalar calculation (no spin). 
      The projections found this way are usually good for spinful calculations as well.
    """

    def __init__(self, spacegroup,
                 win, eig, dmn,
                 froz_min=np.inf, froz_max=-np.inf,
                 outer_min=-np.inf, outer_max=np.inf,
                 degen_thresh=1e-8,
                 trial_positions=None,
                 trial_orbitals=None,
                 trial_projections=None,
                 debug=False
                ):
        if spacegroup.spinor:
            raise NotImplementedError("EBRsearcher does not work with spinors")
        if trial_positions is None:
            trial_positions = []
        if trial_orbitals is None:
            trial_orbitals = []
        assert len(trial_positions) == len(trial_orbitals), "The number of trial positions and orbitals must be the same"
        self.eig = eig.data[dmn.kptirr]
        self.Dwann_list = []
        self.NKirr = dmn.NKirr
        self.nsym_little = [len(l) for l in dmn.isym_little]
        self.debug = debug

        # if projections are specified as a list of Projection objects (overriding trial_positions and trial_orbitals)
        if trial_projections is not None:
            if isinstance(trial_projections, ProjectionsSet):
                trial_projections = trial_projections.projections
            for projs in trial_projections:
                for proj in projs.split():
                    trial_positions.append(proj.positions)
                    trial_orbitals.append(proj.orbitals[0])
        self.num_trial_proj = len(trial_positions)

        # list of list of UniqueList of Irreps for each projection [iproj][ik]
        def highlight(line):
            if self.debug:
                cprint("#" * 80 + "\n#" + " " * 5 + f"{line}\n" + "#" * 80, color="yellow", attrs=['bold'])

        def debug_msg(msg=""):
            if self.debug:
                print("DEBUG: " + msg)
        
        highlight("detecting irreps genberated by eacg projections")

        self.irreps_per_projection = []
        self.num_wann_per_projection = []
        for wyckoff, orbital in zip(trial_positions, trial_orbitals):
            debug_msg(f"detecting irreps by position {wyckoff} and orbital {orbital}")
            dwann = Dwann(spacegroup=spacegroup,
                          positions=wyckoff,
                          orbital=orbital,
                          ORBITALS=ORBITALS).get_on_points_all(kpoints=win.data['kpoints'],
                                                           ikptirr=dmn.kptirr,
                                                           ikptirr2kpt=dmn.kptirr2kpt)
            self.num_wann_per_projection.append(dwann.shape[2])
            self.Dwann_list.append(dwann)
            irrs = []
            for ik, D in enumerate(dwann):
                irr = get_irreps_wann(D[dmn.isym_little[ik]])
                if self.debug:
                    debug_msg(f"kpoint={ik} has the following irreducible representations:")
                    for i, ir in enumerate(irr):
                        debug_msg(f"{i} : {ir}")
                    debug_msg()
                irrs.append(irr)
            debug_msg()
            self.irreps_per_projection.append(irrs)


            # self.irreps_per_projection.append([get_irreps_wann(D[dmn.isym_little[ik]])
            #                                    for ik,D in enumerate(dwann)])

        debug_msg(f"num_wann_per_projection={self.num_wann_per_projection}")

        highlight("collecting all possible irreps")        
        # for each k-point collect all possible irreps that are generated by the trial_projections
        self.all_possible_irreps = []
        for ik in range(self.NKirr):
            irreps = UniqueList(count=False)
            for irreps_proj in self.irreps_per_projection:
                for ir in irreps_proj[ik]:
                    irreps.append(ir)
            self.all_possible_irreps.append(irreps)
            if self.debug:
                debug_msg(f"The following possible irreps can be generated at ikirr={ik} by the given projections:")
                for i, irr in enumerate(irreps):
                    debug_msg(f"    {i} : {irr}")
                debug_msg()

        # Now for each k-point characterize each projection as a vector
        # of the number of times each irrep (from all possible ones) appears in the projection
        # [ik][iproj]
        self.irreps_per_projection_vectors = []
        for ik in range(self.NKirr):
            irreps = np.zeros((self.num_trial_proj, len(self.all_possible_irreps[ik]),), dtype=int)
            for j, irreps_proj in enumerate(self.irreps_per_projection):
                for i, ir in enumerate(irreps_proj[ik]):
                    irreps[j, self.all_possible_irreps[ik].index(ir)] += irreps_proj[ik].counts[i]
            self.irreps_per_projection_vectors.append(irreps)


        highlight(" deternine the irreps in the DFT bands")
        self.irreps_frozen_vectors = []
        self.irreps_outer_vectors = []
        for ik in range(self.NKirr):
            frozen = select_window_degen(self.eig[ik], thresh=degen_thresh,
                                     win_min=froz_min, win_max=froz_max, return_indices=True)
            outer = select_window_degen(self.eig[ik], thresh=degen_thresh,
                                     win_min=outer_min, win_max=outer_max, return_indices=True)
            nfrozen = len(frozen)
            char_outer_conj = np.array([dmn.d_band_diagonal(ik, isym)[outer].sum() for isym in dmn.isym_little[ik]]).conj()
            char_frozen_conj = np.array([dmn.d_band_diagonal(ik, isym)[frozen].sum() for isym in dmn.isym_little[ik]]).conj()
            debug_msg(f"ik= {ik} contains {nfrozen} frozen states\n" + 
                  f"the little group contains {len(dmn.isym_little[ik])} symmetries: \n {dmn.isym_little[ik]}\n" +
                  f"characters in outer window : {np.round(char_outer_conj,3)}\n" +
                  f"characters in frozen window: {np.round(char_frozen_conj,3)}") 
            # if self.debug:
            #     print(f"""all eigenvalues = \n{np.array([
            #         dmn.d_band_diagonal(ik,isym) for isym in dmn.isym_little[ik]]).round(4).T} """)
            vector_frozen = np.zeros(len(self.all_possible_irreps[ik]), dtype=float)
            vector_outer = np.zeros(len(self.all_possible_irreps[ik]), dtype=float)

            for i, irrep in enumerate(self.all_possible_irreps[ik]):
                vector_frozen[i] = np.dot(irrep.characters, char_frozen_conj).real / self.nsym_little[ik]
                vector_outer[i] = np.dot(irrep.characters, char_outer_conj).real / self.nsym_little[ik]
            vector_frozen = np.ceil(np.round(vector_frozen, 3)).astype(int)
            vector_outer = np.ceil(np.round(vector_outer, 3)).astype(int)
            self.irreps_frozen_vectors.append(vector_frozen)
            self.irreps_outer_vectors.append(vector_outer)
            debug_msg(f"Frozen states are represented by vector {vector_frozen}")
            debug_msg(f"Outer states are represented by vector {vector_outer}")
            
            nband_frozen_irreps = np.round(sum(v * irrep.characters[0]
                                      for v, irrep in zip(vector_frozen, self.all_possible_irreps[ik])), 3)
            if nband_frozen_irreps < nfrozen:
                raise RuntimeError("Some frozen bands are not covered by irreps induced by trial projections"
                       "try adding more projections"
                       f"ik={ik} nfrozen={nfrozen} nband_frozen_irreps={nband_frozen_irreps}")

        if self.debug:
            for i in range(self.NKirr):
                debug_msg(f"ik={i} \n frozen={self.irreps_frozen_vectors[i]} \n"
                    f" outer={self.irreps_outer_vectors[i]} \n")
                for j in range(self.num_trial_proj):
                    debug_msg(f"  j={j} {self.irreps_per_projection_vectors[i][j]}")

    def find_combinations(self, max_num_wann=None, fixed=[]):
        """
        find all possible combinations of trial projections that cover all the irreps inside the frozen window
        and fit into the outer window

        Parameters
        ----------
        max_num_wann : int
            The maximum number of wannier functions
        fixed : list of int
            The indices of the trial projections that are fixed and should be taken exactly once

        Returns
        -------
        combinations : np.ndarray(shape=(K,M), dtype=int)
            The K combinations of coefficients that denote multip[licity of each irrep
        """

        lfixed = np.zeros(self.num_trial_proj, dtype=bool)
        lfixed[np.array(fixed, dtype=int)] = True
        combinations = find_combinations_max(vectors=self.irreps_per_projection_vectors[0],
                                             vector_max=self.irreps_outer_vectors[0],
                                             max_num_wann=max_num_wann,
                                             num_wann_per_projection=self.num_wann_per_projection,
                                             lfixed=lfixed,
                                                )
        if self.debug:
            print(f"From first point within outer window {len(combinations)} are valid")
            print(combinations)
        for ik in range(self.NKirr):
            combinations = check_combinations_min_max(combinations=combinations,
                                                      vectors=self.irreps_per_projection_vectors[ik],
                                                      vector_min=self.irreps_frozen_vectors[ik],
                                                      vector_max=self.irreps_outer_vectors[ik])

            if self.debug:
                print(f"From ik={ik}/{self.NKirr} with frozen and outer window {len(combinations)} are valid")
        return combinations




def get_irreps_bands(symmetry_matrices, energies, froz_min=-np.inf, froz_max=np.inf, thresh=1e-8):
    """
    Get the irreducible representations of the symmetry matrices

    Parameters
    ----------
    symmetry_matrices : list of np.ndarray(shape=(Nw,Nw), dtype=complex)
        The symmetry matrices of the spacegroup
    TR : bool
        If True, the time-reversal symmetry is included

    Returns
    -------
    characters : np.ndarray(shape=(Nirr,Nw), dtype=complex)
        The irreducible representations of the symmetry matrices
    multiplicity : np.ndarray(shape=(Nirr,), dtype=int)
        The multiplicity of each irreducible representation
    """
    Nsym, NB, NB1 = symmetry_matrices.shape
    assert NB == NB1, "The symmetry matrices must be square"
    # generate a random Hamiltonian
    # get the eigenvalues of the symmetry matrices
    # group by degenerate bands
    print(energies)
    borders = [0] + list(np.where(np.abs(np.diff(energies)) > 1e-4)[0] + 1) + [NB]
    print(borders)
    return UniqueList([Irrep([S[np.arange(b1, b2), np.arange(b1, b2)].sum()
                              for S in symmetry_matrices])
                       for b1, b2 in zip(borders[:-1], borders[1:])], count=True)





def get_irreps_wann(symmetry_matrices):
    """
    Get the irreducible representations of the symmetry matrices

    Parameters
    ----------
    symmetry_matrices : list of np.ndarray(shape=(Nw,Nw), dtype=complex)
        The symmetry matrices of the spacegroup
    TR : bool
        If True, the time-reversal symmetry is included

    Returns
    -------
    characters : unique_list of Irrep
        The irreducible representations of the symmetry matrices and thjeit multiplicities
    """
    Nsym, Nw, Nw1 = symmetry_matrices.shape
    assert Nw == Nw1, "The symmetry matrices must be square"
    # generate a random Hamiltonian
    H = np.random.rand(Nw, Nw) + 1j * np.random.rand(Nw, Nw)
    H = (H + H.T.conj()) / 2
    # symmetrize it
    H = sum(sym.T.conj() @ H @ sym for sym in symmetry_matrices) / Nsym
    # get the eigenvalues
    E, V = np.linalg.eigh(H)
    # get the eigenvalues of the symmetry matrices
    characters = np.array([(V.T.conj() @ sym @ V).diagonal() for sym in symmetry_matrices]).T
    # group by degenerate bands
    borders = [0] + list(np.where(np.abs(np.diff(E)) > 1e-5)[0] + 1) + [Nw]
    characters = [characters[b1:b2].sum(axis=0) for b1, b2 in zip(borders[:-1], borders[1:])]
    # print ("E=",E, "borders=",borders , "characters=", characters)
    return UniqueList([Irrep(char) for char in characters], count=True)


def group_by_TR(irreps):
    """
    Group the irreps by time-reversal symmetry

    Parameters
    ----------
    irreps : list of Irrep
        The irreducible representations

    Returns
    -------
    irreps_TR : list of Irrep
        The irreducible representations grouped by time-reversal symmetry
    """	

    # pair by time-reversal symmetry
    irreps_TR = UniqueList(count=True)
    while len(irreps) > 0:
        irr = irreps[0]
        irrconj = irr.conj()
        assert irrconj in irreps, f"the conjugate of irr1={irr}, is not present in irreps, but time-reversal is required"
        i = irreps.index(irrconj)
        assert irreps.counts[i] == irreps.counts[-1], f"the conjugate of irr1={irr}, is not present in irreps, but time-reversal is required"
        if i == 0:
            irreps_TR.append(irr, count=irreps.counts[0])
            irreps.remove(irr, all=True)
        else:
            irreps_TR.append(irr + irrconj, count=irreps.counts[0])
            irreps.remove(irr, all=True)
            irreps.remove(irrconj, all=True)
    return irreps_TR


class Irrep:
    """a class to store irreducible representation, 
    represented by a list of its characters 
    allows for approximate comparison
    assumed immutable

    Parameters
    ----------
    characters : np.ndarray(shape=(Nw,), dtype=complex)
        The characters of the irreducible representation
    precision : float
        The precision for comparison
    """

    def __init__(self, *args, precision=1e-4, **kwargs):
        self.characters = np.array(*args, **kwargs)
        self.precision = precision

    def __eq__(self, other):
        assert isinstance(other, Irrep), "Can only compare with another Irrep"
        return np.allclose(self.characters, other.characters, atol=self.precision)

    def is_real(self):
        return np.allclose(self.characters.imag, 0, atol=self.precision)

    def is_conjugate_of(self, other):
        assert isinstance(other, Irrep), "Can only compare with another Irrep"
        return np.allclose(self.characters, other.characters.conj(), atol=self.precision)

    def conj(self):
        return Irrep(self.characters.conj(), precision=self.precision)

    def __add__(self, other):
        assert isinstance(other, Irrep), "Can only add with another Irrep"
        return Irrep(self.characters + other.characters, precision=max(self.precision, other.precision))

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return str(np.round(self.characters, 6))

    def __repr__(self):
        return str(self)


def find_combinations_max(vectors, vector_max, lfixed, max_num_wann=None, num_wann_per_projection=None, rec=False):
    """
    Find all combinations of integer coefficients that satisfy the constraints
    sum( c*v for c,v in zip (coeffeicients,vectors) <= vector_max
    for all components of the vectors

    Parameters
    ----------
    vector_max : np.ndarray(shape=(N,), dtype=int)
        The maximum vector
    vectors : np.ndarray(shape=(M,N), dtype=int)
        The vectors
    max_num_wann : int
        The maximum number of wannier functions

    Returns
    -------
    combinations : np.ndarray(shape=(K,M), dtype=int)
        The K combinations of coefficients that denote multip[licity of each irrep
    """
    if max_num_wann is None:
        max_num_wann = np.inf
        num_wann_per_projection = np.ones(vectors.shape[0], dtype=int)
        print("unlimited number of wannier functions")
    else:
        assert num_wann_per_projection is not None, "The number of wannier functions per projection must be specified if max_num_wann is specified"
        assert len(num_wann_per_projection) == vectors.shape[0], "The number of wannier functions per projection must be specified for each projection"
    if vectors.shape[0] == 0:
        return [[]]
    else:
        i = 0 if not lfixed[0] else 1
        combinations = []
        while np.all(i * vectors[0] <= vector_max) and i * num_wann_per_projection[0] <= max_num_wann:
            for comb in find_combinations_max(vector_max=vector_max - i * vectors[0],
                                              vectors=np.copy(vectors[1:, :]),
                                              lfixed=lfixed[1:],
                                              max_num_wann=max_num_wann - i * num_wann_per_projection[0],
                                              num_wann_per_projection=num_wann_per_projection[1:],
                                              rec=True):
                combinations.append([i] + comb)
            if lfixed[0]:
                break
            i += 1
    if not rec:
        combinations = np.array(combinations)
        num_wann_per_comb = np.dot(combinations, num_wann_per_projection)
        srt = np.argsort(num_wann_per_comb)
        combinations = combinations[srt]
    return combinations


def check_combinations_min_max(combinations, vectors, vector_min, vector_max):
    """	
    Find all combinations of integer coefficients that satisfy the constraints
    vector_min <= sum( c*v for c,v in zip (coeffeicients,vectors) <= vector_max

    Parameters
    ----------
    combinations : np.ndarray(shape=(K,M), dtype=int)
        The K trial combinations
    vector_min : np.ndarray(shape=(N,), dtype=int)
        The minimum vector
    vector_max : np.ndarray(shape=(N,), dtype=int)
        The maximum vector
    vectors : np.ndarray(shape=(M,N), dtype=int)
        The vectors

    Returns
    -------
    combinations : np.ndarray(shape=(L,M), dtype=int)
        The L valid combinations (L <= K)
    """
    combinations_new = []
    for comb in combinations:
        dot = np.dot(comb, vectors)
        if (np.all(vector_min <= dot) and np.all(dot <= vector_max)):
            combinations_new.append(comb)
    return combinations_new
