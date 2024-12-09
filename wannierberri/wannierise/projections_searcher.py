import numpy as np
from termcolor import cprint
from ..symmetry.Dwann import Dwann
from .utility import char_to_vector, get_irreps, select_window_degen
from .projections import ORBITALS, ProjectionsSet


class EBRsearcher:
    """
    A class  to search for EBRs that cover the frozen window, but fit into the outer window

    Parameters
    ----------
    symmetrizer : `~wannierberri.symmetry.symmetrizer_sawf.SymmetrizerSAWF`
        The file of symmetry properties (only the band part is used, the wann part may be anything or empty)
    froz_min, froz_max : float
        The minimum and maximum energy of the frozen window
    outer_min, outer_max : float
        The minimum and maximum energy of the outer window
    degen_thresh : float
        The threshold to consider bands as degenerate
    trial_positions : list of np.ndarray(shape=(3,), dtype=float)
        The trial projections to be tested. Only one point of the orbit is needed to determine the Wyckoff position
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
    nsym_little : list of int (length=NKirr)
        The number of little group symmetries for each irreducible k-point

    Notes
    -----
    * So far it does not work with time-reversal symmetry, and was not tested with spinor wavefuncitons.
      It is recommended to search for projections on a scalar calculation (no spin). 
      The projections found this way are usually good for spinful calculations as well.
    """
    def __init__(self,
                 symmetrizer,
                 froz_min=np.inf, froz_max=-np.inf,
                 outer_min=-np.inf, outer_max=np.inf,
                 degen_thresh=1e-8,
                 trial_positions=None,
                 trial_orbitals=None,
                 trial_projections=None,
                 debug=False
                ):
        from spgrep.representation import get_character
        spacegroup = symmetrizer.spacegroup
        if spacegroup.spinor:
            raise NotImplementedError("EBRsearcher does not work with spinors")
        if trial_positions is None:
            trial_positions = []
        if trial_orbitals is None:
            trial_orbitals = []
        assert len(trial_positions) == len(trial_orbitals), "The number of trial positions and orbitals must be the same"
        self.eig = symmetrizer.eig_irr
        self.NKirr = symmetrizer.NKirr
        self.nsym_little = [len(l) for l in symmetrizer.isym_little]
        self.debug = debug

        # list of list of UniqueList of Irreps for each projection [iproj][ik]
        def highlight(line):
            if self.debug:
                cprint("#" * 80 + "\n#" + " " * 5 + f"{line}\n" + "#" * 80, color="yellow", attrs=['bold'])

        def debug_msg(msg=""):
            if self.debug:
                print("DEBUG: " + msg)

        highlight("Detrimine all possible irreps")
        all_possible_irreps_conj = []
        for ik in range(self.NKirr):
            kpoint = symmetrizer.kpoints_all[symmetrizer.kptirr[ik]]
            irreps, mapping = get_irreps(spacegroup=spacegroup, kpoint=kpoint)
            assert np.all(mapping == symmetrizer.isym_little[ik]), f"{mapping} != {symmetrizer.isym_little[ik]}"
            all_possible_irreps_conj.append(irreps.conj())



            

        # if projections are specified as a list of Projection objects (overriding trial_positions and trial_orbitals)
        if trial_projections is not None:
            if isinstance(trial_projections, ProjectionsSet):
                trial_projections = trial_projections.projections
            for projs in trial_projections:
                for proj in projs.split():
                    trial_positions.append(proj.positions)
                    trial_orbitals.append(proj.orbitals[0])
        self.num_trial_proj = len(trial_positions)



        highlight("detecting irreps genberated by each projections")

        self.num_wann_per_projection = []
        
        # find which charagters are produced by each projection
        characters_wann = [ [] for _ in range(self.NKirr)]
        for wyckoff, orbital in zip(trial_positions, trial_orbitals):
            debug_msg(f"detecting irreps by position {wyckoff} and orbital {orbital}")
            dwann = Dwann(spacegroup=spacegroup,
                          positions=wyckoff,
                          orbital=orbital,
                          ORBITALS=ORBITALS).get_on_points_all( kpoints=symmetrizer.kpoints_all,
                                                                ikptirr=symmetrizer.kptirr,
                                                            	ikptirr2kpt=symmetrizer.kptirr2kpt)
            self.num_wann_per_projection.append(dwann.shape[2])
            for ik in range(self.NKirr):
                characters_wann[ik].append(get_character(dwann[ik][symmetrizer.isym_little[ik]]))
        characters_wann = [ np.array(char) for char in characters_wann]


        # Now for each k-point characterize each projection as a vector
        # of the number of times each irrep (from all possible ones) appears in the projection
        # [ik][iproj]
        
        debug_msg(f"num_wann_per_projection={self.num_wann_per_projection}")

        highlight(" deternine the irreps in the DFT bands")
        self.irreps_frozen_vectors = []
        self.irreps_outer_vectors = []
        self.irreps_per_projection_vectors = []
        for ik in range(self.NKirr):
            frozen = select_window_degen(self.eig[ik], thresh=degen_thresh,
                                     win_min=froz_min, win_max=froz_max, return_indices=True)
            outer = select_window_degen(self.eig[ik], thresh=degen_thresh,
                                     win_min=outer_min, win_max=outer_max, return_indices=True)
            nfrozen = len(frozen)
            char_outer = np.array([symmetrizer.d_band_diagonal(ik, isym)[outer].sum() for isym in symmetrizer.isym_little[ik]])
            char_frozen = np.array([symmetrizer.d_band_diagonal(ik, isym)[frozen].sum() for isym in symmetrizer.isym_little[ik]])
            debug_msg(f"ik= {ik} contains {nfrozen} frozen states\n" +
                  f"the little group contains {len(symmetrizer.isym_little[ik])} symmetries: \n {symmetrizer.isym_little[ik]}\n" +
                  f"characters in outer window : {np.round(char_outer, 3)}\n" +
                  f"characters in frozen window: {np.round(char_frozen, 3)}")
            
            vector_frozen = char_to_vector(char_frozen, all_possible_irreps_conj[ik])
            vector_outer = char_to_vector(char_outer, all_possible_irreps_conj[ik])
            vector_wann =  char_to_vector(characters_wann[ik], all_possible_irreps_conj[ik])

            self.irreps_frozen_vectors.append(vector_frozen)
            self.irreps_outer_vectors.append(vector_outer)
            self.irreps_per_projection_vectors.append(vector_wann)
            debug_msg(f"Frozen states are represented by vector {vector_frozen}")
            debug_msg(f"Outer states are represented by vector {vector_outer}")
            debug_msg(f"Projections are represented by vectors \n {vector_wann}")

            nband_frozen_irreps = np.round(sum(v * irrep[0]
                                      for v, irrep in zip(vector_frozen, all_possible_irreps_conj[ik])), 3)
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
