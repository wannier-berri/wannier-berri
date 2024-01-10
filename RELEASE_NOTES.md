Version 0.15.0
++++++++++++++++++
## What's Changed
* remove wannier90io from dependencies by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/265
* added python-3.11 by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/266
* changed all zeros_like to np.zeros by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/267
* source code is now uploaded to pypi by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/268
* removed old_API and corresponding tests and documentation by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/269
* compatibility with spglib>=2 by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/270
* more tests of sym_wann by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/271
* Fixing pyfftw by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/272
* Easy access to k-points by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/185
* replaced data for test Fe_sym, and reference files by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/278
* Fix spin factor by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/280
* Allow to manually set spin pairs by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/279
* refactor utils to be included in codecov by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/281
* testing data in systems by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/285
* fixed bug in the Velocity formula by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/287
* Optimization of sym_wann -NEW! by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/282
* more specific tests by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/294
* more tests by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/295
* implemented k-resolved static calculators by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/276
* Disentanglement from wannier90 inputs by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/288
* fixed most of pep8 issues.  by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/297


**Full Changelog**: https://github.com/wannier-berri/wannier-berri/compare/v...v0.15.0


Version 0.14.0
++++++++++++++++++
## What's Changed
* refactoring ??_R  by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/224
* Fix an edge case in DynamicCalculator by @jhryoo in https://github.com/wannier-berri/wannier-berri/pull/216
* modified ci.yml to avoid double builds by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/228
* also avoid double checks for lint and codecov by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/229
* Boosting speed in run.py by @jhryoo in https://github.com/wannier-berri/wannier-berri/pull/217
* allowed avoid writing frmsf files by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/226
* utils.postw90 by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/188
* fixed the tetrahedron method to be more accurate with der=0 by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/231
* allowed evaluation of multiple sets of Fermi levels in one calculatio… by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/234
* fixed a bug for tetrahedron method by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/232
* SystemSparse by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/240
* Tetrahedron grids by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/233
* fixed rename of w90io by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/246
* rearrangements in sym_wann by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/245
* created docs index by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/259
* adding documentation to the main repository by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/260
* Kdotp by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/243
* documentation for kdotp by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/261
* added test for dynamical caclculators with symmetrising results by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/262
* changes TRodd, Iodd, TRtrans to more general transformations by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/263
* Implement shift current and injection current via Calculator by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/197


**Full Changelog**: https://github.com/wannier-berri/wannier-berri/compare/v0.13.5...v0.14.0


Version 0.13.5
++++++++++++++++++
## What's Changed
* Solved index issue in the FPLO Hamiltonian reading function. by @philipp-eck in https://github.com/wannier-berri/wannier-berri/pull/203
* Allow running tests without gpaw and ase by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/195
* Fix use of factor in static calculators by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/202
* Fix ray cluster script for pbs, add num-cpus-per-node argument by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/205
* added test python 3.10, removed 3.7 by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/212
* symmetrization of magnetic system works for multi_atom unit cells. by @Liu-Xiaoxiong in https://github.com/wannier-berri/wannier-berri/pull/179
* fixed spglib<2 by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/213
* Bump protobuf from 3.20.1 to 3.20.2 by @dependabot in https://github.com/wannier-berri/wannier-berri/pull/211
* Phonons by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/191
* Change the attribute KpointBZ._max into a lazy property by @jhryoo in https://github.com/wannier-berri/wannier-berri/pull/218

## New Contributors
* @philipp-eck made their first contribution in https://github.com/wannier-berri/wannier-berri/pull/203
* @dependabot made their first contribution in https://github.com/wannier-berri/wannier-berri/pull/211
* @jhryoo made their first contribution in https://github.com/wannier-berri/wannier-berri/pull/218

**Full Changelog**: https://github.com/wannier-berri/wannier-berri/compare/v0.13.4...v0.13.5

Version 0.13.0
++++++++++++++++++

## What's Changed
* typo in cumdos by @Liu-Xiaoxiong in https://github.com/wannier-berri/wannier-berri/pull/166
* smoothwers in calculators by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/161
* allowed tabulation along a path using by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/164
* separated the old API to a submodule by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/163
* Fermiocean to calculator with doc by @Liu-Xiaoxiong in https://github.com/wannier-berri/wannier-berri/pull/165
* Change unit to SI by @Liu-Xiaoxiong in https://github.com/wannier-berri/wannier-berri/pull/168
* All Class in Static are children of `StaticCalculator` by @Liu-Xiaoxiong in https://github.com/wannier-berri/wannier-berri/pull/169
* Cleanup by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/162
* Re-arranged files into submodules `result`, `system` , `formula`
* rewrote most functionality as `Calculators`
* EnergyResult writes comment into output file
* separate classes Serial() and Parallel()


**Full Changelog**: https://github.com/wannier-berri/wannier-berri/compare/v0.12.0...v0.13.0


Version 0.12.0
++++++++++++++++++


## What's Changed
* Add python3.9 to CI by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/124
* Saving data as binary, update tests by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/122
* removed obsolete op,ed,kpart from the code by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/126
* Fix CI problem related to codecov and multiprocessing by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/132
* Split testing and coverage calculation in CI by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/133
* allowed calculation of symmetric and antisymmetric parts of optical c… by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/130
* Setup flake8 static analysis in CI, format some files, fix trivial bugs by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/134
* removed unused imports by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/136
* `run()`  instead of `integrate()` and `tabulate()` by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/127
* Tabulation of spin Berry curvature by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/138
* more tests with chirality and TR-braking by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/141
* Cleanup test by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/140
* Fix Path when input k_nodes is a list of numpy arrays, add test by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/143
* Fix symmetry Group constructor to raise for an infinite group by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/146
* Automatic symmetry detection using spglib by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/148
* Symmetrization of Wannier Hamiltonian and matrices by @Liu-Xiaoxiong in https://github.com/wannier-berri/wannier-berri/pull/89
* Fix wrong symmetry of opt_shiftcurrent by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/150
* Apply yapf to tests, fix all flake8 errors by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/151
* Apply yapf to wannierberri, fix all flake8 errors by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/154
* Importing WFs from Ase by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/155
* separated Systems and utils (mmn2uHu, vaspspn, tab_plot) to a submodules by @stepan-tsirkin in  https://github.com/wannier-berri/wannier-berri/pull/157
* Symmetrization of orbital magnetic moment. (BB_R and CC_R) by @Liu-Xiaoxiong in https://github.com/wannier-berri/wannier-berri/pull/156

Version 0.11.1
++++++++++++++++++


## What's Changed
* update for releasing v0.11.0 by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/103
* Optimize to_grid in tabulate by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/107
* Add guiding_centers argument to System_w90 by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/108
* Remove node duplication in Path by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/110
* Add option to disable printing of K points by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/111
* removed duplicate methods reverseR and conj_XX_R in System() by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/112
* Move install_script.sh from ci to github folder by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/119
* fixes for tabulation along path by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/104
* Fix ray cluster by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/113


**Full Changelog**: https://github.com/wannier-berri/wannier-berri/compare/v0.11.0...v0.11.1


Version 0.11.0
++++++++++++++++++

## What's Changed
* Static Spin Hall conductivity in FermiOcean by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/85
* fixed a bug with specific_parameters for user quantities by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/88
* fixes in models, tabulating morb and der_morb by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/86
* Automatically add tag when running uploadpypi.sh by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/87
* Models by @TomushT in https://github.com/wannier-berri/wannier-berri/pull/90
* interface to FPLO code by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/91
* use_ws=True with all other system() classes by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/92
* degen_thresh and degen_Kramers by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/94
* Fix bug in Path with k_nodes argument by @jaemolihm in https://github.com/wannier-berri/wannier-berri/pull/82
* Fixes for tabulate by @stepan-tsirkin in https://github.com/wannier-berri/wannier-berri/pull/100

## New Contributors
* @TomushT made their first contribution in https://github.com/wannier-berri/wannier-berri/pull/90

**Full Changelog**: https://github.com/wannier-berri/wannier-berri/compare/v0.10.0...asdf