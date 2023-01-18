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