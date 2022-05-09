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