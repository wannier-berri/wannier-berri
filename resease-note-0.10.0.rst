#This is a draft of the comment to a merge commit. 
#Please add here what you feel necessary

This PR implements a number of changes, which include 

  * transition to the new evaluation using FermiOcean and Formula_ln classes

  * Multi-node parallelization

  * Fermi-sea formulas for many quantities

  * External vs internal terms

  * Tetrahedron method

  * One quantity with differnet parameters in one run

  * user-defined quantities

  * two conventions for the Bloch sums

  * more quantities to tabulate

  * pre-defined TB models for illustration
  
  * deprecation of the parallelization using multiprocessing module (ray is now the only option)


New Reference data are added for the Chiral model


A number of Reference data changed:

  * Data for Morb were changed due to a bugfix:

    - berry_Fe_W90-Morb_iter-0000.dat

    - berry_Fe_W90-Morb-sym_iter-0000.dat

    - berry_Fe_W90-Morb-sym_iter-0001.dat

    - berry_Fe_W90-Morb-wcc_iter-0000.dat


  * reference data for all fermi-surface quantities were changed, because FermiOcean treats 
    fermi-level scan a bit diferently than the old nonabelian module:\

    - nonabelian was choosing bands in the interval [E-dE/2:E+dE/2]

    - fermiocean takes a numerical derivative of Fermi-sea result,
      which for fder=1 is equivalent to choosing bands in the interval [E-dE:E+dE]

    The changed data are:

    - berry_Fe_W90-conductivity_ohmic_fsurf_iter-0000.dat

    - berry_Fe_W90-conductivity_ohmic_fsurf-sym_iter-0000.dat

    - berry_Fe_W90-conductivity_ohmic_fsurf-sym_iter-0001.dat

    - berry_Fe_W90-conductivity_ohmic_iter-0000.dat

    - berry_Fe_W90-conductivity_ohmic-sym_iter-0000.dat

    - berry_Fe_W90-conductivity_ohmic-sym_iter-0001.dat

    - berry_Fe_W90-dos_iter-0000.dat

    - berry_Fe_W90-dos-sym_iter-0000.dat

    - berry_Fe_W90-dos-sym_iter-0001.dat

    - berry_GaAs_tb-berry_dipole_fsurf_iter-0000.dat

    - berry_GaAs_W90-berry_dipole_fsurf_iter-0000.dat

    - berry_Haldane_tbmodels-conductivity_ohmic_iter-0000.dat

    - berry_Haldane_tbmodels-conductivity_ohmic-sym_iter-0000.dat

    - berry_Haldane_tbmodels-conductivity_ohmic-sym_iter-0001.dat

    - berry_Haldane_tbmodels-dos_iter-0000.dat

    - berry_Haldane_tbmodels-dos-sym_iter-0000.dat

    - berry_Haldane_tbmodels-dos-sym_iter-0001.dat


  * In addition, all data from iter-0001 needed changes, because choice of refinement points is determined
    by all results on iter_0000, which is affected by the changes listed above

    - berry_Fe_W90-ahc-sym_iter-0001.dat

    - berry_Fe_W90-cumdos-sym_iter-0001.dat
