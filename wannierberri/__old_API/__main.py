#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
#------------------------------------------------------------
#  This is the main file of the module

import functools
from .__evaluate import evaluate_K
from wannierberri.smoother import get_smoother
from wannierberri.result import __tabresult
from . import __integrate
from . import __tabulate
from wannierberri.__path import Path
import numpy as np
from collections.abc import Iterable

integrate_options = __integrate.calculators.keys()
tabulate_options = __tabulate.calculators.keys()
from wannierberri.utils.mmn2uHu import hlp as hlp_mmn
from wannierberri.utils.vaspspn import hlp as hlp_spn
from time import time
from wannierberri.parallel import Serial
import sys

from colorama import init
from termcolor import cprint
from pyfiglet import figlet_format


def figlet(text, font='cosmike', col='red'):
    init(strip=not sys.stdout.isatty())  # strip colors if stdout is redirected
    letters = [figlet_format(X, font=font).rstrip("\n").split("\n") for X in text]
    logo = []
    for i in range(len(letters[0])):
        logo.append("".join(L[i] for L in letters))
    cprint("\n".join(logo), col, attrs=['bold'])


def print_options():
    """
    Prints all options available for the moment.
    """

    def addparam(param, param_des):
        if len(param) > 0:
            return ". Additional parameters: \n" + "\n".join(
                (" " * 10 + "{0:10s} [ default =  {1}] ---  {2}".format(k, param[k], param_des[k]) for k in param))
        else:
            return ""

    for modname, mod in ("integrate", __integrate), ("tabulate", __tabulate):
        cprint("Options available to {}:".format(modname), 'green', attrs=['bold'])
        print(
            "\n".join(
                "{0:10s}  :  {1} {2}".format(
                    key, mod.descriptions[key],
                    addparam(mod.additional_parameters[key], mod.additional_parameters_description[key]))
                for key in mod.calculators.keys()) + "\n\n")
    hlp_mmn()
    hlp_spn()


#    cprint ("Options available to tabulate:",'green', attrs=['bold'])
#    print("\n".join("{0:10s}  :  {1} ".format(key,__tabulate.descriptions[key]) for key in tabulate_options)+"\n\n")



def check_option(quantities, avail, tp):
    for opt_full in quantities:
        opt = opt_full.split('^')[0]
        if opt not in avail:
            raise RuntimeError(
                "Quantity {} is not available for {}. Available options are : \n{}\n".format(opt, tp, avail))


# TODO: Unify the two methods, to do make possible doing
# integration and tabulating everything in one shot


def integrate(
        system,
        grid,
        Efermi=None,
        omega=None,
        Ef0=0,
        smearEf=10,
        smearW=None,
        quantities=[],
        user_quantities={},
        adpt_num_iter=0,
        adpt_fac=1,
        use_irred_kpt=True,
        symmetrize=True,
        fout_name="wberri",
        write_txt=True,
        write_bin=False,
        restart=False,
        suffix="",
        file_Klist="Klist",
        parallel=Serial(),
        print_Kpoints=True,
        parameters={},
        parameters_K={},
        specific_parameters={}):
    """
    Integrate

    Parameters
    ----------
    system : :class:`~wannierberri.system.System`
        System under investigation
    grid : :class:`~wannierberri.Grid`
        initial grid for integration
    Efermi : numpy.array
        The list of Fermi levels to be scanned (for Fermi-sea or Fermi-surface properties)
    omega : numpy.array
        The list of ferequencies levels to be scanned (for optical properties)
    smearEf : float
        smearing over Fermi levels (in Kelvin)
    quantities : list of str
        quantities to be integrated. See :ref:`sec-capabilities`
        One can append any label after a '^' symbol. In this case same quantity
        may be calculated several times in one run, but with different parameters (see specifuc_parameters)
    user_quantities : dict
        a dictionary `{name:function}`, where `name` is any string, and `function(data_K,Efermi)`
        takes two arguments
        `data_K` of  of type :class:`~wannierberri.data_K.Data_K`  and Efermi -  `np.array`
        and returns an object  :class:`~wannierberri.result.EnergyResult`
    adpt_num_iter : int
        number of recursive adaptive refinement iterations. See :ref:`sec-refine`
    adpt_fac : int
        number of K-points to be refined per quantity and criteria.
    parallel : :class:`~wannierberri.parallel.Parallel`
        object describing parallelization scheme
    use_irred_kpt : bool
        evaluate only symmetry-irreducible K-points
    symmetrize : bool
        symmetrize the result (always `True` if `use_irred_kpt == True`)
    fout_name : str
        beginning of the output files for each quantity after each iteration
    write_txt : bool
        write results after each iteration as text files (".dat")
    write_bin : bool
        write results after each iteration as binary files (".npz").
        See :func:`~wannierberri.result.EnergyResult.save` for file format
    print_Kpoints : bool
        print the list of K points
    parameters : dict
        `{'name':value,...}` , Each quantity that
        recognizes a parameter with the given name will use it
    specific_parameters : dict
        `'quantity^label':dict`, where dict is analogous to  `parameters`. This values will override
        for the instance of the quantity labeled by '^label'
    parameters_K : dict
        parameters to be passed to the :class:`~wannierberri.data_K.Data_K`,
        so they are common for the calculation.

    Returns
    --------
    dictionary of  :class:`~wannierberri.result.EnergyResult`

    Notes
    -----
    Results are also printed to ASCII files

    """
    #    smearW : float
    #        smearing over frequencies (in Kelvin)
    #    Ef0 : float
    #        a single  Fermi level for optical properties

    if use_irred_kpt:
        symmetrize = True
    cprint(
        "\nIntegrating the following  standard      quantities: " + ", ".join(quantities) + "\n",
        'green',
        attrs=['bold'])
    cprint(
        "\nIntegrating the following  user-defined  quantities: " + ", ".join(user_quantities.keys()) + "\n",
        'green',
        attrs=['bold'])
    check_option(quantities, integrate_options, "integrate")

    def to_array(energy):
        if energy is not None:
            if not isinstance(energy, Iterable):
                energy = [energy]
            return np.array(energy)
        else:
            return None

    omega = to_array(omega)
    Efermi = to_array(Efermi)
    # TODO : either remove smearW from here, or remove any smearing from inside kubo. This will not allow adaptive smearing though
    if smearW is not None:
        print(
            "WARNING : smearW parameter is neglected, smearing of frequency is currently done inside the kubo routine.\n"
            "          To specify smearing, pass smearing parameters to the variable 'parameters' as a dict.\n"
            "          See parameters_optical for details.")
        smearW = None
    smoothEf = get_smoother(Efermi, smearEf, "Fermi-Dirac")  # smoother for functions of Fermi energy
    smoothW = get_smoother(omega, smearW)  # smoother for functions of frequency
    # smoothW  = get_smoother(omega,  smearW,  "Gaussian") # smoother for functions of frequency

    eval_func = functools.partial(
        __integrate.intProperty,
        Efermi=Efermi,
        omega=omega,
        smootherEf=smoothEf,
        smootherOmega=smoothW,
        quantities=quantities,
        user_quantities=user_quantities,
        parameters=parameters,
        specific_parameters=specific_parameters)
    res = evaluate_K(
        eval_func,
        system,
        grid,
        adpt_num_iter=adpt_num_iter,
        adpt_nk=adpt_fac,
        use_irred_kpt=use_irred_kpt,
        symmetrize=symmetrize,
        fout_name=fout_name,
        suffix=suffix,
        write_txt=write_txt,
        write_bin=write_bin,
        restart=restart,
        file_Klist=file_Klist,
        parallel=parallel,
        parameters_K=parameters_K,
        print_Kpoints=print_Kpoints,
    )
    cprint("Integrating finished successfully", 'green', attrs=['bold'])
    return res


def tabulate(
    system,
    grid,
    quantities=[],
    user_quantities={},
    frmsf_name=None,
    ibands=None,
    suffix="",
    Ef0=0.,
    use_irred_kpt=True,
    symmetrize=True,
    parameters={},
    parameters_K={},
    specific_parameters={},
    degen_thresh=1e-4,
    degen_Kramers=False,
    parallel=Serial(),
    print_Kpoints=True,
):
    """
    Tabulate quantities to be plotted

    Parameters
    ----------
    system : :class:`~wannierberri.system.System`
        System under investigation
    grid : :class:`~wannierberri.Grid`
        initial grid for integration
    Ef0 : float
        a single  Fermi level. all energies are given with respect to Ef0
    quantities : list of str
        quantities to be integrated. See :ref:`sec-capabilities`
    user_quantities : dict
        a dictionary `{name:formula}`, where `name` is any string, and `formula`
        is a name of a child class of  :class:`~wannierberri.formula.Formula_ln`
        which should have defined attributes `nn` , `TRodd`, `Iodd`
    use_irred_kpt : bool
        evaluate only symmetry-irreducible K-points
    symmetrize : bool
        symmetrize the result
    frmsf_name :  str
        if not None, the results are also printed to text files, ready to plot by for `FermiSurfer <https://fermisurfer.osdn.jp/>`_
    parallel : :class:`~wannierberri.parallel.Parallel`
        object describing parallelization scheme
    print_Kpoints : bool
        print the list of K points
    parameters : dict
        `{'name':value,...}` , Each quantity that
        recognizes a parameter with the given name will use it
    specific_parameters : dict
        `'quantity^label':dict`, where dict is analogous to  `parameters`. This values will override
        for the instance of the quantity labeled by '^label'
    parameters_K : dict
        parameters to be passed to the :class:`~wannierberri.data_K.Data_K`,
        so they are common for the calculation.

    Returns
    --------
    :class:`~wannierberri.result.TABresult`


    """

    mode = '3D'
    if isinstance(grid, Path):
        mode = 'path'
        use_irred_kpt = False
        symmetrize = False
    cprint(
        "\nTabulating the following standard     quantities: " + ", ".join(quantities) + "\n", 'green', attrs=['bold'])
    cprint(
        "\nTabulating the following user-defined quantities: " + ", ".join(user_quantities.keys()) + "\n",
        'green',
        attrs=['bold'])
    check_option(quantities, tabulate_options, "tabulate")
    eval_func = functools.partial(
        __tabulate.tabXnk,
        ibands=ibands,
        quantities=quantities,
        user_quantities=user_quantities,
        parameters=parameters,
        specific_parameters=specific_parameters,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)
    t0 = time()
    res = evaluate_K(
        eval_func,
        system,
        grid,
        adpt_num_iter=0,
        restart=False,
        suffix=suffix,
        file_Klist=None,
        use_irred_kpt=use_irred_kpt,
        symmetrize=symmetrize,
        parallel=parallel,
        parameters_K=parameters_K,
        print_Kpoints=print_Kpoints)

    t1 = time()
    if mode == '3D':
        res = res.to_grid(grid.dense)
        t2 = time()
        ttxt, twrite = __tabresult.write_frmsf(
            frmsf_name,
            Ef0,
            parallel.num_cpus if parallel is not None else 1,
            quantities + list(user_quantities.keys()),
            res,
            suffix=suffix)

    t4 = time()

    cprint("Tabulating finished successfully", 'green', attrs=['bold'])
    print(("Time     : Total : {} s\n" + "        evaluate : {} s\n").format(t4 - t0, t1 - t0))

    if mode == '3D':
        print(
            ("         to_grid : {} s\n" + "         txt     : {} s\n" + "         write   : {} s\n").format(
                t2 - t1, ttxt, twrite))
    return res
