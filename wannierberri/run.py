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
# ------------------------------------------------------------

import os
import numpy as np
from collections.abc import Iterable
from time import time
import pickle
import glob
from termcolor import cprint
import warnings
from .utility import remove_dir
from .data_K import get_data_k
from .grid import exclude_equiv_points, Path, Grid, GridTetra
from .parallel import get_ray_cpus_count
from .result import ResultDict


def print_progress(count, total, t0, tprev, progress_step_time):
    t = time() - t0
    if count == 0:
        t_remain = "unknown"
        t_est_tot = "unknown"
    else:
        t_rem_s = t / count * (total - count)
        t_remain = f"{t_rem_s:22.1f}"
        t_est_tot = f"{t_rem_s + t:22.1f}"
    if t - tprev > progress_step_time:
        print(f"{count:20d}{t:17.1f}{t_remain:>22s}{t_est_tot:>22s}", flush=True)
        tprev = t
    return tprev


def process(paralfunc,
            K_list,
            parallel,
            dump_results,
            remote_parameters,
            progress_step_time=5,
            progress_step_percent=1):
    if remote_parameters is None:
        remote_parameters = {}
    # print(f"pointgroup : {pointgroup}")
    t0 = time()
    t_print_prev = 0
    selK = [ik for ik, k in enumerate(K_list) if not k.has_result()]
    numK = len(selK)
    dK_list = [K_list[ik] for ik in selK]
    if len(dK_list) == 0:
        print("nothing to process now")
        return 0

    print(f"processing {len(dK_list)} K points :", end=" ")
    nproc_loc = get_ray_cpus_count()
    if nproc_loc == 1:
        print("in serial.")
    else:
        print(f"using  {nproc_loc} processes.")

    print("# K-points calculated  Wall time (sec)  Est. remaining (sec)   Est. total (sec)", flush=True)
    nstep_print = max(1, nproc_loc, int(round(numK * progress_step_percent / 100)))

    if not parallel:
        for count, Kp in enumerate(dK_list):
            res = paralfunc(Kp, **remote_parameters)
            Kp.set_result(res, dump=dump_results)
            if (count + 1) % nstep_print == 0:
                t_print_prev = print_progress(count=count + 1,
                                              total=numK,
                                              t0=t0,
                                              tprev=t_print_prev,
                                              progress_step_time=progress_step_time)
    else:
        if dump_results:
            raise NotImplementedError("dump_results=True is not implemented for parallel processing yet")
        import ray
        remotes = [paralfunc.remote(dK, **remote_parameters) for dK in dK_list]
        num_remotes = len(remotes)
        num_remotes_calculated = 0
        while True:
            remotes_calculated, _ = ray.wait(
                remotes, num_returns=min(num_remotes_calculated + nstep_print, num_remotes),
                timeout=60)  # even, if the required number of remotes had not finished,
            # the progress will be printed every minute
            num_remotes_calculated = len(remotes_calculated)
            if num_remotes_calculated >= num_remotes:
                break
            t_print_prev = print_progress(count=num_remotes_calculated,
                                          total=numK,
                                          t0=t0,
                                          tprev=t_print_prev,
                                          progress_step_time=progress_step_time)
        ray.get(remotes)

    t = time() - t0

    print(f"time for processing {numK:6d} K-points on {nproc_loc:3d} processes: ", end="")
    print(f"{t:10.4f} ; per K-point {t / numK:15.4f} ; proc-sec per K-point {t * nproc_loc / numK:15.4f}", flush=True)
    return len(dK_list)


def run(
        system,
        grid,
        calculators,
        adpt_num_iter=0,
        use_irred_kpt=True,
        symmetrize=True,
        fout_name="result",
        suffix="",
        parameters_K=None,
        file_Klist_path=None,
        restart=False,
        restart_iteration=-1,
        Klist_part=10,
        parallel=True,  # will fall into serial if ray is not installed/initialized
        print_Kpoints=False,
        adpt_mesh=2,
        adpt_fac=1,
        dump_results=False,
        print_progress_step_time=5,
        print_progress_step_percent=1,
):
    """
    The function to run a calculation. Substitutes the old (obsolete and removed) `integrate()` and `tabulate()`
    and allows to integrate and tabulate in one run.

    Parameters
    ----------
    system : :class:`~wannierberri.system.System`
        System under investigation
    grid : :class:`~wannierberri.Grid` or :class:`~wannierberri.Path`
        initial grid for integration. or path for tabulation
    calculators : dict
        a dictionary where keys aare any string identifiers, and the values are of :class:`~wannierberri.calculators.Calculator`
    adpt_num_iter : int
        number of recursive adaptive refinement iterations. See :ref:`sec-refine`
    adpt_mesh : int
        the size of the refinement grid (usuallay no need to change)
    adpt_fac : int
        number of K-points to be refined per quantity and criteria.
    parallel : bool
        wether to use parallelization with ray or not. if True - ray should be initialized before.
    use_irred_kpt : bool
        evaluate only symmetry-irreducible K-points
    symmetrize : bool
        symmetrize the result (always `True` if `use_irred_kpt == True`)
    fout_name : str
        beginning of the output files for each quantity after each iteration
    suffix : str
        extra marker inserted into output files to mark this particular calculation run
    print_Kpoints : bool
        print the list of K points
    file_Klist : str or None
        name of file where to store the Kpoint list of each iteration. May be needed to restart a calculation
        to get more iterations. If `None` -- the file is not written
    restart : bool
        if `True` : reads restart information from `file_Klist` and starts from there
    Klist_part : int
        write the file_Klist by portions. Increase for speed, decrease for memory saving
    parameters_K: dict
        parameters to be passed to :class:`~wannierberri.data_K.Data_K` class
    dump_results : bool
        if `True` : dumps the results of each K-point in separate files. This may be slower due 
        to read/write operations, but may save memory if the results are large (many K-points, many fermi levels/frequencies, multidimensinal tensors etc..)
    print_progress_step_time : float or int
        minimal intervals (in seconds) to print progress


    Returns
    --------
    dictionary of  :class:`~wannierberri.result.EnergyResult`

    Notes
    -----
    Results are also printed to ASCII files
    """
    assert isinstance(parallel, bool), "parallel should be True or False"
    if parallel:
        try:
            import ray
            if not ray.is_initialized():
                warnings.warn("ray package found, but ray is not initialized, running in serial mode")
                parallel = False
        except ImportError:
            warnings.warn("ray package not found, running in serial mode")
            parallel = False

    if file_Klist_path is None:
        file_Klist_path = "_tmp_wb"
    file_Klist = os.path.join(file_Klist_path, "K_list.pickle")


    def get_Kpoint_storage_path(ik):
        return os.path.join(file_Klist_path, f"_Kp-{ik}.pickle")

    def write_factors(factors, iter):
        with open(os.path.join(file_Klist_path, f"factors_iter-{iter:08d}.npy"), 'wb') as f:
            np.save(f, factors) 

    def read_factors(iter):
        if iter >= 0:
            with open(os.path.join(file_Klist_path, f"factors_iter-{iter:08d}.npy"), 'rb') as f:
                factors = np.load(f)
            return iter, factors
        else:
            files = glob.glob(os.path.join(file_Klist_path, "factors_iter-*.npy"))
            iter_indices = np.array([int(f.split("-")[-1].split(".")[0]) for f in files])
            iter_index = iter_indices[-1] + iter + 1
            if iter_index < 0:
                iter_index = 0
            else:
                if iter_index not in iter_indices:
                    Warning(f"requested iteration {iter} (index {iter_index}) is not found in factors files, will try to use the closest previous one")
                    iter_index = iter_indices[iter_indices <= iter_index][-1]
            return read_factors(iter_index)
            
    cprint("Starting run()", 'red', attrs=['bold'])
    if parameters_K is None:
        parameters_K = {}
    print_calculators(calculators)
    # along a path only tabulating is possible
    if isinstance(grid, Path):
        print("Calculation along a path - checking calculators for compatibility")
        for key, calc in calculators.items():
            print(key, calc)
            if not calc.allow_path:
                raise ValueError(
                    f"Calculation along a Path is running, but calculator `{key}` is not compatible with a Path")
        print("All calculators are compatible")
        if symmetrize:
            print("Symmetrization switched off for Path")
            symmetrize = False
    else:
        print("Calculation on  grid - checking calculators for compatibility")
        if use_irred_kpt:
            symmetrize = True
        for key, calc in calculators.items():
            print(key, calc)
            if not calc.allow_grid:
                raise ValueError(
                    f"Calculation on Grid is running, but calculator `{key}` is not compatible with a Grid")
        print("All calculators are compatible")

    if isinstance(grid, GridTetra):
        print("Grid is tetrahedral")
    else:
        print("Grid is regular")

    print(f"The set of k points is a {grid.str_short}")

    remote_parameters = {'_system': system, '_grid': grid, '_calculators': calculators, 'symmetrize': symmetrize}
    if parallel:
        remote_parameters = {k: ray.put(v) for k, v in remote_parameters.items()}

        @ray.remote
        def paralfunc(Kpoint, _system, _grid, _calculators, symmetrize):
            # import sys
            # print("Worker sys.path:", sys.path)
            # from wannierberri.system.rvectors import Rvectors
            with get_data_k(_system, Kpoint.Kp_fullBZ, grid=_grid, Kpoint=Kpoint, **parameters_K) as data:
                resultdic = {k: v(data) for k, v in _calculators.items()}
            result = ResultDict(resultdic)
            if symmetrize:
                result = _system.pointgroup.symmetrize(result)
            return result
    else:
        def paralfunc(Kpoint, _system, _grid, _calculators, symmetrize):
            with get_data_k(_system, Kpoint.Kp_fullBZ, grid=_grid, Kpoint=Kpoint, **parameters_K) as data:
                resultdic = {k: v(data) for k, v in _calculators.items()}
            result = ResultDict(resultdic)
            if symmetrize:
                result = _system.pointgroup.symmetrize(result)
            return result

    if restart:
        fr = open(file_Klist, "rb")
        K_list = []
        with open(file_Klist, "rb") as fr:
            while True:
                try:
                    K_list += pickle.load(fr)
                except EOFError:
                    print(f"Finished reading Klist from file {file_Klist}")
                    break
            print(f"{len(K_list)} K-points were read from {file_Klist}")
        nk_prev = len(K_list)
        start_iter, factors = read_factors(restart_iteration)
        factors = np.hstack([factors, np.zeros(len(K_list) - len(factors))])  # If we have more K-points than factors, add zeros for the new ones
        for ik, (Kp, fac) in enumerate(zip(K_list, factors)):
            Kp.set_factor(fac)
    else:
        K_list = grid.get_K_list(use_symmetry=use_irred_kpt)
        factors = np.array([Kp.factor for Kp in K_list])
        print("Done, sum of weights:{}".format(sum(Kp.factor for Kp in K_list)))
        start_iter = 0
        nk_prev = 0
        remove_dir(file_Klist_path)
        os.makedirs(file_Klist_path)
        write_factors(factors, iter=0)


        # remove_file(file_Klist_factor_changed)

    if adpt_num_iter < 0:
        adpt_num_iter = -adpt_num_iter * np.prod(grid.div) / np.prod(adpt_mesh) / adpt_fac / 3
    adpt_num_iter = int(round(adpt_num_iter))

    if (adpt_mesh is None) or np.max(adpt_mesh) <= 1:
        adpt_num_iter = 0
    else:
        if not isinstance(adpt_mesh, Iterable):
            adpt_mesh = [adpt_mesh] * 3
        adpt_mesh = np.array(adpt_mesh)

    do_write_Klist = True
    counter = 0
    result_all = None
    factors_old = None

    for i_iter in range(adpt_num_iter + 1):
        for ik in range(nk_prev, len(K_list)):
            K_list[ik].set_storage_path(get_Kpoint_storage_path(ik))
        i_iter_global = i_iter + start_iter
        print("\n" + "#" * 60)
        print(f"Iteration {i_iter_global} out of {adpt_num_iter + start_iter} ")
        if print_Kpoints:
            print("iteration {0} - {1} points. New points are:".format(i_iter + start_iter,
                                                                       len([K for K in K_list if K.result is None])))
            for i, K in enumerate(K_list):
                if not K.evaluated:
                    print(f" K-point {i} : {K} ")
        counter += process(
            paralfunc=paralfunc,
            K_list=K_list,
            parallel=parallel,
            dump_results=dump_results,
            progress_step_time=print_progress_step_time,
            remote_parameters=remote_parameters)

        nk = len(K_list)
        if do_write_Klist:
            # append new (refined) k-points only
            fw = open(file_Klist, "ab")
            for ink in range(nk_prev, nk, Klist_part):
                pickle.dump(K_list[ink:ink + Klist_part], fw)
            fw.close()

        time0 = time()

        if (result_all is None):
            result_all = sum(kp.get_result_factor() for kp in K_list)
        else:
            factors_old = factors
            factors = np.array([kp.factor for kp in K_list])
            factors_diff = np.copy(factors)
            factors_diff[:len(factors_old)] -= factors_old
            factors_diff_dict = {i: fac for i, fac in enumerate(factors_diff) if abs(fac) > 1.e-8}
            result_all += sum(K_list[i].get_result() * fac for i, fac in factors_diff_dict.items())
            write_factors(factors, iter=i_iter_global)

        time1 = time()
        print("time1 = ", time1 - time0)
        if not (restart and i_iter == 0):
            result_all.savedata(prefix=fout_name, suffix=suffix, i_iter=i_iter + start_iter)

        if i_iter >= adpt_num_iter:
            break

        # Now add some more points
        Kmax = np.array([K.max for K in K_list]).T
        select_points = set().union(*(np.argsort(Km)[-adpt_fac:] for Km in Kmax))

        time2 = time()
        print("time2 = ", time2 - time1)
        l1 = len(K_list)

        nk_prev = nk

        for iK in select_points:
            K_list += K_list[iK].divide(ndiv=adpt_mesh, periodic=system.periodic, use_symmetry=use_irred_kpt)

        if use_irred_kpt and isinstance(grid, Grid):
            exclude_equiv_points(K_list, new_points=len(K_list) - l1)

        print("sum of weights now :{}".format(sum(Kp.factor for Kp in K_list)))

    print(f"Totally processed {counter} K-points ")
    print("run() finished")

    return result_all


def print_calculators(calculators):
    cprint("Using the follwing calculators : \n" + "#" * 60 + "\n", "cyan", attrs=["bold"])
    for key, val in calculators.items():
        cprint(f" '{key}' ", "magenta", attrs=["bold"], end="")
        print(" : ", end="")
        cprint(f" {val} ", "yellow", attrs=["bold"], end="")
        print(f" : {val.comment}")
    cprint("#" * 60, "cyan", attrs=["bold"])
