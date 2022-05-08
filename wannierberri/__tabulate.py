#                                                            #l
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

import numpy as np
from collections.abc import Iterable
from collections import defaultdict
from time import time
import multiprocessing
import functools
from . import __result as result
from . import covariant_formulak as frml
from . import covariant_formulak_basic as frml_basic

#If one whants to add  new quantities to tabulate, just modify the following dictionaries

#should be classes Formula_ln
# TODO : add factors to the calculation
calculators = {
    'spin': frml.Spin,
    'V': frml.Velocity,
    'berry': frml.Omega,
    'Der_berry': frml.DerOmega,
    'morb': frml.morb,
    'Der_morb': frml_basic.Der_morb,
    'spin_berry': frml.SpinOmega,
}

additional_parameters = defaultdict(lambda: defaultdict(lambda: None))
additional_parameters_description = defaultdict(lambda: defaultdict(lambda: "no description"))

descriptions = defaultdict(lambda: "no description")
descriptions['berry'] = "Berry curvature (Ang^{2})"
descriptions['Der_berry'] = "1st deravetive of Berry curvature (Ang^{3})"
descriptions['V'] = "velocity (eV*Ang)"
descriptions['spin'] = "Spin"
descriptions['morb'] = "orbital moment of Bloch states <nabla_k u_n| X(H-E_n) | nabla_k u_n> (eV*Ang**2)"
descriptions[
    'Der_morb'] = "1st derivative orbital moment of Bloch states <nabla_k u_n| X(H-E_n) | nabla_k u_n> (eV*Ang**2)"
descriptions['spin_berry'] = "Spin Berry curvature (hbar * Ang^{2})"

parameters_ocean = {
    'external_terms': (True, "evaluate external terms"),
    'internal_terms': (True, "evaluate internal terms"),
}

for key, val in parameters_ocean.items():
    for calc in ['berry', 'Der_berry', 'morb', 'Der_morb']:
        additional_parameters[calc][key] = val[0]
        additional_parameters_description[calc][key] = val[1]

additional_parameters['spin_berry']['spin_current_type'] = 'ryoo'
additional_parameters_description['spin_berry']['spin_current_type'] = 'type of the spin current'


def tabXnk(
        data_K,
        quantities=[],
        user_quantities={},
        degen_thresh=-1,
        degen_Kramers=False,
        ibands=None,
        parameters={},
        specific_parameters={}):

    if ibands is None:
        ibands = np.arange(data_K.nbands)
    else:
        ibands = np.array(ibands)

    tabulator = Tabulator(data_K, ibands, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)

    results = {'Energy': result.KBandResult(data_K.E_K[:, ibands], TRodd=False, Iodd=False)}
    for qfull in quantities:
        q = qfull.split('^')[0]
        __parameters = {}
        for param in additional_parameters[q]:
            if param in parameters:
                __parameters[param] = parameters[param]
            else:
                __parameters[param] = additional_parameters[q][param]
        results[qfull] = tabulator(calculators[q](data_K, **__parameters))

    for q, formula in user_quantities.items():
        if q in specific_parameters:
            __parameters = specific_parameters[q]
        else:
            __parameters = {}
        results[q] = tabulator(formula(data_K, **__parameters))

    return TABresult(kpoints=data_K.kpoints_all, recip_lattice=data_K.system.recip_lattice, results=results)


class Tabulator():

    def __init__(self, data_K, ibands, degen_thresh=1e-4, degen_Kramers=False):

        self.nk = data_K.nk
        self.NB = data_K.num_wann
        self.ibands = ibands

        band_groups = data_K.get_bands_in_range_groups(
            -np.Inf, np.Inf, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers, sea=False)
        # bands_groups  is a digtionary (ib1,ib2):E
        # now select only the needed groups
        self.band_groups = [
            [n for n in groups.keys() if np.any((ibands >= n[0]) * (ibands < n[1]))] for groups in band_groups
        ]  # select only the needed groups
        self.group = [[] for ik in range(self.nk)]
        for ik in range(self.nk):
            for ib in self.ibands:
                for n in self.band_groups[ik]:
                    if ib < n[1] and ib >= n[0]:
                        self.group[ik].append(n)
                        break

    def __call__(self, formula):
        """formula  - TraceFormula to evaluate"""
        rslt = np.zeros((self.nk, len(self.ibands)) + (3, ) * formula.ndim)
        for ik in range(self.nk):
            values = {}
            for n in self.band_groups[ik]:
                inn = np.arange(n[0], n[1])
                out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], self.NB)))
                values[n] = formula.trace(ik, inn, out) / (n[1] - n[0])
            for ib, b in enumerate(self.ibands):
                rslt[ik, ib] = values[self.group[ik][ib]]

        return result.KBandResult(rslt, TRodd=formula.TRodd, Iodd=formula.Iodd)


class TABresult(result.Result):

    def __init__(self, kpoints, recip_lattice, results={}):
        self.nband = results['Energy'].nband
        self.grid = None
        self.gridorder = None
        self.recip_lattice = recip_lattice
        self.kpoints = np.array(kpoints, dtype=float) % 1

        self.results = results
        for r in results:
            assert len(kpoints) == results[r].nk
            assert self.nband == results[r].nband

    @property
    def Enk(self):
        return self.results['Energy']

    def __mul__(self, other):
        #K-point factors do not play arole in tabulating quantities
        return self

    def __truediv__(self, other):
        #K-point factors do not play arole in tabulating quantities
        return self

    def __add__(self, other):
        if other == 0:
            return self
        if self.nband != other.nband:
            raise RuntimeError(
                "Adding results with different number of bands {} and {} - not allowed".format(self.nband, other.nband))
        results = {r: self.results[r] + other.results[r] for r in self.results if r in other.results}
        return TABresult(np.vstack((self.kpoints, other.kpoints)), recip_lattice=self.recip_lattice, results=results)

    def save(self, name):
        return  # do nothing so far

    def savetxt(self, name):
        return  # do nothing so far

    def savedata(self, name, prefix, suffix, i_iter):
        if i_iter > 0:
            pass  # so far do nothing on iterations, chang in future
        else:
            self.self_to_grid()
            write_frmsf(
                prefix + "-" + name, Ef0=0., numproc=None, quantities=self.results.keys(), res=self,
                suffix=suffix)  # so far let it be the only mode, implement other modes in future
            # TODO : remove this messy call to external routine, which calls back an internal one

    @property
    def find_grid(self):
        """ Find the full grid."""
        #TODO: make it cleaner, to work with iterations
        grid = np.zeros(3, dtype=int)
        kp = np.array(self.kpoints)
        for i in range(3):
            k = np.sort(kp[:, i])
            dk = np.max(k[1:] - k[:-1])
            grid[i] = int(np.round(1. / dk))
        return grid

    def transform(self, sym):
        results = {r: self.results[r].transform(sym) for r in self.results}
        kpoints = [sym.transform_reduced_vector(k, self.recip_lattice) for k in self.kpoints]
        return TABresult(kpoints=kpoints, recip_lattice=self.recip_lattice, results=results)

    def to_grid(self, grid, order='C'):
        print("setting the grid")
        grid1 = [np.linspace(0., 1., g, False) for g in grid]
        print("setting new kpoints")
        k_new = np.array(np.meshgrid(grid1[0], grid1[1], grid1[2], indexing='ij')).reshape((3, -1), order=order).T
        print("finding equivalent kpoints")
        # check if each k point is on the regular grid
        kpoints_int = np.rint(self.kpoints * grid[None, :]).astype(int)
        on_grid = np.all(abs(kpoints_int / grid[None, :] - self.kpoints) < 1e-5, axis=1)

        # compute the index of each k point on the grid
        kpoints_int = kpoints_int % grid[None, :]
        ind_grid = kpoints_int[:, 2] + grid[2] * (kpoints_int[:, 1] + grid[1] * kpoints_int[:, 0])

        # construct the map from the grid indices to the k-point indices
        k_map = [[] for i in range(np.prod(grid))]
        for ik in range(len(self.kpoints)):
            if on_grid[ik]:
                k_map[ind_grid[ik]].append(ik)
            else:
                print(f"WARNING: k-point {ik}={self.kpoints[ik]} is not on the grid, skipping.")
        t0 = time()
        print("collecting")
        results = {r: self.results[r].to_grid(k_map) for r in self.results}
        t1 = time()
        print("collecting: to_grid  : {}".format(t1 - t0))
        res = TABresult(k_new, recip_lattice=self.recip_lattice, results=results)
        t2 = time()
        print("collecting: TABresult  : {}".format(t2 - t1))
        res.grid = np.copy(grid)
        res.gridorder = order
        t3 = time()
        print("collecting - OK : {} ({})".format(t3 - t0, t3 - t2))
        return res

    def self_to_grid(self):
        res = self.to_grid(self.find_grid, order='C')
        self.__dict__.update(res.__dict__)  # another dirty trick, TODO : clean it

    def __get_data_grid(self, quantity, iband, component=None, efermi=None):
        if quantity == 'Energy':
            return self.Enk.data[:, iband].reshape(self.grid)
        elif component is None:
            return self.results[quantity].data[:, iband].reshape(tuple(self.grid) + (3, ) * self.results[quantity].rank)
        else:
            return self.results[quantity].get_component(component)[:, iband].reshape(self.grid)

    def __get_data_path(self, quantity, iband, component=None, efermi=None):
        if quantity == 'Energy':
            return self.Enk.data[:, iband]
        elif component is None:
            return self.results[quantity].data[:, iband]
        else:
            return self.results[quantity].get_component(component)[:, iband]

    def get_data(self, quantity, iband, component=None, efermi=None):
        if self.grid is None:
            return self.__get_data_path(quantity, iband, component=component, efermi=efermi)
        else:
            return self.__get_data_grid(quantity, iband, component=component, efermi=efermi)

    def fermiSurfer(self, quantity=None, component=None, efermi=0, npar=0, iband=None, frmsf_name=None):
        if iband is None:
            iband = np.arange(self.nband)
        elif isinstance(iband, int):
            iband = [iband]
        if not (quantity is None):
            Xnk = self.results[quantity].get_component(component)
        if self.grid is None:
            raise RuntimeError("the data should be on a grid before generating FermiSurfer files. use to_grid() method")
        if self.gridorder != 'C':
            raise RuntimeError("the data should be on a 'C'-ordered grid for generating FermiSurfer files")
        FSfile = ""
        FSfile += (" {0}  {1}  {2} \n".format(self.grid[0], self.grid[1], self.grid[2]))
        FSfile += ("1 \n")  # so far only this option of Fermisurfer is implemented
        FSfile += ("{} \n".format(len(iband)))
        FSfile += ("".join(["  ".join("{:14.8f}".format(x) for x in v) + "\n" for v in self.recip_lattice]))

        FSfile += _savetxt(a=self.Enk.data[:, iband].flatten(order='F') - efermi, npar=npar)

        if quantity is None:
            return FSfile

        if quantity not in self.results:
            raise RuntimeError("requested quantity '{}' was not calculated".format(quantity))
            return FSfile
        FSfile += _savetxt(a=Xnk[:, iband].flatten(order='F'), npar=npar)
        if frmsf_name is not None:
            if not (frmsf_name.endswith(".frmsf")):
                frmsf_name += ".frmsf"
            open(frmsf_name, "w").write(FSfile)
        return FSfile

    def plot_path_fat(
            self,
            path,
            quantity=None,
            component=None,
            save_file=None,
            Eshift=0,
            Emin=None,
            Emax=None,
            iband=None,
            mode="fatband",
            fatfactor=20,
            kwargs_line={},
            label=None,
            fatmax=None,
            cut_k=True,
            linecolor = 'black',
            close_fig=True,
            show_fig=True
                    ):
        """
        a routine to plot a result along the path
        The circle size (size of quantity) changes linearly below 2 and logarithmically above 2.
        """

        if fatmax is None : fatmax = fatfactor*10

        import matplotlib.pyplot as plt
        if iband is None:
            iband = np.arange(self.nband)
        elif isinstance(iband, int):
            iband = np.array([iband])
        elif isinstance(iband, Iterable):
            iband = np.array(iband)
            if iband.dtype != int:
                raise ValueError("iband should be integer")
        else:
            raise ValueError("iband should be either an integer, or array of intergers, or None")

        kline = path.getKline()
        E = self.get_data(quantity='Energy', iband=iband) - Eshift
#        print("shape of Energy", E.shape)

        plt.ylabel(r"$E$, eV")
        if Emin is None:
            Emin = E.min() - 0.5
        if Emax is None:
            Emax = E.max() + 0.5

        klineall = []
        for ib in range(len(iband)):
            e = E[:, ib]
            selE = (e <= Emax) * (e >= Emin)
            e[~selE] = None
            if np.any(selE):
                klineselE = kline[selE]
                klineall.append(klineselE)
                _line, = plt.plot(kline, e, c = linecolor,**kwargs_line)
        if label is not None:
            _line.set_label(label)
            plt.legend()
        if cut_k:
            klineall = [k for kl in klineall for k in kl]
            kmin = min(klineall)
            kmax = max(klineall)
        else:
            kmin = kline.min()
            kmax = kline.max()

        if quantity is not None:
            data = self.get_data(quantity=quantity, iband=iband, component=component)
#            print("shape of data", data.shape)
            if mode == "fatband":
                for ib in range(len(iband)):
                    e = E[:, ib]
                    selE = (e <= Emax) * (e >= Emin)
                    klineselE = kline[selE]
                    y = data[selE][:, ib]
                    select = abs(y) > 2
                    y[select] = np.log2(abs(y[select]))*np.sign(y[select])
                    y[~select] *= 0.5
                    e1=e[selE]
                    for col,sel in [("red",(y>0)),("blue",(y<0))]:
                        sz = abs(y[sel])*fatfactor
                        sz[sz>fatmax] = fatmax
                        plt.scatter(klineselE[sel],e1[sel],s=sz,color=col)
            else :
                raise ValueError("So far only fatband mode is implemented")

        x_ticks_labels = []
        x_ticks_positions = []
        for k, v in path.labels.items():
            x_ticks_labels.append(v)
            x_ticks_positions.append(kline[k])
            plt.axvline(x=kline[k])
        plt.xticks(x_ticks_positions, x_ticks_labels)
        plt.ylim([Emin, Emax])
        plt.xlim([kmin, kmax])

        fig = plt.gcf()

        if save_file is not None:
            fig.savefig(save_file)

        if show_fig:
            plt.show()

        if close_fig:
            plt.close(fig)
        else:
            return fig

    def max(self):
        return -1  # tabulating does not contribute to adaptive refinement


def _savetxt(limits=None, a=None, fmt=".8f", npar=0):
    assert a.ndim == 1, "only 1D arrays are supported. found shape{}".format(a.shape)
    if npar is None:
        npar = multiprocessing.cpu_count()
    if npar <= 0:
        if limits is None:
            limits = (0, a.shape[0])
        fmtstr = "{0:" + fmt + "}\n"
        return "".join(fmtstr.format(x) for x in a[limits[0]:limits[1]])
    else:
        if limits is not None:
            raise ValueError("limits shpould not be used in parallel mode")
        nppproc = a.shape[0] // npar + (1 if a.shape[0] % npar > 0 else 0)
        print("using a pool of {} processes to write txt frmsf of {} points per process".format(npar, nppproc))
        asplit = [(i, i + nppproc) for i in range(0, a.shape[0], nppproc)]
        p = multiprocessing.Pool(npar)
        res = p.map(functools.partial(_savetxt, a=a, fmt=fmt, npar=0), asplit)
        p.close()
        p.join()
        return "".join(res)


def write_frmsf(frmsf_name, Ef0, numproc, quantities, res, suffix=""):
    if len(suffix) > 0:
        suffix = "-" + suffix
    if frmsf_name is not None:
        open(f"{frmsf_name}_E{suffix}.frmsf", "w").write(res.fermiSurfer(quantity=None, efermi=Ef0, npar=numproc))
        ttxt = 0
        twrite = 0
        for Q in quantities:
            for comp in res.results[Q].get_component_list():
                try:
                    t31 = time()
                    txt = res.fermiSurfer(quantity=Q, component=comp, efermi=Ef0, npar=numproc)
                    t32 = time()
                    open(f"{frmsf_name}_{Q}-{comp}{suffix}.frmsf", "w").write(txt)
                    t33 = time()
                    ttxt += t32 - t31
                    twrite += t33 - t32
                except result.NoComponentError:
                    pass
    else:
        ttxt = 0
        twrite = 0
    return ttxt, twrite
