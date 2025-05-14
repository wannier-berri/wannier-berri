import numpy as np
from time import time
import multiprocessing
import functools
from collections.abc import Iterable
import warnings
from .result import Result
from .kbandresult import get_component


class TABresult(Result):

    def __init__(self, kpoints, recip_lattice, results=None, mode="grid", save_mode="bin"):
        if results is None:
            results = {}
        self.nband = results['Energy'].nband
        self.mode = mode
        self.grid = None
        self.gridorder = None
        self.recip_lattice = recip_lattice
        self.kpoints = np.array(kpoints, dtype=float) % 1
        self.save_mode = save_mode
        self.results = results
        for k, res in results.items():
            assert len(kpoints) == res.nk
            if hasattr(res, "nband"):
                assert self.nband == res.nband

    @property
    def Enk(self):
        return self.results['Energy']

    def __mul__(self, other):
        # K-point factors do not play arole in tabulating quantities
        return self

    def __truediv__(self, other):
        # K-point factors do not play arole in tabulating quantities
        return self

    def __add__(self, other):
        if other == 0:
            return self
        assert self.mode == other.mode
        assert self.save_mode == other.save_mode
        assert self.nband == other.nband, \
            f"Adding results with different number of bands {self.nband} and {other.nband} - not allowed"
        results = {r: self.results[r] + other.results[r] for r in self.results if r in other.results}
        return TABresult(np.vstack((self.kpoints, other.kpoints)),
                         recip_lattice=self.recip_lattice,
                         results=results, mode=self.mode, save_mode=self.save_mode)

    def as_dict(self):
        raise NotImplementedError()

    def save(self, name):
        return  # do nothing so far

    def savetxt(self, name):
        return  # do nothing so far

    def savedata(self, name, prefix, suffix, i_iter):
        if i_iter > 0:
            pass  # so far do nothing on iterations, chang in future
        elif self.mode == "grid":
            self.self_to_grid()
            if "frmsf" in self.save_mode or "txt" in self.save_mode:
                write_frmsf(
                    frmsf_name=prefix + "-" + name,
                    Ef0=0., numproc=None,
                    quantities=self.results.keys(),
                    res=self,
                    suffix=suffix)  # so far let it be the only mode, implement other modes in future
            if "bin" in self.save_mode:
                write_npz(prefix + "-" + name,
                          quantities=self.results.keys(),
                          res=self,
                          suffix=suffix
                          )
            # TODO : remove this messy call to external routine, which calls back an internal one
        else:
            pass  # so far . TODO : implement writing to a text file

    def write_frmsf(self,
                    name,
                    quantity,
                    Ef0=0.,
                    numproc=None,
                    components=None):
        """
        Write the frmsf file for the given quantities and components
        Parameters
        ----------
        name : str
            name of the frmsf file to write (without .frmsf)
        quantity: str
            name of the quantity to write to the frmsf file.
            if None, only the energy is written
        components : list of str
            list of components to write to the frmsf file. if None, all components are written
        Ef0 : float
            Fermi energy
        numproc : int
            number of processes to use for writing the frmsf file
        Returns
        -------
        see `write_frmsf()`

        """
        return write_frmsf(name, Ef0=Ef0, numproc=numproc,
                           quantities=[quantity],
                           res=self,
                           components=[components])

    @property
    def find_grid(self):
        """ Find the full grid."""
        # TODO: make it cleaner, to work with iterations
        grid = np.zeros(3, dtype=int)
        kp = np.array(self.kpoints)
        kp = np.concatenate((kp, [[1, 1, 1]]))  # in case only k=0 is used in some direction
        for i in range(3):
            k = np.sort(kp[:, i])
            dk = np.max(k[1:] - k[:-1])
            grid[i] = int(np.round(1. / dk))
        return grid

    def transform(self, sym):
        results = {r: self.results[r].transform(sym) for r in self.results}
        kpoints = [sym.transform_reduced_vector(k, self.recip_lattice) for k in self.kpoints]
        return TABresult(kpoints=kpoints, recip_lattice=self.recip_lattice, results=results, mode=self.mode,
                         save_mode=self.save_mode)

    def to_grid(self, grid, order='C'):
        assert (self.mode == "grid")
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
                warnings.warn(f"k-point {ik}={self.kpoints[ik]} is not on the grid, skipping.")
        t0 = time()
        print("collecting")
        results = {r: self.results[r].to_grid(k_map) for r in self.results}
        t1 = time()
        print(f"collecting: to_grid  : {t1 - t0}")
        res = TABresult(k_new, recip_lattice=self.recip_lattice, results=results, mode=self.mode,
                        save_mode=self.save_mode)
        t2 = time()
        print(f"collecting: TABresult  : {t2 - t1}")
        res.grid = np.copy(grid)
        res.gridorder = order
        t3 = time()
        print(f"collecting - OK : {t3 - t0} ({t3 - t2})")
        return res

    def self_to_grid(self):
        res = self.to_grid(self.find_grid, order='C')
        self.__dict__.update(res.__dict__)  # another dirty trick, TODO : clean it

    def __get_data_grid(self, quantity, iband, component=None, efermi=None):
        if isinstance(iband, Iterable):
            shape = tuple(self.grid) + (len(iband),)
        else:
            shape = tuple(self.grid)
        if quantity == 'Energy':
            return self.Enk.data[:, iband].reshape(shape)
        elif component is None:
            return self.results[quantity].data[:, iband].reshape(shape + (3,) * self.results[quantity].rank)
        else:
            return self.results[quantity].get_component(component)[:, iband].reshape(shape)

    def __get_data_path(self, quantity, iband, component=None, efermi=None):
        if quantity == 'Energy':
            return self.Enk.data[:, iband]
        elif component is None:
            return self.results[quantity].data[:, iband]
        else:
            return self.results[quantity].get_component(component)[:, iband]

    def get_data(self, quantity, iband=None, component=None, efermi=None):
        if iband is None:
            iband = np.arange(self.nband)
        if self.grid is None:
            return self.__get_data_path(quantity, iband=iband, component=component, efermi=efermi)
        else:
            return self.__get_data_grid(quantity, iband=iband, component=component, efermi=efermi)

    def fermiSurfer(self, quantity=None, component=None, efermi=0, npar=0, iband=None, frmsf_name=None):
        if iband is None:
            iband = np.arange(self.nband)
        elif isinstance(iband, int):
            iband = [iband]
        if quantity is not None:
            Xnk = self.get_data(quantity=quantity, iband=iband, component=component)
        else:
            Xnk = None
        if self.grid is None:
            raise RuntimeError("the data should be on a grid before generating FermiSurfer files. use to_grid() method")
        if self.gridorder != 'C':
            raise RuntimeError("the data should be on a 'C'-ordered grid for generating FermiSurfer files")
        return fermiSurfer(recip_lattice=self.recip_lattice,
                           Enk=self.get_data(quantity="Energy", iband=iband),
                           data=Xnk,
                           efermi=efermi,
                           npar=npar,
                           frmsf_name=frmsf_name
                           )

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
            linecolor='black',
            close_fig=True,
            show_fig=True,
            axes=None,
            fig=None
    ):
        """
        a routine to plot a result along the path
        The circle size (size of quantity) changes linearly below 2 and logarithmically above 2.
        """
        import matplotlib.pyplot as plt
        if axes is None:
            axes = plt.gca()

        if fatmax is None:
            fatmax = fatfactor * 10

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

        axes.set_ylabel(r"$E$, eV")
        if Emin is None:
            Emin = E.min() - 0.5
        if Emax is None:
            Emax = E.max() + 0.5

        klineall = []
        for ib in range(len(iband)):
            e = E[:, ib]
            selE = (e <= Emax) * (e >= Emin)
            e[~selE] = None
            _line = None
            if np.any(selE):
                klineselE = kline[selE]
                klineall.append(klineselE)
                _kwargs = dict(c=linecolor)
                _kwargs.update(kwargs_line)
                if 'c' in _kwargs and 'color' in _kwargs:
                    del _kwargs['c']
                _line, = axes.plot(kline, e, **_kwargs)
        if None not in [label, _line]:
            _line.set_label(label)
            axes.legend()
        if cut_k:
            klineall = [k for kl in klineall for k in kl]
            if len(klineall) == 0:
                kmin = kline.min()
                kmax = kline.max()
            else:
                kmin = min(klineall)
                kmax = max(klineall)
        else:
            kmin = kline.min()
            kmax = kline.max()

        if quantity is not None:
            data = self.get_data(quantity=quantity, iband=iband, component=component)
            if mode == "fatband":
                for ib in range(len(iband)):
                    e = E[:, ib]
                    selE = (e <= Emax) * (e >= Emin)
                    klineselE = kline[selE]
                    y = data[selE][:, ib]
                    select = abs(y) > 2
                    y[select] = np.log2(abs(y[select])) * np.sign(y[select])
                    y[~select] *= 0.5
                    e1 = e[selE]
                    for col, sel in [("red", (y > 0)), ("blue", (y < 0))]:
                        sz = abs(y[sel]) * fatfactor
                        sz[sz > fatmax] = fatmax
                        axes.scatter(klineselE[sel], e1[sel], s=sz, color=col)
            else:
                raise ValueError("So far only fatband mode is implemented")

        x_ticks_labels = []
        x_ticks_positions = []
        for k, v in path.labels.items():
            x_ticks_labels.append(v)
            x_ticks_positions.append(kline[k])
            axes.axvline(x=kline[k])
        axes.set_xticks(x_ticks_positions, x_ticks_labels)
        axes.set_ylim([Emin, Emax])
        axes.set_xlim([kmin, kmax])

        if fig is None:
            fig = plt.gcf()

        if save_file is not None:
            fig.savefig(save_file)

        if show_fig:
            fig.show()

        if close_fig:
            plt.close(fig)
        else:
            return fig

    @property
    def max(self):
        return np.array([-1.])  # tabulating does not contribute to adaptive refinement


def write_frmsf(frmsf_name,
                Ef0,
                numproc,
                quantities,
                res,
                components=None,
                suffix=""):
    """
    Write the frmsf file for the given quantities and components

    Parameters
    ----------
    frmsf_name : str
        name of the frmsf file to write (without .frmsf)
    Ef0 : float
        Fermi energy
    numproc : int
        number of processes to use for writing the frmsf file
    quantities : list of str
        list of quantities to write to the frmsf file
    components : list of list of str
        list of components to write to the frmsf file. 
        a list of components for each quantity.
        if None, all components are written
    res : TABresult
        TABresult object containing the data to write
    suffix : str
        suffix to add to the frmsf file name. if empty, no suffix is added
        the resulting filename will be {frmsf_name}_{Q}-{comp}{suffix}.frmsf

    Returns
    -------
    ttxt : float
        time taken to write the text part of the frmsf file
    twrite : float
        time taken to write the binary part of the frmsf file
    """
    if len(suffix) > 0:
        suffix = "-" + suffix
    if frmsf_name is not None:
        if components is None:
            components = [None] * len(quantities)
        # open(f"{frmsf_name}_E{suffix}.frmsf", "w").write(res.fermiSurfer(quantity=None, efermi=Ef0, npar=numproc))
        ttxt = 0
        twrite = 0
        for Q, c in zip(quantities, components):
            if c is None:
                comp_loc = res.results[Q].get_component_list()
            else:
                comp_loc = c
            for comp in comp_loc:
                t31 = time()
                txt = res.fermiSurfer(quantity=Q, component=comp, efermi=Ef0, npar=numproc)
                t32 = time()
                open(f"{frmsf_name}_{Q}-{comp}{suffix}.frmsf", "w").write(txt)
                t33 = time()
                ttxt += t32 - t31
                twrite += t33 - t32
    else:
        ttxt = 0
        twrite = 0
    return ttxt, twrite


def write_npz(npz_name, quantities, res, suffix=""):
    """
    Write the data to a npz file

    Parameters
    ----------
    npz_name : str
        name of the npz file to write (without .npz)
    quantities : list of str
        list of quantities to write to the npz file
    res : TABresult
        TABresult object containing the data to write
    suffix : str
        suffix to add to the npz file name. if empty, no suffix is added
        the resulting filename will be {npz_name}{suffix}.npz

    Returns
    -------
    None
    """

    dic = {Q: res.get_data(quantity=Q) for Q in quantities}
    if len(suffix) > 0:
        suffix = "-" + suffix
    np.savez_compressed(f"{npz_name}{suffix}.npz",
                        recip_lattice=res.recip_lattice,
                        **dic
                        )


def _savetxt(limits=None, a=None, fmt=".8f", npar=0):
    assert a.ndim == 1, f"only 1D arrays are supported. found shape{a.shape}"
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
        print(f"using a pool of {npar} processes to write txt frmsf of {nppproc} points per process")
        asplit = [(i, i + nppproc) for i in range(0, a.shape[0], nppproc)]
        p = multiprocessing.Pool(npar)
        res = p.map(functools.partial(_savetxt, a=a, fmt=fmt, npar=0), asplit)
        p.close()
        p.join()
        return "".join(res)


def fermiSurfer(recip_lattice, Enk, data=None, efermi=0, npar=0, frmsf_name=None):
    print("Enk", Enk.shape)
    grid = Enk.shape[:3]
    nband = Enk.shape[3]
    FSfile = ""
    FSfile += f" {grid[0]}  {grid[1]}  {grid[2]} \n"
    FSfile += "1 \n"  # so far only this option of Fermisurfer is implemented
    FSfile += f"{nband} \n"
    FSfile += "".join(["  ".join(f"{x:14.8f}" for x in v) + "\n" for v in recip_lattice])
    for ib in range(nband):
        FSfile += _savetxt(a=Enk[:, :, :, ib].flatten(order='C') - efermi, npar=npar)
    if data is not None:
        print("Xnk", data.shape)
        assert data.shape[:4] == Enk.shape
        for ib in range(nband):
            FSfile += _savetxt(a=data[:, :, :, ib].flatten(order='C'), npar=npar)
    if frmsf_name is not None:
        if not (frmsf_name.endswith(".frmsf")):
            frmsf_name += ".frmsf"
        open(frmsf_name, "w").write(FSfile)
    return FSfile


def npz_to_fermisurfer(npz_file, quantity=None, frmsf_file=None, component=None):
    """
    Convert npz file to frmsf

    Parameters
    ----------
    npz_file : str
        name of the npz file to read
    frmsf_file : str
        name of the frmsf file to write. If `None` - not written, just returned as str
    quantity : str
        name of the quantity to write to the frmsf file
    component : str or tuple
        cartesian component, e.g. `'x'`, `'xxz'`, `(0,2)`, `'trace'`, `'norm'`

    Returns
    -------
    str
        the text of the frmsf file
    """
    res = np.load(npz_file)
    Enk = res['Energy']
    if quantity is not None:
        Xnk = res[quantity]
        if component is not None:
            Xnk = get_component(Xnk, ndim=Xnk.ndim - 4, component=component)
    else:
        Xnk = None
    recip_lattice = res['recip_lattice']
    return fermiSurfer(recip_lattice=recip_lattice, Enk=Enk, data=Xnk, frmsf_name=frmsf_file)
