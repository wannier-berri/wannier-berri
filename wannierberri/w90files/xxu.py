import numpy as np
from .w90file import W90_file
from .utility import readstr
from ..io import FortranFileR


class UXU(W90_file):
    """
    Read and setup uHu or uIu object.
    pw2wannier90 writes data_pw2w90[n, m, ib1, ib2, ik] = <u_{m,k+b1}|X|u_{n,k+b2}>
    in column-major order. (X = H for UHU, X = I for UIU.)
    Here, we read to have data[ik, ib1, ib2, m, n] = <u_{m,k+b1}|X|u_{n,k+b2}>.
    """

    @property
    def n_neighb(self):
        """	
        number of nearest neighbours indices
        """
        return 2


    def from_w90_file(self, seedname='wannier90', suffix='uXu', formatted=False):
        print(f"----------\n  {suffix}   \n---------")
        print(f'formatted == {formatted}')
        if formatted:
            f_uXu_in = open(seedname + "." + suffix, 'r')
            header = f_uXu_in.readline().strip()
            NB, NK, NNB = (int(x) for x in f_uXu_in.readline().split())
        else:
            f_uXu_in = FortranFileR(seedname + "." + suffix)
            header = readstr(f_uXu_in)
            NB, NK, NNB = f_uXu_in.read_record('i4')

        print(f"reading {seedname}.{suffix} : <{header}>")

        self.data = np.zeros((NK, NNB, NNB, NB, NB), dtype=complex)
        if formatted:
            tmp = np.array([f_uXu_in.readline().split() for i in range(NK * NNB * NNB * NB * NB)], dtype=float)
            tmp_cplx = tmp[:, 0] + 1.j * tmp[:, 1]
            self.data = tmp_cplx.reshape(NK, NNB, NNB, NB, NB).transpose(0, 2, 1, 3, 4)
        else:
            for ik in range(NK):
                for ib2 in range(NNB):
                    for ib1 in range(NNB):
                        tmp = f_uXu_in.read_record('f8').reshape((2, NB, NB), order='F').transpose(2, 1, 0)
                        self.data[ik, ib1, ib2] = tmp[:, :, 0] + 1j * tmp[:, :, 1]
        print(f"----------\n {suffix} OK  \n---------\n")
        f_uXu_in.close()

    def select_bands(self, selected_bands):
        if selected_bands is not None:
            self.data = self.data[:, :, :, selected_bands, :][:, :, :, :, selected_bands]


class UHU(UXU):
    """
    UHU.data[ik, ib1, ib2, m, n] = <u_{m,k+b1}|H(k)|u_{n,k+b2}>
    """

    def __init__(self, seedname='wannier90', **kwargs):
        super().__init__(seedname=seedname, ext='uHu', suffix='uHu', **kwargs)


class UIU(UXU):
    """
    UIU.data[ik, ib1, ib2, m, n] = <u_{m,k+b1}|u_{n,k+b2}>
    """

    def __init__(self, seedname='wannier90', **kwargs):
        super().__init__(seedname=seedname, ext='uIu', suffix='uIu', **kwargs)


class SXU(W90_file):
    """
    Read and setup sHu or sIu object.
    pw2wannier90 writes data_pw2w90[n, m, ipol, ib, ik] = <u_{m,k}|S_ipol * X|u_{n,k+b}>
    in column-major order. (X = H for SHU, X = I for SIU.)
    Here, we read to have data[ik, ib, m, n, ipol] = <u_{m,k}|S_ipol * X|u_{n,k+b}>.
    """

    @property
    def n_neighb(self):
        """	
        number of nearest neighbours indices
        """
        return 1

    def from_w90_file(self, seedname='wannier90', formatted=False, suffix='sHu', **kwargs):
        """	
        Read the sHu or sIu file

        Parameters
        ----------
        seedname : str
            the prefix of the file (including relative/absolute path, but not including the extensions, like `.sHu`, `sIu`)   
        formatted : bool
            if True, the file is expected to be formatted, otherwise it is binary
        suffix : str
            the suffix of the file, e.g. 'sHu', 'sIu'
        kwargs : dict(str, Any)
            the keyword arguments to be passed to the constructor of the file
            see `~wannierberri.system.w90_files.SIU`, `~wannierberri.system.w90_files.SHU`
            for more details

        Raises
        ------
        AssertionError
            if the file does not conform with the other files

        Sets 
        -----
        self.data : numpy.ndarray(complex, shape=(NK, NNB, NB, NB, 3)
            the data of the file
        """

        print(f"----------\n  {suffix}   \n---------")

        if formatted:
            f_sXu_in = open(seedname + "." + suffix, 'r')
            header = f_sXu_in.readline().strip()
            NB, NK, NNB = (int(x) for x in f_sXu_in.readline().split())
        else:
            f_sXu_in = FortranFileR(seedname + "." + suffix)
            header = readstr(f_sXu_in)
            NB, NK, NNB = f_sXu_in.read_record('i4')

        print(f"reading {seedname}.{suffix} : <{header}>")

        self.data = np.zeros((NK, NNB, NB, NB, 3), dtype=complex)

        if formatted:
            tmp = np.array([f_sXu_in.readline().split() for i in range(NK * NNB * 3 * NB * NB)], dtype=float)
            tmp_cplx = tmp[:, 0] + 1j * tmp[:, 1]
            self.data = tmp_cplx.reshape(NK, NNB, 3, NB, NB).transpose(0, 1, 3, 4, 2)
        else:
            for ik in range(NK):
                for ib in range(NNB):
                    for ipol in range(3):
                        tmp = f_sXu_in.read_record('f8').reshape((2, NB, NB), order='F').transpose(2, 1, 0)
                        # tmp[m, n] = <u_{m,k}|S_ipol*X|u_{n,k+b}>
                        self.data[ik, ib, :, :, ipol] = tmp[:, :, 0] + 1j * tmp[:, :, 1]

        print(f"----------\n {suffix} OK  \n---------\n")
        f_sXu_in.close()

    def select_bands(self, selected_bands):
        if selected_bands is not None:
            self.data = self.data[:, :, :, selected_bands, :][:, :, :, :, selected_bands]


class SIU(SXU):
    """
    SIU.data[ik, ib, m, n, ipol] = <u_{m,k}|S_ipol|u_{n,k+b}>

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extensions, like `.sIu`, `sHu`)
    formatted : bool
        if True, the file is expected to be formatted, otherwise it is binary
    kwargs : dict(str, Any)
        the keyword arguments to be passed to the parent constructor,
        see `~wannierberri.system.w90_files.SXU` for more details
    """

    def __init__(self, seedname='wannier90', formatted=False, **kwargs):
        super().__init__(seedname=seedname, ext='sIu', formatted=formatted, suffix='sIu', **kwargs)


class SHU(SXU):
    """
    SHU.data[ik, ib, m, n, ipol] = <u_{m,k}|S_ipol*H(k)|u_{n,k+b}>

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extensions, like `.sHu`, `sIu`)
    formatted : bool
        if True, the file is expected to be formatted, otherwise it is binary
    kwargs : dict(str, Any)
        the keyword arguments to be passed to the parent constructor,
        see `~wannierberri.system.w90_files.SXU` for more details
    """

    def __init__(self, seedname='wannier90', formatted=False, **kwargs):
        super().__init__(seedname=seedname, ext='sHu', formatted=formatted, suffix='sHu', **kwargs)
