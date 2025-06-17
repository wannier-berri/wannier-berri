from .w90file import W90_file, check_shape, auto_kptirr
import numpy as np


class EIG(W90_file):

    extension = "eig"

    def __init__(self, data, NK=None):
        super().__init__(data=data, NK=NK)
        self.NB = check_shape(self.data)[0]


    @classmethod
    def from_w90_file(cls, seedname, selected_kpoints=None):
        data = np.loadtxt(seedname + ".eig")
        NB = int(round(data[:, 0].max()))
        NK = int(round(data[:, 1].max()))
        if selected_kpoints is None:
            selected_kpoints = np.arange(NK)
        data = data.reshape(NK, NB, 3)
        assert np.linalg.norm(data[:, :, 0] - 1 - np.arange(NB)[None, :]) < 1e-15
        assert np.linalg.norm(data[:, :, 1] - 1 - np.arange(NK)[:, None]) < 1e-15
        data = data[:, :, 2]
        data = {ik: data[ik] for ik in selected_kpoints}
        return EIG(data=data, NK=NK)

    def to_w90_file(self, seedname):
        file = open(seedname + ".eig", "w")
        for ik in range(self.NK):
            for ib in range(self.NB):
                file.write(f" {ib + 1:4d} {ik + 1:4d} {self.data[ik, ib]:17.12f}\n")


    @classmethod
    def from_bandstructure(cls, bandstructure, selected_kpoints=None,
                           kptirr=None,
                           verbose=False,
                           NK=None):
        """
        Create an EIG object from a BandStructure object

        Parameters
        ----------
        bandstructure : bandstructure : irrep.bandstructure.BandStructure

            the band structure object
        """
        NK, selected_kpoints, kptirr = auto_kptirr(
            bandstructure, selected_kpoints=selected_kpoints, kptirr=kptirr, NK=NK)

        if verbose:
            print("Creating eig.")
        data = {}
        for ikirr in kptirr:
            data[ikirr] = bandstructure.kpoints[selected_kpoints[ikirr]].Energy_raw
        return EIG(data=data, NK=NK)
