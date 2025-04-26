from .w90file import W90_file
import numpy as np


class EIG(W90_file):

    def __init__(self, seedname="wannier90", **kwargs):
        self.npz_tags = ["data"]
        super().__init__(seedname=seedname, ext="eig", **kwargs)

    def from_w90_file(self, seedname):
        data = np.loadtxt(seedname + ".eig")
        NB = int(round(data[:, 0].max()))
        NK = int(round(data[:, 1].max()))
        data = data.reshape(NK, NB, 3)
        assert np.linalg.norm(data[:, :, 0] - 1 - np.arange(NB)[None, :]) < 1e-15
        assert np.linalg.norm(data[:, :, 1] - 1 - np.arange(NK)[:, None]) < 1e-15
        self.data = data[:, :, 2]

    def to_w90_file(self, seedname):
        file = open(seedname + ".eig", "w")
        for ik in range(self.NK):
            for ib in range(self.NB):
                file.write(f" {ib + 1:4d} {ik + 1:4d} {self.data[ik, ib]:17.12f}\n")

    def select_bands(self, selected_bands):
        if selected_bands is not None:
            self.data = self.data[:, selected_bands]


    # def get_disentangled(self, v_left, v_right):
    #     data = np.einsum("klm,km...,kml->kl", v_left, self.data, v_right).real
    #     return self.__class__(data=data)
