import numpy as np
from . import Calculator
from wannierberri.result import KBandBandResult


# The base class for Tabulating
# particular calculators are below


class TabulatorOD(Calculator):

    def __init__(self, kwargs_formula={}):
        self.ibands = None
        self.jbands = None
        self.kwargs_formula = kwargs_formula
        super().__init__()

#    def matrix(self,data_K):
#        raise NotImplementedError()

    def __call__(self, data_K):
        # formula = self.Formula(data_K, **self.kwargs_formula)
        # nk = data_K.nk
        NB = data_K.num_wann
        ibands = self.ibands
        jbands = self.jbands
        if ibands is None:
            ibands = np.arange(NB)
        if jbands is None:
            jbands = np.arange(NB)

        rslt = self.matrix(data_K)[:,ibands][:,:,jbands]
        return KBandBandResult(rslt)


###############################################
###############################################
###############################################
###############################################
####                                     ######
####        Implemented calculators      ######
####                                     ######
###############################################
###############################################
###############################################
###############################################




class BerryConnection(TabulatorOD):

    def matrix(self,data_K):
        return data_K.A_H

