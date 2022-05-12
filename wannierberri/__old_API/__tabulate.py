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
from collections import defaultdict
from wannierberri.result import TABresult, KBandResult
from wannierberri.formula import covariant as frml
from wannierberri.formula import covariant_basic as frml_basic

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

    results = {'Energy': KBandResult(data_K.E_K[:, ibands], TRodd=False, Iodd=False)}
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

        return KBandResult(rslt, TRodd=formula.TRodd, Iodd=formula.Iodd)




