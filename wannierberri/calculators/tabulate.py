import numpy as np
from .calculator import Calculator
from ..formula import covariant as frml
from ..result import KBandResult, TABresult


# The base classes for Tabulating
# particular calculators are below


class Tabulator(Calculator):

    def __init__(self, Formula, ibands=None, kwargs_formula=None, **kwargs):
        self.Formula = Formula
        self.ibands = np.array(ibands) if (ibands is not None) else None
        self.kwargs_formula = kwargs_formula if kwargs_formula is not None else {}
        super().__init__(**kwargs)

    def __call__(self, data_K):
        formula = self.Formula(data_K, **self.kwargs_formula)
        nk = data_K.nk
        NB = data_K.num_wann
        ibands = self.ibands
        if ibands is None:
            ibands = np.arange(NB)
        band_groups = data_K.get_bands_in_range_groups(
            -np.Inf, np.Inf, degen_thresh=self.degen_thresh, degen_Kramers=self.degen_Kramers, sea=False)
        # bands_groups  is a digtionary (ib1,ib2):E
        # now select only the needed groups
        band_groups = [
            [n for n in groups.keys() if np.any((ibands >= n[0]) * (ibands < n[1]))] for groups in band_groups
        ]  # select only the needed groups
        group = [[] for _ in range(nk)]
        for ik in range(nk):
            for ib in ibands:
                for n in band_groups[ik]:
                    if n[1] > ib >= n[0]:
                        group[ik].append(n)
                        break

        rslt = np.zeros((nk, len(ibands)) + (3,) * formula.ndim)
        for ik in range(nk):
            values = {}
            for n in band_groups[ik]:
                inn = np.arange(n[0], n[1])
                out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], NB)))
                values[n] = formula.trace(ik, inn, out) / (n[1] - n[0])
            for ib, b in enumerate(ibands):
                rslt[ik, ib] = values[group[ik][ib]]
        return KBandResult(rslt, transformTR=formula.transformTR, transformInv=formula.transformInv)


class TabulatorAll(Calculator):
    """
    TabulatorAll - a pack of all k-resolved calculators (Tabulators)
    """

    def __init__(self, tabulators, ibands=None, mode="grid", save_mode="bin", print_comment=False):
        """ tabulators - dict 'key':tabulator
        one of them should be "Energy" """
        self.tabulators = tabulators
        mode = mode.lower()
        assert mode in ("grid", "path")
        self.mode = mode
        self.save_mode = save_mode
        if "Energy" not in self.tabulators.keys():
            self.tabulators["Energy"] = Energy()
        if ibands is not None:
            ibands = np.array(ibands)
        for k, v in self.tabulators.items():
            if hasattr(v, 'ibands'):
                if v.ibands is not None:
                    try:
                        assert len(v.ibands) == len(ibands)
                        assert np.all(v.ibands == ibands)
                    except AssertionError:
                        raise ValueError(
                            f"tabulator {k} has ibands={v.ibands} not equal to ibands={ibands} required in TabulatorAll")
                else:
                    v.ibands = ibands

        self.comment = (self.__doc__ + "\n Includes the following tabulators : \n" + "-" * 50 + "\n" + "\n".join(
            f""" "{key}" : {val} : {val.comment}\n""" for key, val in self.tabulators.items()) +
            "\n" + "-" * 50 + "\n")
        self._set_comment(print_comment)

    def __call__(self, data_K):
        return TABresult(
            kpoints=data_K.kpoints_all.copy(),
            mode=self.mode,
            recip_lattice=data_K.system.recip_lattice,
            save_mode=self.save_mode,
            results={k: v(data_K)
                     for k, v in self.tabulators.items()})

    @property
    def allow_path(self):
        return self.mode == "path"

    @property
    def allow_grid(self):
        return self.mode == "grid"


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


class Energy(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Eavln, **kwargs)


class Velocity(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Velocity, **kwargs)


class InvMass(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.InvMass, **kwargs)


class Der3E(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Der3E, **kwargs)


class BerryCurvature(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Omega, **kwargs)


class DerBerryCurvature(Tabulator):
    r"Derivative of Berry curvature :math:`X_{ab}\partial_b\Omega_a`"

    def __init__(self, **kwargs):
        super().__init__(frml.DerOmega, **kwargs)


class Der2BerryCurvature(Tabulator):
    r"Second Derivative of Berry curvature :math:`X_{ab}\partial_bc\Omega_a`"

    def __init__(self, **kwargs):
        super().__init__(frml.Der2Omega, **kwargs)


class Spin(Tabulator):
    r""" Spin expectation :math:` \langle u | \mathbf{\sigma} | u \rangle`"""

    def __init__(self, **kwargs):
        super().__init__(frml.Spin, **kwargs)


class DerSpin(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.DerSpin, **kwargs)


class Der2Spin(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Der2Spin, **kwargs)


class OrbitalMoment(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.morb, **kwargs)


class DerOrbitalMoment(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Dermorb, **kwargs)


class DerOrbitalMoment_test(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.DerMorb_test, **kwargs)


class Der2OrbitalMoment(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Der2morb, **kwargs)


class SpinBerry(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.SpinOmega, **kwargs)

class Quantum_M(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Quantum_M, **kwargs)

class Quantum_M_kp(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Quantum_M_kp, **kwargs)

class BerryCurvature_kp(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Omegakp, **kwargs)

class OrbitalMoment_kp(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Morbkp, **kwargs)


class BCP_G(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.BCP_G_kp, **kwargs)

class BCP_G_kp(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.BCP_G_kp, **kwargs)
