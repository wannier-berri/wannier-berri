import numpy as np
from collections.abc import Iterable
from .grid import Grid
from .data_K import get_data_k
from collections import defaultdict
from .calculators import tabulate

available_quantities = {
    "energy": tabulate.Energy(print_comment=False),
    "band_gradients": tabulate.Velocity(print_comment=False, kwargs_formula={"external_terms": False}),
    "berry_curvature": tabulate.BerryCurvature(print_comment=False),
    "berry_curvature_internal_terms": tabulate.BerryCurvature(kwargs_formula={"external_terms": False}, print_comment=False),
    "berry_curvature_external_terms": tabulate.BerryCurvature(kwargs_formula={"internal_terms": False}, print_comment=False),
}

comments = {
    "berry_curvature_internal_terms": "only the internal terms are evaluated",
    "berry_curvature_external_terms": "only the external terms are evaluated",
}


def hlp():
    _hlp = "Available named quantities:\n"
    for q, c in available_quantities.items():
        _hlp += f" '{q}' : {c.comment} \n"
        if q in comments:
            _hlp += f"       Note : {comments[q]}\n"
    return _hlp


def evaluate_k(system=None,
               k=(0., 0., 0.),
               quantities=(),
               calculators=None,
               formula=None,
               param_formula=None,
               iband=None,
               return_single_as_dict=False,
               parameters_K=None,
              ):
    """This function presents a shortcut to evaluate some property at a particular k-point
    The main goal is to be convenient, rather than efficient

    Parameters
    ----------
    system : :class:`~wannierberri.system.System`
        system for which the calculation is performed
    k : tuple
        3 numbers denoting the k-vector in reciprocal lattice vectors (reduced coordinates)
    quantities : list of str
        Which qpre-defined quantities to calculate. Use `evaluate()` to get the list of available quantities
    calculators : dict
        Dictionary str : :class:`~wannierberri.calculators.Calculator`
        alows to evaluate quantities that are not available here, or using specific parameters
    formula : dict
        Dictionary str : :class:`~wannierberri.formula.Formula_ln`
        allows to evaluate block-diagonal part of a covariant formula (considering iband as "inner" states)
    param_formula : dict
        Dictionary str : dict -parameters to be passed to the corresponding formula
    return_single_as_dict : bool
        wether to pack a result into dict if only one calculator/quantity is requested
    parameters_K: dict
        parameters to be passed to :class:`~wannierberri.data_K.Data_K` class
    iband : int or list(int)
        bands to select (only for `quantities` and `formula`
    Returns
    -------
    dict or array
        Returns a dictionary {key:array} where keys are strings identifying the calculated quantities,
        if only one quantity is calculated and  `return_single_as_dict=False`, then only an array is returned,
        without packing into dict
    """

    if calculators is None:
        calculators = {}
    if formula is None:
        formula = {}
    if param_formula is None:
        param_formula = {}
    if parameters_K is None:
        parameters_K = {}

    # without parameters - just print the available options
    if system is None:
        print(__doc__)
        print(hlp())
        return

    set1 = set(quantities)
    set2 = set(formula.keys())
    set3 = set(calculators.keys())
    try:
        assert len(set1.intersection(set2)) == 0
        assert len(set2.intersection(set3)) == 0
        assert len(set1.intersection(set3)) == 0
    except AssertionError:
        raise ValueError("names of calculators, formula and quantities should be unique")

    grid = Grid(system, NK=1, NKFFT=1)
    data_k = get_data_k(system, grid=grid, dK=k, **parameters_K)

    result = {c: calc(data_k) for c, calc in calculators.items()}

    if iband is None:
        iband = np.arange(system.num_wann)
    if not isinstance(iband, Iterable):
        iband = [iband]
    iband_out = sorted(np.array(list(set(range(system.num_wann)) - set(iband))))

    for q in quantities:
        if q not in available_quantities:
            raise ValueError(f"unknown quantity {q}. known quantities are {available_quantities.keys()}")
        result[q] = available_quantities[q](data_k).data[0][list(iband)]

    _param_formula = defaultdict(lambda: {})
    _param_formula.update(param_formula)

    for k, f in formula.items():
        form = f(data_k, **_param_formula[k])
        result[k] = form.nn(0, iband, iband_out)

    if not return_single_as_dict:
        if len(result) == 1:
            result = list(result.values())[0]

    return result
