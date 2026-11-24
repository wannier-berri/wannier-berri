


from ..fourier.rvectors import merge_Rvectors

import numpy as np
import copy
import logging
logger = logging.getLogger(__name__)



class SystemInterpolator:
    """
    In terploates between several systems

    Parameters
    ----------
    systems_list : list
        A list of systems to interpolate between.
    use_pointgroup : int
        selects which pointgroup to use for the interpolation. (selects systems_list[use_pointgroup].pointgroup). 
        If None, no pointgroup will be used.
    check_wcc_precision : float
        The precision to check the WCC (Wannier Charge Centers) for each system.
    """

    def __init__(self, systems_list, use_pointgroup=0, check_wcc_precision=0.1):
        self.systems_list = [copy.deepcopy(sys) for sys in systems_list]
        self.rvec, rvectors_map_list = merge_Rvectors([system.rvec for system in self.systems_list])

        self.pointgroup = self.systems_list[use_pointgroup].pointgroup if use_pointgroup is not None else None

        matrix_keys = [set(system._XX_R.keys()) for system in self.systems_list]
        # set of keys that are present in only one of the systems
        # matrix_keys_common = matrix_keys0.intersection(matrix_keys1)
        matrix_keys_all = set.union(*matrix_keys)
        matrix_keys_intersection = set.intersection(*matrix_keys)
        matrix_keys_exclude = matrix_keys_all - matrix_keys_intersection
        if len(matrix_keys_exclude) > 0:
            logger.warning("The following matrix elements are present NIT in all systems: %s , they will be excluded from the interpolation", matrix_keys_exclude)
        for sys, iRmap  in zip(self.systems_list, rvectors_map_list):
            for key in matrix_keys_exclude:
                if key in sys._XX_R:
                    del sys._XX_R[key]
            for key in sys._XX_R:
                shape = sys._XX_R[key].shape
                new_matrix = np.zeros((self.rvec.nRvec,) + shape[1:], dtype=complex)
                new_matrix[iRmap] = sys._XX_R[key]
                sys._XX_R[key] = new_matrix
            shifts_left, shifts_right = copy.copy(self.rvec.shifts_left_red), copy.copy(self.rvec.shifts_right_red)
            sys.rvec = copy.deepcopy(self.rvec)
            sys.rvec.shifts_left_red, self.rvec.shifts_right_red = shifts_left, shifts_right
            sys.clear_cached_R()

        if check_wcc_precision is not None:
            wcc_mean = np.mean([system.wannier_centers_cart for system in self.systems_list], axis=0)
            for i, sys in enumerate(self.systems_list):
                wcc_diff = np.max(np.abs(sys.wannier_centers_cart - wcc_mean))
                if wcc_diff > check_wcc_precision:
                    raise ValueError(f"WCC of system {i} differs from the mean by {wcc_diff:.3f} Angstrom, which is larger than the allowed precision of {check_wcc_precision:.3f} Angstrom. Please check the systems or increase the check_wcc_precision parameter.")


    def interpolate(self, weights):
        weights = np.array(weights)
        assert len(weights) == len(self.systems_list), "weights must have the same length as systems_list"
        assert abs(weights.sum() - 1) < 1e-8, f"weights must sum to 1, but sum({weights}) = {np.sum(weights)}"
        new_system = copy.deepcopy(self.systems_list[0])
        numn_elec_list = [system.number_of_electrons for system in self.systems_list]
        if None not in numn_elec_list:
            new_system.number_of_electrons = weights @ numn_elec_list
        else:
            new_system.number_of_electrons = None
        new_system.wannier_centers_cart = sum([w * system.wannier_centers_cart for w, system in zip(weights, self.systems_list)])
        for key in self.systems_list[0]._XX_R:
            new_system._XX_R[key] = sum([w * system._XX_R[key] for w, system in zip(weights, self.systems_list)])
        new_system.rvec.shifts_left_red = sum([w * system.rvec.shifts_left_red for w, system in zip(weights, self.systems_list)])
        new_system.rvec.shifts_right_red = sum([w * system.rvec.shifts_right_red for w, system in zip(weights, self.systems_list)])
        new_system.set_pointgroup(pointgroup=self.pointgroup)
        return new_system


class SystemInterpolatorSOC(SystemInterpolator):
    """
    Interpolates between two systems with SOC

    Parameters
    ----------
    systems_list : list of SystemSOC
        The list of systems to interpolate between
    use_pointgroup : int
        If 0, the pointgroup of system0 will be used, if 1, the pointgroup of the system1 will be used. If -1, no pointgroup will be used. 
    """

    def __init__(self, systems_list, use_pointgroup=0, check_wcc_precision=0.1):
        sys_soc_part_list = []
        sys_up_list = []
        sys_down_list = []
        for sys in systems_list:
            sys1 = copy.deepcopy(sys)
            sys_up_list.append(sys1.system_up)
            delattr(sys1, 'system_up')
            if hasattr(sys1, 'system_down'):
                sys_down_list.append(sys1.system_down)
                delattr(sys1, 'system_down')
            else:
                sys_down_list.append(None)
            sys_soc_part_list.append(sys1)
        super().__init__(sys_soc_part_list, use_pointgroup, check_wcc_precision)
        self.interpolator_up = SystemInterpolator(systems_list=sys_up_list, use_pointgroup=use_pointgroup, check_wcc_precision=check_wcc_precision)
        if all(sys is None for sys in sys_down_list):
            self.interpolator_down = None
        else:
            sys_down_list = [sys_down if sys_down is not None else sys_up for sys_down, sys_up in zip(sys_down_list, sys_up_list)]
            self.interpolator_down = SystemInterpolator(systems_list=sys_down_list, use_pointgroup=use_pointgroup, check_wcc_precision=check_wcc_precision)


    def interpolate(self, weights):
        new_system = super().interpolate(weights)
        new_system.system_up = self.interpolator_up.interpolate(weights)
        if self.interpolator_down is not None:
            new_system.system_down = self.interpolator_down.interpolate(weights)
            new_system.nspin = 2
        else:
            new_system.system_down = new_system.system_up
            new_system.nspin = 1
        return new_system
