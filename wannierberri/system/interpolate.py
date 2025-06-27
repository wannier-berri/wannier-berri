



import copy
import warnings

import numpy as np
from ..fourier.rvectors import Rvectors


class SystemInterpolator:
    """
    In terploates between two systems

    Parameters
    ----------
    system0 : System_R
        The first system (corresponds to alpha = 0)
    system1 : System_R
        The second system (corresponds to alpha = 1) 
    use_pointgroup : int
        If 0, the pointgroup of system0 will be used, if 1, the pointgroup of the system1 will be used. If -1, no pointgroup will be used. 
    """

    def __init__(self, system0, system1, use_pointgroup=1):
        self.system0 = copy.deepcopy(system0)
        self.system1 = copy.deepcopy(system1)
        iRvec0_list = [tuple(ir) for ir in self.system0.rvec.iRvec]
        iRvec1_list = [tuple(ir) for ir in self.system1.rvec.iRvec]
        iRvec_new_list = list(set(iRvec0_list).union(set(iRvec1_list)))
        iRvec_index = {ir: i for i, ir in enumerate(iRvec_new_list)}
        iRvec_map_0 = [iRvec_index[ir] for ir in iRvec0_list]
        iRvec_map_1 = [iRvec_index[ir] for ir in iRvec1_list]
        if use_pointgroup == 1:
            self.pointgroup = self.system1.pointgroup
        elif use_pointgroup == 0:
            self.pointgroup = self.system0.pointgroup
        elif use_pointgroup < 0:
            self.pointgroup = None
        else:
            raise ValueError("use_pointgroup should be 0, 1 or -1")
        iRvec_array = np.array(iRvec_new_list)

        matrix_keys0 = set(self.system0._XX_R.keys())
        matrix_keys1 = set(self.system1._XX_R.keys())
        # set of keys that are present in only one of the systems
        # matrix_keys_common = matrix_keys0.intersection(matrix_keys1)
        matrix_keys_all = matrix_keys0.union(matrix_keys1)
        matrix_keys_exclude = matrix_keys_all - matrix_keys0.intersection(matrix_keys1)
        if len(matrix_keys_exclude) > 0:
            warnings.warn(f"The following matrix elements are present in only one of the systems: {matrix_keys_exclude} , they will be excluded from the interpolation")
        for sys, iRmap  in zip([self.system0, self.system1], [iRvec_map_0, iRvec_map_1]):
            for key in sys._XX_R:
                if key in matrix_keys_exclude:
                    del sys._XX_R[key]
                else:
                    shape = sys._XX_R[key].shape
                    new_matrix = np.zeros((len(iRvec_new_list),) + shape[1:], dtype=complex)
                    new_matrix[iRmap] = sys._XX_R[key]
                    sys._XX_R[key] = new_matrix
            sys.rvec = Rvectors(lattice=sys.rvec.lattice, iRvec=iRvec_array, shifts_left_red=sys.rvec.shifts_left_red, shifts_right_red=sys.rvec.shifts_right_red)
            sys.clear_cached_R()


    def interpolate(self, alpha):
        new_system = copy.deepcopy(self.system0)
        new_system.wannier_centers_cart = (1 - alpha) * self.system0.wannier_centers_cart + alpha * self.system1.wannier_centers_cart
        for key in self.system0._XX_R:
            new_system._XX_R[key] = (1 - alpha) * self.system0._XX_R[key] + alpha * self.system1._XX_R[key]
        new_system.set_pointgroup(pointgroup=self.pointgroup)
        return new_system
