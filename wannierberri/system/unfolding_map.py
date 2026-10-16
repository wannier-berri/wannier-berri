
import numpy as np
from ..utility import cached_einsum


def unfolded_weights(system_SC, 
                     projections, 
                     k_PC,
                     transform_PC_to_SC,
                     ecut=300, 
                     normalize=True,
                     param_degen_groups=None,
                     ):
    from ..data_K import get_data_k_class_from_system
    data_K_class = get_data_k_class_from_system(system_SC)
    data_k = data_K_class(system_SC, dK=k, grid=[1,1,1])
    k_SC = (transform_PC_to_SC @ k_PC ) %1
    E_K = data_k.E_K
    U_K = data_k.UU_K
    proj_gk, igall_SC, Eg = projections.get_proj_gk_ecut(k, ecut)
    igall_PC = np.linalg.inv(transform_PC_to_SC) @ igall_SC[:,:3].T - k_PC[:,None]
    igall_PC_int = np.round(igall_PC).astype(int)
    selected = np.all(np.isclose(igall_PC, igall_PC_int), axis=1)

    UUg = cached_einsum('mn,mg->ng', U_K[0], proj_gk)
    if normalize:
        UUg/= np.linalg.norm(UUg, axis=1)[None, :]
    Unorm = np.linalg.norm(UUg[:,selected], axis=1)**2

    degen_groups = data_k.get_bands_in_range_groups_ik(ik=0, **param_degen_groups)

    weights = { e: Unorm[ib1:ib2].sum() for (ib1, ib2), e in degen_groups.items() }
    return weights