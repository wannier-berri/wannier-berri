
from functools import lru_cache

import numpy as np

sq2 = np.sqrt(1 / 2)
jsq2 = 1j * np.sqrt(1 / 2)

# from here:
real_from_complex = {"s": [0],
                    "pz": [0, 1, 0],  # in basis [-l, ... , +l]
                    "px": [sq2, 0, -sq2],
                    "py": [jsq2, 0, jsq2],
                    "dxy": [jsq2, 0, 0, 0, -jsq2],
                    "dyz": [0, jsq2, 0, jsq2, 0],
                    "dz2": [0, 0, 1, 0, 0],
                    "dxz": [0, sq2, 0, -sq2, 0],
                    "dx2-y2": [sq2, 0, 0, 0, sq2],
                    "fxyz": [0, jsq2, 0, 0, 0, -jsq2, 0],
                    "fxz2": [0, 0, sq2, 0, -sq2, 0, 0],
                    "fyz2": [0, 0, jsq2, 0, jsq2, 0, 0],
                    "fz3": [0, 0, 0, 1, 0, 0, 0],
                    "fzx2-zy2": [0, sq2, 0, 0, 0, sq2, 0],
                    "fx3-3xy2": [sq2, 0, 0, 0, 0, 0, -sq2],
                    "f3yx2-y3": [jsq2, 0, 0, 0, 0, 0, jsq2]}



r2, r3, r6, r10 = np.sqrt(2), np.sqrt(3), np.sqrt(6), np.sqrt(10)


def _build(h, l):
    """Lx, Ly, Lz from the half ladder elements h[k] = <m+1|L+|m>/2, m = -l..l-1"""
    Lx = np.diag(h, 1) + np.diag(h, -1)
    Ly = 1j * np.diag(h, 1) - 1j * np.diag(h, -1)
    Lz = np.diag(np.arange(-l, l + 1)).astype(complex)
    return np.array([Lx, Ly, Lz], dtype=complex)


# s: m = 0
Lm_s = np.zeros((3, 1, 1), dtype=complex)
# p: m = -1, 0, 1
Lm_p = _build(np.array([r2 / 2, r2 / 2]), 1)
# d: m = -2..2   (off-diagonal elements: 1, sqrt6/2, sqrt6/2, 1)
Lm_d = _build(np.array([1, r6 / 2, r6 / 2, 1]), 2)
# f: m = -3..3   (off-diagonal elements: sqrt6/2, sqrt10/2, sqrt3, sqrt3, sqrt10/2, sqrt6/2)
Lm_f = _build(np.array([r6 / 2, r10 / 2, r3, r3, r10 / 2, r6 / 2]), 3)

Lm = {"s": Lm_s, "p": Lm_p, "d": Lm_d, "f": Lm_f}


l_shell = {"s": 0, "p": 1, "d": 2, "f": 3}


@lru_cache
def get_l_shell(orb_tuple):
    shell = set(s[0] for s in orb_tuple)
    if len(shell) != 1:
        raise ValueError(f"Orbitals {orb_tuple} are not from the same shell")
    shell = shell.pop()
    assert shell in Lm, f"Shell {shell} is not supported"
    assert len(set(orb_tuple)) == 1 + 2 * l_shell[shell], f"Orbitals {orb_tuple} are not complete for shell {shell}"
    Lm_shell = Lm[shell]
    matrix = np.array([real_from_complex[orb] for orb in orb_tuple], dtype=complex)
    L_shell = np.einsum("ijk,lj,mk->ilm", Lm_shell, matrix.conj(), matrix)
    return L_shell
