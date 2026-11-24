import pytest

from wannierberri.symmetry.angular_momentum import get_l_shell
import numpy as np


r3 = np.sqrt(3)
r6 = np.sqrt(6)
r6h = np.sqrt(6) / 2
r10h = np.sqrt(10) / 2

# s: ['s']
L_s = np.zeros((3, 1, 1), dtype=complex)

# p: ['pz', 'px', 'py']
L_p = np.array([
    # Lx
    [[0, 0, 1j],
     [0, 0, 0],
     [-1j, 0, 0]],
    # Ly
    [[0, -1j, 0],
     [1j, 0, 0],
     [0, 0, 0]],
    # Lz
    [[0, 0, 0],
     [0, 0, -1j],
     [0, 1j, 0]],
], dtype=complex)

# d: ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy']
L_d = np.array([
    # Lx
    [[0, 0, 1j * r3, 0, 0],
     [0, 0, 0, 0, 1j],
     [-1j * r3, 0, 0, -1j, 0],
     [0, 0, 1j, 0, 0],
     [0, -1j, 0, 0, 0]],
    # Ly
    [[0, -1j * r3, 0, 0, 0],
     [1j * r3, 0, 0, -1j, 0],
     [0, 0, 0, 0, -1j],
     [0, 1j, 0, 0, 0],
     [0, 0, 1j, 0, 0]],
    # Lz
    [[0, 0, 0, 0, 0],
     [0, 0, -1j, 0, 0],
     [0, 1j, 0, 0, 0],
     [0, 0, 0, 0, -2j],
     [0, 0, 0, 2j, 0]],
], dtype=complex)

# f: ['fz3', 'fxz2', 'fyz2', 'fzx2-zy2', 'fxyz', 'fx3-3xy2', 'f3yx2-y3']
L_f = np.array([
    # Lx
    [[0, 0, 1j * r6, 0, 0, 0, 0],
     [0, 0, 0, 0, 1j * r10h, 0, 0],
     [-1j * r6, 0, 0, -1j * r10h, 0, 0, 0],
     [0, 0, 1j * r10h, 0, 0, 0, 1j * r6h],
     [0, -1j * r10h, 0, 0, 0, -1j * r6h, 0],
     [0, 0, 0, 0, 1j * r6h, 0, 0],
     [0, 0, 0, -1j * r6h, 0, 0, 0]],
    # Ly
    [[0, -1j * r6, 0, 0, 0, 0, 0],
     [1j * r6, 0, 0, -1j * r10h, 0, 0, 0],
     [0, 0, 0, 0, -1j * r10h, 0, 0],
     [0, 1j * r10h, 0, 0, 0, -1j * r6h, 0],
     [0, 0, 1j * r10h, 0, 0, 0, -1j * r6h],
     [0, 0, 0, 1j * r6h, 0, 0, 0],
     [0, 0, 0, 0, 1j * r6h, 0, 0]],
    # Lz
    [[0, 0, 0, 0, 0, 0, 0],
     [0, 0, -1j, 0, 0, 0, 0],
     [0, 1j, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, -2j, 0, 0],
     [0, 0, 0, 2j, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, -3j],
     [0, 0, 0, 0, 0, 3j, 0]],
], dtype=complex)

L_dict = {"s": L_s, "p": L_p, "d": L_d, "f": L_f}

# sanity check
if __name__ == '__main__':
    for l, L in enumerate([L_s, L_p, L_d, L_f]):
        Lx, Ly, Lz = L
        assert np.allclose(Lx, Lx.conj().T)
        assert np.allclose(Lx @ Ly - Ly @ Lx, 1j * Lz)
        assert np.allclose(Ly @ Lz - Lz @ Ly, 1j * Lx)
        assert np.allclose(Lz @ Lx - Lx @ Lz, 1j * Ly)
        assert np.allclose(Lx @ Lx + Ly @ Ly + Lz @ Lz, l * (l + 1) * np.eye(2 * l + 1))
    print("ok")


shells = [["s"],
          ["pz", "px", "py"],
          ["dz2", "dxz", "dyz", "dx2-y2", "dxy"],
          ["fz3", "fxz2", "fyz2", "fzx2-zy2", "fxyz", "fx3-3xy2", "f3yx2-y3"]]
shell_names = ["s", "p", "d", "f"]


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_get_l_shell(l):
    assert np.allclose(get_l_shell(tuple(shells[l])), L_dict[shell_names[l]])



@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_commutation_relations(l):
    shell = shells[l]
    # randomly permute the orbitals
    perm = np.random.permutation(len(shell))
    shell = [shell[i] for i in perm]
    L = get_l_shell(tuple(shell))
    Lx, Ly, Lz = L
    # check commutation relations
    assert np.allclose(Lx, Lx.conj().T)
    assert np.allclose(Lx @ Ly - Ly @ Lx, 1j * Lz)
    assert np.allclose(Ly @ Lz - Lz @ Ly, 1j * Lx)
    assert np.allclose(Lz @ Lx - Lx   @ Lz, 1j * Ly)
    assert np.allclose(Lx @ Lx + Ly @ Ly + Lz @ Lz, l * (l + 1) * np.eye(2 * l + 1))



def test_projection_L():
    from irrep.spacegroup import SpaceGroup
    sg = SpaceGroup.get_trivial(lattice=np.eye(3))
    from wannierberri.symmetry.projections import Projection
    ps = Projection(position_num=[[0, 0, 0]], orbital="s", zaxis=[0, 0, 1], xaxis=[1, 0, 0], spacegroup=sg)
    L = ps.get_orbital_moment_matrix()
    Lp = get_l_shell(tuple(["s"])).transpose(1, 2, 0)
    assert np.allclose(L, Lp)
    assert np.allclose(L, 0)

    p0 = Projection(position_num=[[0, 0, 0]], orbital="p", zaxis=[0, 0, 1], xaxis=[1, 0, 0], spacegroup=sg)
    L0 = p0.get_orbital_moment_matrix()
    L0p = get_l_shell(tuple(["pz", "px", "py"])).transpose(1, 2, 0)
    assert np.allclose(L0, L0p)

    # this is like having non-rotated basis, buit the orbitals are pz, py, -px
    p1 = Projection(position_num=[[0, 0, 0]], orbital="p", zaxis=[0, 0, 1], xaxis=[0, 1, 0], spacegroup=sg)
    L1 = p1.get_orbital_moment_matrix()
    L1p = get_l_shell(tuple(["pz", "py", "px"])).transpose(1, 2, 0)
    L1p[2, :, :] = -L1p[2, :, ::]
    L1p[:, 2, :] = -L1p[:, 2, :]
    assert np.allclose(L1, L1p)
