import numpy as np
from tqdm import tqdm
from ase.units import Hartree
from gpaw import GPAW
from gpaw.spinorbit import soc
import matplotlib.pyplot as plt

theta = np.pi / 2
phi = np.pi / 2

calc = GPAW("band.gpw", txt=None)
calc.initialize_positions()
nspin = calc.get_number_of_spins()

C_ss = np.array(
    [
        [
            np.cos(theta / 2) * np.exp(-1.0j * phi / 2),
            -np.sin(theta / 2) * np.exp(-1.0j * phi / 2),
        ],
        [
            np.sin(theta / 2) * np.exp(1.0j * phi / 2),
            np.cos(theta / 2) * np.exp(1.0j * phi / 2),
        ],
    ]
)

sx_ss = np.array([[0, 1], [1, 0]], complex)
sy_ss = np.array([[0, -1.0j], [1.0j, 0]], complex)
sz_ss = np.array([[1, 0], [0, -1]], complex)
s_vss = [
    C_ss.T.conj() @ sx_ss @ C_ss,
    C_ss.T.conj() @ sy_ss @ C_ss,
    C_ss.T.conj() @ sz_ss @ C_ss,
]


kd = calc.wfs.kd
dVL_avii = {
    a: soc(calc.wfs.setups[a], calc.hamiltonian.xc, D_sp)
    for a, D_sp in calc.density.D_asp.items()
}

m = calc.get_number_of_bands()
nk = len(calc.get_ibz_k_points())
h_soc = np.zeros((nk, 2, 2, m, m), complex)

H_a = {}
for a, dVL_vii in dVL_avii.items():
    ni = dVL_vii.shape[1]
    H_ssii = np.zeros((2, 2, ni, ni), complex)
    H_ssii[0, 0] = dVL_vii[2]
    H_ssii[0, 1] = dVL_vii[0] - 1j * dVL_vii[1]
    H_ssii[1, 0] = dVL_vii[0] + 1j * dVL_vii[1]
    H_ssii[1, 1] = -dVL_vii[2]
    H_a[a] = H_ssii

for q in range(nk):
    for a, H_ssii in H_a.items():
        h_ssii = np.einsum(
            "ab,bcij,cd->adij", C_ss.T.conj(), H_ssii, C_ss, optimize=True
        )
        for s1 in range(2):
            for s2 in range(2):
                h_ii = h_ssii[s1, s2]
                if nspin == 2:
                    P1_mi = calc.wfs.kpt_qs[q][s1].P_ani[a]
                    P2_mi = calc.wfs.kpt_qs[q][s2].P_ani[a]
                if nspin == 1:
                    P1_mi = calc.wfs.kpt_qs[q][0].P_ani[a]
                    P2_mi = calc.wfs.kpt_qs[q][0].P_ani[a]

                h_soc[q, s1, s2] += np.dot(np.dot(P1_mi.conj(), h_ii), P2_mi.T)

alpha = calc.wfs.gd.dv / calc.wfs.gd.N_c.prod()

overlap = np.zeros((nk, 2, 2, m, m), complex)

for q in range(nk):
    for s1 in range(2):
        psi1 = calc.wfs.kpt_qs[q][s1].psit_nG[:]
        for s2 in range(2):
            psi2 = calc.wfs.kpt_qs[q][s2].psit_nG[:]
            overlap[q, s1, s2] += alpha * psi1.conj() @ psi2.T
            for a, setup in enumerate(calc.wfs.setups):
                P1_mi = calc.wfs.kpt_qs[q][s1].P_ani[a]
                P2_mi = calc.wfs.kpt_qs[q][s2].P_ani[a]
                overlap_ii = setup.dO_ii
                overlap[q, s1, s2] += P1_mi.conj() @ overlap_ii @ P2_mi.T


# Save the SOC Hamiltonian h(k,s,s',n,n')
np.savez(
    "h_soc.npz", h_soc=h_soc, sx=s_vss[0], sy=s_vss[1], sz=s_vss[2], overlap=overlap
)

plot = True
if plot:
    h_band = np.zeros((nk, 2, 2, m, m), complex)
    for kpt in calc.wfs.kpt_u:
        h_band[kpt.q, kpt.s, kpt.s] = np.diag(kpt.eps_n)
    h = h_band + h_soc
    h = h.transpose(0, 1, 3, 2, 4).reshape(nk, 2 * m, 2 * m)

    e_km, v_knm = np.linalg.eigh(h)
    v_ksnm = v_knm.reshape(nk, 2, m, 2 * m)
    e_km *= Hartree

    sx_km = np.einsum(
        "kani,ab,kabnm,kbmj->kij",
        v_ksnm.conj(),
        s_vss[0],
        overlap,
        v_ksnm,
        optimize=True,
    )
    sy_km = np.einsum(
        "kani,ab,kabnm,kbmj->kij",
        v_ksnm.conj(),
        s_vss[1],
        overlap,
        v_ksnm,
        optimize=True,
    )
    sz_km = np.einsum(
        "kani,ab,kabnm,kbmj->kij",
        v_ksnm.conj(),
        s_vss[2],
        overlap,
        v_ksnm,
        optimize=True,
    )

    path = calc.atoms.cell.bandpath("MGM", npoints=nk)
    x_path, x_ticks, x_labels = path.get_linear_kpoint_axis()
    x_labels[0] = "+M"
    x_labels[-1] = "-M"
    ef = calc.get_fermi_level()

    plt.figure(figsize=(5, 4))
    for n in range(2 * m):
        im = plt.scatter(
            x_path,
            e_km[:, n] - ef,
            c=sy_km[:, n, n].real,
            s=1,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
    cbar = plt.colorbar(im, label=r"$\langle \hat{S}_y \rangle$", aspect=30)
    cbar.ax.minorticks_on()
    plt.ylim(-1, -0.2)
    plt.xlim(x_path[0], x_path[-1])
    plt.xticks(x_ticks, x_labels)
    for x_ in x_ticks:
        plt.axvline(x_, c="k", lw=0.5, ls="--", zorder=-np.inf)
    plt.ylabel("Energy (eV)")
    plt.axhline(0, c="k", lw=0.5, ls="--", zorder=-np.inf)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig("bands_soc.png", dpi=300)
