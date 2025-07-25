#!/usr/bin/env python2
# =========================== Wave2spn =============================
#                                                                 #
#    An utility to convert WAVECAR to .spn file for WANNIER90     #
#                                                                 #
# =========================== Wave2spn =============================
#
# Written by Stepan Tsirkin (University of the Basque Country)
#    now at : Iniversity of Zurich
import warnings
import numpy as np
from ..io import FortranFileW
from ..utility import cached_einsum, time_now_iso

PAW_warning = """vaspspn uses pseudo-wavefunction instead of the full PAW
(`Blöchl 1994 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.50.17953>`__) wavefunction
This approximation usually gives reasonable results for interpolation of spin in non-spin-degenerate materials,
but is known to give wrong results for spin hall conductivity in spin-degenerate (preserving P*T symmetry)
bandstructures (see `Issue #274 <https://github.com/wannier-berri/wannier-berri/issues/274>`__)

A better alternative is provided by `Chengcheng Xiao <https://github.com/Chengcheng-Xiao/VASP2WAN90_v2_fix>`__

New versions of VASP compute the spn file directly, so this utility is mostly obsolete. see `LWRITE_SPN <https://www.vasp.at/wiki/index.php/LWRITE_SPN>`__ 
"""

__doc__ = r"""
An utility to calculate the ``.spn`` file for wannier90 from ``WAVECAR`` file generated by
`VASP <https://www.vasp.at/>`_.

Computes :math:`s_{mn}({\bf q})=\langle u_{m{\bf q}}\vert\hat{\sigma}\vert u_{n{\bf q}}\rangle` based on the normalized
pseudo-wavefunction read from the ``WAVECAR`` file.

WARNING : """ + PAW_warning + """

usage : ::

    python3 -m wannierberri.utils.vaspspn   option=value

Options
    -h
        |  print this help message

    fin
        |  inputfile  name.
        |  default: WAVECAR
    fout
        |  outputfile name
        |  default: wannier90.spn
    IBstart
        |  the first band to be considered (counting starts from 1).
        |  default: 1
    NB
        |  number of bands in the output. If NB<=0 all bands are used.
        |  default: 0
    norm
        |  how to normalize the eigenstates, if they are not perfectly orthonormal
        |  norm=norm  (D) -   normalize each state individually
        |  norm=none      -   do not normalize WFs, take as they are.
"""


def hlp():
    from termcolor import cprint
    cprint("vaspspn  (utility)", 'green', attrs=['bold'])
    print(__doc__)


def main(argv):
    warnings.warn(PAW_warning)

    fin = "WAVECAR"
    fout = "wannier90.spn"
    NBout = 0
    IBstart = 1
    normalize = "norm"
    for arg in argv:
        if arg == "-h":
            hlp()
            exit()
        else:
            k, v = arg.split("=")
            if k == "fin":
                fin = v
            elif k == "fout":
                fout = v
            elif k == "NB":
                NBout = int(v)
            elif k == "IBstart":
                IBstart = int(v)
            elif k == "norm":
                normalize = v

    print(f"reading {fin}\n writing to {fout}")

    WAV = open(fin, "rb")
    RECL = 3

    def record(irec, cnt=np.inf, dtype=float):
        WAV.seek(irec * RECL)
        return np.fromfile(WAV, dtype=dtype, count=min(RECL, cnt))

    RECL, ispin, iprec = [int(x) for x in record(0)]

    print(RECL, ispin, iprec)

    if iprec != 45200:
        raise RuntimeError('double precision WAVECAR is not supported')
    if ispin != 1:
        raise RuntimeError(f'WAVECAR does not contain spinor wavefunctions. ISPIN={ispin}')

    NK, NBin = [int(x) for x in record(1, 2)]

    IBstart -= 1
    if IBstart < 0:
        IBstart = 0
    if NBout <= 0:
        NBout = NBin
    if NBout + IBstart > NBin:
        warnings.warn(f"NB+IBstart-1={NBout + IBstart} exceeds the number of bands in WAVECAR NBin={NBin}"
                      f"We set NBout={NBin - IBstart}")
        NBout = NBin - IBstart

    print(f"WAVECAR contains {NK} k-points and {NBin} bands.\n Writing {NBout} bands in the output starting from")

    SPN = FortranFileW(fout)
    header = f"Created from WAVECAR at {time_now_iso()}"
    header = header[:60]
    header += " " * (60 - len(header))
    SPN.write_record(bytearray(header, encoding='ascii'))
    SPN.write_record(np.array([NBout, NK], dtype=np.int32))

    for ik in range(NK):
        npw = int(record(2 + ik * (NBin + 1), 1))
        npw12 = npw // 2
        if npw != npw12 * 2:
            raise RuntimeError(f"odd number of coefs {npw}")
        print(f"k-point {ik:3d} : {npw:6d} plane waves")
        WF = np.zeros((npw, NBout), dtype=complex)
        for ib in range(NBout):
            WF[:, ib] = record(3 + ik * (NBin + 1) + ib + IBstart, npw, np.complex64)
       
        if normalize == "norm":
            WF /= np.linalg.norm(WF, axis=0)[None, :]

        SIGMA = np.array(
            [
                [
                    cached_einsum("ki,kj->ij",
                              WF.conj()[npw12 * i:npw12 * (i + 1), :], WF[npw12 * j:npw12 * (j + 1), :])
                    for j in (0, 1)
                ] for i in (0, 1)
            ])
        SX = SIGMA[0, 1] + SIGMA[1, 0]
        SY = -1.j * (SIGMA[0, 1] - SIGMA[1, 0])
        SZ = SIGMA[0, 0] - SIGMA[1, 1]
        A = np.array([s[n, m] for m in range(NBout) for n in range(m + 1) for s in (SX, SY, SZ)], dtype=np.complex128)
        SPN.write_record(A)


if __name__ == "__main__":
    from sys import argv
    main(argv[1:])
