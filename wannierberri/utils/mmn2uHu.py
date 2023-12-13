#                                                            #
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
# ------------------------------------------------------------
"""This utility calculates the matrices .uHu, .uIu, .sHu, and/or .sIu from the .mmn, .spn matrices, and also reduces the number of bands in .amn, .mmn, .eig  and .spn files

        Usage example: ::

                python3 -m wannierberri.utils.mmn2uHu seedname NBout=10 NBsum=100,200  targets=mmn,uHu formatted=uHu


        Options
            -h
                | print the help message

            IBstart
                |  the first band in the output file (counting starts from 1).
                |  default: 1
            IBstartSum
                |  the first band in the sum         (counting starts from 1).
                |  default: 1
            NBout
                |  the number of bands in the output files.
                |  Default : all bands
            NBsum
                |  the number of bands in the summation. (one may specify several numbers, usefull to test convergence with the number of bands).
                |  Default:all bands
            input
                |  path to the input files.
                |  Default: ./
            output
                |  path to the output files

            targets
                |  files to write : ``amn``, ``mmn``, ``spn``, ``uHu``, ``uIu``,  ``sHu``, ``sIu``, ``eig``
                |  default: ``amn``,``mmn``,``eig``,``uHu``
            formatted
                |  files to write as formatted  ``uHu``, ``uIu`` , ``sHu``, ``sIu``, ``spn``, ``spn_in``, ``spn_out``, ``all``
                |  default: none


"""

import numpy as np
import os
import datetime
from ..__utility import FortranFileR, FortranFileW


def hlp():
    from termcolor import cprint
    cprint("mmn2uHu  (utility)", 'green', attrs=['bold'])
    print(__doc__)


def run_mmn2uHu(PREFIX, **kwargs):
    # Parse input arguments
    writeAMN = kwargs.get("writeAMN", True)
    writeMMN = kwargs.get("writeMMN", True)
    writeUHU = kwargs.get("writeUHU", True)
    writeEIG = kwargs.get("writeEIG", True)
    writeUIU = kwargs.get("writeUIU", False)
    writeSPN = kwargs.get("writeSPN", False)
    writeSHU = kwargs.get("writeSHU", False)
    writeSIU = kwargs.get("writeSIU", False)

    uHu_formatted = kwargs.get("uHu_formatted", False)
    uIu_formatted = kwargs.get("uIu_formatted", False)
    sHu_formatted = kwargs.get("sHu_formatted", False)
    sIu_formatted = kwargs.get("sIu_formatted", False)
    spn_formatted_out = kwargs.get("spn_formatted_out", False)
    spn_formatted_in = kwargs.get("spn_formatted_in", False)

    NB_out_list = kwargs.get("NB_out_list", [None])
    NB_sum_list = kwargs.get("NB_sum_list", [None])
    INPUTDIR = kwargs.get("INPUTDIR", "./")
    OUTDIR = kwargs.get("OUTDIR", "reduced")

    IBstart = kwargs.get("IBstart", 0)
    IBstartSum = kwargs.get("IBstartSum", 0)

    # Begin calculation

    AMNrd = False
    MMNrd = False
    EIGrd = False

    if not (writeEIG or writeUHU or writeSHU):
        EIGrd = True

    print("----------\n MMN  read\n---------\n")

    if not MMNrd:
        f_mmn_in = open(os.path.join(INPUTDIR, PREFIX + ".mmn"), "r")
        MMNhead = f_mmn_in.readline().strip()
        s = f_mmn_in.readline()
        NB_in, NK, NNB = np.array(s.split(), dtype=int)
        MMN = []
        MMNheadstrings = []
        if writeMMN or writeUHU or writeUIU or writeSHU or writeSIU:
            for ik in range(NK):
                print("k-point {} of {}".format(ik + 1, NK))
                MMN.append([])
                MMNheadstrings.append([])
                for ib in range(NNB):
                    s = f_mmn_in.readline()
                    MMNheadstrings[ik].append(s)
                    # ik1, ik2 = (int(i) - 1 for i in s.split()[:2])
                    tmp = np.array(
                        [[f_mmn_in.readline().split() for n in range(NB_in)] for m in range(NB_in)], dtype=float)
                    tmp = (tmp[:, :, 0] + 1j * tmp[:, :, 1])
                    MMN[ik].append(tmp)
        MMNrd = True

    print("----------\n MMN  read - OK\n---------\n")

    if NB_out_list == [None]:
        NB_out_list = [NB_in]

    for NB_out in NB_out_list:
        RESDIR = "{0}_NB={1}".format(OUTDIR, NB_out)
        try:
            os.mkdir(RESDIR)
        except Exception as ex:
            print(ex)

    if writeMMN:
        f_mmn_out = open(os.path.join(RESDIR, PREFIX + ".mmn"), "w")
        f_mmn_out.write("{0}, reduced to {2} bands {1} \n".format(MMNhead, datetime.datetime.now().isoformat(), NB_out))
        f_mmn_out.write("  {0:10d}  {1:10d}  {2:10d}\n".format(NB_out, NK, NNB))
        for ik in range(NK):
            print("k-point {} of {}".format(ik, NK))
            for ib in range(NNB):
                f_mmn_out.write(MMNheadstrings[ik][ib])
                for m in range(NB_out):
                    for n in range(NB_out):
                        f_mmn_out.write(
                            "  {0:16.12f}  {1:16.12f}\n".format(
                                MMN[ik][ib][m + IBstart, n + IBstart].real, MMN[ik][ib][m + IBstart, n + IBstart].imag))
        f_mmn_out.close()
    print("----------\n MMN OK  \n---------\n")

    if not EIGrd:
        EIG = np.loadtxt(os.path.join(INPUTDIR, PREFIX + ".eig"), usecols=(2, )).reshape((NK, NB_in), order='C')
        EIGrd = True

    if writeEIG:
        feig_out = open(os.path.join(RESDIR, PREFIX + ".eig"), "w")
        for ik in range(NK):
            for ib in range(NB_out):
                feig_out.write(" {0:4d} {1:4d} {2:17.12f}\n".format(ib + 1, ik + 1, EIG[ik, ib + IBstart]))
        feig_out.close()

    print("----------\n AMN   \n---------\n")

    if writeAMN:
        if not AMNrd:
            f_amn_in = open(os.path.join(INPUTDIR, PREFIX + ".amn"), "r")
            head_AMN = f_amn_in.readline().strip()
            s = f_amn_in.readline()
            nb, nk, npr = np.array(s.split(), dtype=int)
            assert nb == NB_in
            assert nk == NK
            AMN = np.loadtxt(f_amn_in, dtype=float)[:, 3:5]
            print("AMN size=", AMN.shape)
            print(nb, npr, nk)
            AMN = np.reshape(AMN[:, 0] + AMN[:, 1] * 1j, (nb, npr, nk), order='F')
            AMNrd = True
            f_amn_in.close()

        f_amn_out = open(os.path.join(RESDIR, PREFIX + ".amn"), "w")
        f_amn_out.write(
            "{0}, reduced to {2} bands {1} \n".format(head_AMN,
                                                      datetime.datetime.now().isoformat(), NB_out))
        f_amn_out.write("  {0:10d}  {1:10d}  {2:10d}\n".format(NB_out, NK, npr))
        for ik in range(nk):
            amn = AMN[IBstart:IBstart + NB_out, :, ik]
            for ipr in range(npr):
                f_amn_out.write(
                    "".join(
                        " {0:4d} {1:4d} {2:4d}  {3:16.12f}  {4:16.12f}\n".format(
                            ib + 1, ipr + 1, ik + 1, amn[ib, ipr].real, amn[ib, ipr].imag) for ib in range(NB_out)))
        f_amn_out.close()
    print("----------\n AMN  - OK \n---------\n")

    UXUlist = []
    if writeUHU:
        UXUlist.append(("uHu", uHu_formatted))
    if writeUIU:
        UXUlist.append(("uIu", uIu_formatted))
    print(UXUlist)
    if len(UXUlist) > 0:
        NB_sum_max = NB_in - IBstartSum
        for NB_sum in NB_sum_list:
            if NB_sum is None or NB_sum > NB_sum_max:
                NB_sum = NB_sum_max
            for UXU in UXUlist:
                print("----------\n  {1}  NBsum={0} \n---------".format(NB_sum, UXU[0]))
                formatted = UXU[1]

                header = "{3} from mmn red to {1} sum {2} bnd {0} ".format(
                    datetime.datetime.now().isoformat(), NB_out, NB_sum, UXU[0])
                header = header[:60]
                header += " " * (60 - len(header))
                print(header)
                print(len(header))
                if formatted:
                    f_uXu_out = open(os.path.join(RESDIR, PREFIX + "_nbs={0:d}.{1}".format(NB_sum, UXU[0])), 'w')
                    f_uXu_out.write("".join(header) + "\n")
                    f_uXu_out.write("{0}   {1}   {2} \n".format(NB_out, NK, NNB))
                else:
                    f_uXu_out = FortranFileW(os.path.join(RESDIR, PREFIX + "_nbs={0:d}.{1}".format(NB_sum, UXU[0])))
                    f_uXu_out.write_record(bytearray(header, encoding='ascii'))
                    f_uXu_out.write_record(np.array([NB_out, NK, NNB], dtype=np.int32))

                for ik in range(NK):
                    print("k-point {} of {}".format(ik + 1, NK))
                    if UXU[0] == "uHu":
                        eig_dum = EIG[ik][IBstartSum:IBstartSum + NB_sum]
                    elif UXU[0] == "uIu":
                        eig_dum = np.ones(NB_sum)
                    else:
                        raise RuntimeError()
                    A = np.zeros((NNB, NNB, NB_out, NB_out), dtype=complex)
                    for ib2 in range(NNB):
                        for ib1 in range(ib2 + 1):
                            A[ib2, ib1] = np.einsum(
                                'ml,nl,l->mn', MMN[ik][ib1][IBstart:IBstart + NB_out,
                                                            IBstartSum:NB_sum + IBstartSum].conj(),
                                MMN[ik][ib2][IBstart:NB_out + IBstart, IBstartSum:NB_sum + IBstartSum], eig_dum)
                            if ib1 == ib2:
                                A[ib2, ib1] = 0.5 * (A[ib2, ib1] + A[ib2, ib1].T.conj())
                            else:
                                A[ib1, ib2] = A[ib2, ib1].T.conj()
                    if (formatted):
                        f_uXu_out.write(
                            "".join(
                                "{0:20.10e}   {1:20.10e}\n".format(a.real, a.imag) for a in A.reshape(-1, order='C')))
                    else:
                        for ib2 in range(NNB):
                            for ib1 in range(NNB):
                                f_uXu_out.write_record(A[ib2][ib1].reshape(-1, order='C'))
                print("----------\n {0} OK  \n---------\n".format(UXU[0]))
                f_uXu_out.close()

    if writeSPN or writeSHU or writeSIU:

        print("----------\n SPN  \n---------\n")

        if spn_formatted_in:
            f_spn_in = open(os.path.join(INPUTDIR, PREFIX + ".spn"), 'r')
            SPNheader = f_spn_in.readline().strip()
            nbnd, NK = (int(x) for x in f_spn_in.readline().split())
        else:
            f_spn_in = FortranFileR(os.path.join(INPUTDIR, PREFIX + ".spn"))
            SPNheader = (f_spn_in.read_record(dtype='c'))
            nbnd, NK = f_spn_in.read_record(dtype=np.int32)
            SPNheader = "".join(a.decode('ascii') for a in SPNheader)

        print(SPNheader)

        assert (nbnd == NB_in)

        indm, indn = np.tril_indices(NB_in)
        indmQP, indnQP = np.tril_indices(NB_out)

        if spn_formatted_out:
            f_spn_out = open(os.path.join(RESDIR, PREFIX + ".spn"), 'w')
            f_spn_out.write(SPNheader + "\n")
            f_spn_out.write("{0}  {1}\n".format(NB_out, NK))
        else:
            f_spn_out = FortranFileW(os.path.join(RESDIR, PREFIX + ".spn"))
            f_spn_out.write_record(SPNheader.encode('ascii'))
            f_spn_out.write_record(np.array([NB_out, NK], dtype=np.int32))

        SPN = []
        for ik in range(NK):
            A = np.zeros((3, nbnd, nbnd), dtype=complex)
            SPN.append([])
            if spn_formatted_in:
                tmp = np.array([f_spn_in.readline().split() for i in range(3 * nbnd * (nbnd + 1) // 2)], dtype=float)
                tmp = tmp[:, 0] + 1.j * tmp[:, 1]
            else:
                tmp = f_spn_in.read_record(dtype=np.complex128)
            A[:, indn, indm] = tmp.reshape(3, nbnd * (nbnd + 1) // 2, order='F')
            check = np.einsum('ijj->', np.abs(A.imag))
            A[:, indm, indn] = A[:, indn, indm].conj()
            if check > 1e-10:
                raise RuntimeError("REAL DIAG CHECK FAILED : {0}".format(check))
            if writeSHU or writeSIU:
                SPN[ik] = A
            if writeSPN:
                A = A[:, indnQP + IBstart, indmQP + IBstart].reshape(-1, order='F')
                if spn_formatted_out:
                    f_spn_out.write("".join("{0:26.16e}  {1:26.16e}\n".format(x.real, x.imag) for x in A))
                else:
                    f_spn_out.write_record(A)

        print("----------\n SPN OK  \n---------\n")

    SXUlist = []
    if writeSHU:
        SXUlist.append(("sHu", sHu_formatted))
    if writeSIU:
        SXUlist.append(("sIu", sIu_formatted))
    print(SXUlist)
    if len(SXUlist) > 0:
        for NB_sum in NB_sum_list:
            if NB_sum is None:
                NB_sum = NB_in
            for SXU in SXUlist:
                print("----------\n  {1}  NBsum={0} \n---------".format(NB_sum, SXU[0]))
                formatted = SXU[1]

                header = "{3} from mmn red to {1} sum {2} bnd {0} ".format(
                    datetime.datetime.now().isoformat(), NB_out, NB_sum, SXU[0])
                header = header[:60]
                header += " " * (60 - len(header))
                print(header)
                print(len(header))
                if formatted:
                    f_sXu_out = open(os.path.join(RESDIR, PREFIX + "_nbs={0:d}.{1}".format(NB_sum, SXU[0])), 'w')
                    f_sXu_out.write("".join(header) + "\n")
                    f_sXu_out.write("{0}   {1}   {2} \n".format(NB_out, NK, NNB))
                else:
                    f_sXu_out = FortranFileW(os.path.join(RESDIR, PREFIX + "_nbs={0:d}.{1}".format(NB_sum, SXU[0])))
                    f_sXu_out.write_record(bytearray(header, encoding='ascii'))
                    f_sXu_out.write_record(np.array([NB_out, NK, NNB], dtype=np.int32))

                for ik in range(NK):
                    print("k-point {} of {}".format(ik + 1, NK))
                    if SPN[ik].shape[1] > NB_in:
                        SPN[ik] = np.resize(SPN[ik], (3, NB_in))
                    if SXU[0] == "sHu":
                        eig_dum = EIG[ik][IBstartSum:IBstartSum + NB_sum]
                    elif SXU[0] == "sIu":
                        eig_dum = np.ones(NB_sum)
                        if NB_sum > NB_in:
                            eig_dum = np.ones(NB_in)
                    else:
                        raise RuntimeError()
                    A = np.zeros((NNB, 3, NB_out, NB_out), dtype=complex)
                    for ib2 in range(NNB):
                        for ipol in range(3):
                            A[ib2, ipol, :, :] = np.einsum(
                                'nl,ml,l->mn', MMN[ik][ib2][IBstart:IBstart + NB_out, IBstartSum:NB_sum + IBstartSum],
                                SPN[ik][ipol][IBstart:IBstart + NB_out, IBstartSum:IBstartSum + NB_sum], eig_dum)
                    if (formatted):
                        f_sXu_out.write(
                            "".join(
                                "{0:20.10e}   {1:20.10e}\n".format(a.real, a.imag) for a in A.reshape(-1, order='C')))
                    else:
                        for ib2 in range(NNB):
                            for ipol in range(3):
                                f_sXu_out.write_record(A[ib2, ipol, :, :].reshape(-1, order='C'))
                print("----------\n {0} OK  \n---------\n".format(SXU[0]))
                f_sXu_out.close()

    return NB_out_list


def main(argv):
    hlp()

    if len(argv) == 0 or argv[0] == "-h":
        return

    PREFIX = argv[0]

    kwargs = {}

    for arg in argv[1:]:
        arg = arg.split("=")
        if arg[0] == "NBout":
            kwargs["NB_out_list"] = [int(s) for s in arg[1].split(',')]
        if arg[0] == "NBsum":
            kwargs["NB_sum_list"] = [int(s) for s in arg[1].split(',')]
        if arg[0] == "IBstart":
            kwargs["IBstart"] = int(arg[1]) - 1
        if arg[0] == "IBstartSum":
            kwargs["IBstartSum"] = int(arg[1]) - 1
        if arg[0] == "input":
            kwargs["INPUTDIR"] = arg[1]
        if arg[0] == "output":
            kwargs["OUTDIR"] = arg[1]
        if arg[0] == "targets":
            tarlist = arg[1].split(",")
            kwargs["writeAMN"] = "amn" in tarlist
            kwargs["writeEIG"] = "eig" in tarlist
            kwargs["writeMMN"] = "mmn" in tarlist
            kwargs["writeUHU"] = "uHu" in tarlist
            kwargs["writeUIU"] = "uIu" in tarlist
            kwargs["writeSPN"] = "spn" in tarlist
            kwargs["writeSHU"] = "sHu" in tarlist
            kwargs["writeSIU"] = "sIu" in tarlist
        if arg[0] == "formatted":
            tarlist = arg[1].split(",")
            kwargs["uHu_formatted"] = any(x in tarlist for x in ("uHu", "all"))
            kwargs["uIu_formatted"] = any(x in tarlist for x in ("uIu", "all"))
            kwargs["sHu_formatted"] = any(x in tarlist for x in ("sHu", "all"))
            kwargs["sIu_formatted"] = any(x in tarlist for x in ("sIu", "all"))
            kwargs["spn_formatted_out"] = any(x in tarlist for x in ("spn", "spn_out", "all"))
            kwargs["spn_formatted_in"] = any(x in tarlist for x in ("spn", "spn_in", "all"))

    return run_mmn2uHu(PREFIX, **kwargs)


if __name__ == "__main__":
    from sys import argv
    main(argv[1:])
