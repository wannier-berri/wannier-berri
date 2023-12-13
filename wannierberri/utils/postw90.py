#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                                                            #
# ------------------------------------------------------------
"""This utility emulates (at least partly) the behaviour of postw90.x. It sahould start from the same *.win file and produce the same result.

    optionally, parameters can be given in the command line (theose will override the parameters in the ffile)
    additional options starting with "__wb" can be provided"


    **Note:** to use this module the user needs to manually install the wannier90io module from github:
    `pip install git+https://github.com/jimustafa/wannier90io-python.git`

        Usage example: ::

                python3 -m wannierberri.utils.postw90 seedname [parameters] [ __wb_fft_lib=<value> ]

        Options
            seedname
                | seedname for wanier90 files (default : wannier90)
            parameters
                | any parameters that can be accepted by postw90.x (e.g. berry=True), but no spaces are allowed in the parameters
                | command-line parameters override those found in the seedname.win file
            __wb_fft_lib
                | fftw3 (default) or numpy
"""

from .. import run, Grid, calculators, System_w90, Parallel
import numpy as np
from ..system.w90_files import parse_win_raw

# default parameters
parameters = {
    "berry": False,
    "berry_task": "",
    "berry_kmesh": None,
    "fermi_energy": None,
    "fermi_energy_min": None,
    "fermi_energy_max": None,
    "fermi_energy_step": None,
    "use_ws_distance": True,
    "transl_inv": True,
    "__wb_fft_lib": "fftw",
}


def main(argv):
    seedname = argv[0]
    parsed_win = parse_win_raw(seedname + ".win")
    parsed_param = parsed_win["parameters"]
    if len(argv) > 1:
        parsed_command_line = parse_win_raw(text="\n".join(argv[1:]))
        parsed_param.update(parsed_command_line["parameters"])

    for p in parameters:
        try:
            parameters[p] = parsed_param[p]
        except KeyError:
            pass
    if parameters["fermi_energy"] is not None:
        efermi = np.array([parameters["fermi_energy"]])
    else:
        mn, mx, st = [parameters["fermi_energy_" + s] for s in ("min", "max", "step")]
        efermi = np.linspace(mn, mx, int(abs((mx - mn) / st)) + 1)

    calc = {}
    if parameters["berry"]:
        if "ahc" in parameters["berry_task"]:
            calc["ahc"] = calculators.static.AHC(Efermi=efermi, tetra=False)

    system = System_w90(seedname, berry=parameters["berry"],
                        use_ws=parameters["use_ws_distance"],
                        fft=parameters["__wb_fft_lib"],
                        transl_inv=parameters["transl_inv"]
                        )
    grid = Grid(system, NK=parameters["berry_kmesh"])
    parallel = Parallel()  # parallel with  "ray",num_cpus - auto

    run(system,
        grid=grid,
        calculators=calc,
        parallel=parallel,
        adpt_num_iter=0,
        fout_name='Fe',
        suffix="",
        parameters_K={"fftlib": parameters["__wb_fft_lib"]},
        restart=False,
        )

    parallel.shutdown()


if __name__ == "__main__":
    from sys import argv

    main(argv[1:])
