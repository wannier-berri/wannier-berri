#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                                                            #
#------------------------------------------------------------
"""This utility emulates (at least partly) the behaviour of postw90.x. It sahould start from the same *.win file and produce the same result.

        Usage example: ::

                python3 -m wannierberri.utils.postw90 seedname 


        Options
            seedname
                | seedname for wanier90 files (default : wannier90)
"""

import sys 
import w90io
from .. import run,Grid,calculators,System_w90, Parallel
import numpy as np

# default parameters
parameters = {
    "berry" : False,
    "berry_task" : "",
    "berry_kmesh" : None,
    "fermi_energy":None,
    "fermi_energy_min":None,
    "fermi_energy_max":None,
    "fermi_energy_step":None,
#    "kmesh" : None ,
#    "gyrotropic" : False,
#    "gyrotropic_task" : "",
    }

def main():
    seedname = sys.argv[1]  if len(sys.argv)==2 else "wannier90"
    with open(seedname+".win") as f:
        parsed_win = w90io.parse_win_raw(f.read())
    parsed_param = parsed_win["parameters"]
    for p in parameters:
        try:
            parameters[p] = parsed_param[p]
        except KeyError:
            pass
    if parameters["fermi_energy"] is not None:
        efermi = np.array([parameters["fermi_energy"]])
    else:
        mn,mx,st = [parameters["fermi_energy_"+s] for s in ("min","max","step")]
        efermi = np.linspace( mn,mx,(mx-mn)//st+1)

    calc = {}
    if parameters["berry"] :
        if "ahc" in parameters["berry_task"]:
            calc["ahc"] = calculators.static.AHC(Efermi=efermi,tetra=False)


    system = System_w90(seedname,berry=True)
    grid = Grid(system,NK=parameters["berry_kmesh"])
    parallel = Parallel() # parallel with  "ray",num_cpus - auto

    run(system,
            grid=grid,
            calculators = calc,
            parallel=parallel,
            adpt_num_iter=0,
            fout_name='Fe',
            suffix = "",
            restart=False,
            )



if __name__ == "__main__":
    main()
