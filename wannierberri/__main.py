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
#------------------------------------------------------------
#  This is the main file of the module


import functools 
from .__evaluate import evaluate_K
from .__utility import smoother 
from . import __integrate 
from . import __tabulate  
from . import symmetry

from .__version import __version__
from .__result import NoComponentError
from collections import Iterable
integrate_options=__integrate.calculators.keys()
tabulate_options =__tabulate.calculators.keys()
from .__mmn2uHu import hlp as hlp_mmn
from .__vaspspn import hlp as hlp_spn





import sys,glob


from colorama import init
from termcolor import cprint 
from pyfiglet import figlet_format

def figlet(text,font='cosmike',col='red'):
    init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
    letters=[figlet_format(X, font=font).rstrip("\n").split("\n") for X in text]
#    print (letters)
    logo=[]
    for i in range(len(letters[0])):
        logo.append("".join(L[i] for L in letters))
    cprint("\n".join(logo),col, attrs=['bold'])



def print_options():
    def addparam(param,param_des):
        if len(param)>0:
            return ". Additional parameters: \n"+ "\n".join( (" "*10+"{0:10s} [ default =  {1}] ---  {2}".format(k,param[k],param_des[k]) for k in param) )
        else:
             return ""
    for modname,mod in ("integrate",__integrate),("tabulate",__tabulate):
         cprint ("Options available to {}:".format(modname),'green', attrs=['bold'])
         print("\n".join("{0:10s}  :  {1} {2}".format(key,
                 mod.descriptions[key],addparam(mod.additional_parameters[key],mod.additional_parameters_description[key])
                               ) for key in mod.calculators.keys())+"\n\n")
    hlp_mmn()
    hlp_spn()


#    cprint ("Options available to tabulate:",'green', attrs=['bold'])
#    print("\n".join("{0:10s}  :  {1} ".format(key,__tabulate.descriptions[key]) for key in tabulate_options)+"\n\n")
      



def welcome():
#figlet("WANN IER BERRI",font='cosmic',col='yellow')
    logo="""
.::    .   .::: .:::::::.  :::.    :::.:::.    :::. :::.,::::::  :::::::..       :::::::.  .,::::::  :::::::..   :::::::..   :::
';;,  ;;  ;;;' '  ;;`;;  ` `;;;;,  `;;;`;;;;,  `;;; ;;;;;;;''''  ;;;;``;;;;       ;;;'';;' ;;;;''''  ;;;;``;;;;  ;;;;``;;;;  ;;;
 '[[, [[, [['    ,[[ '[[,    [[[[[. '[[  [[[[[. '[[ [[[ [[cccc    [[[,/[[['       [[[__[[\. [[cccc    [[[,/[[['   [[[,/[[['  [[[
   Y$c$$$c$P    c$$$cc$$$c   $$$ "Y$c$$  $$$ "Y$c$$ $$$ $$\"\"\"\"    $$$$$$c         $$\"\"\"\"Y$$ $$\"\"\"\"    $$$$$$c     $$$$$$c    $$$
    "88"888      888   888,  888    Y88  888    Y88 888 888oo,__  888b "88bo,    _88o,,od8P 888oo,__  888b "88bo, 888b "88bo,888
     "M "M"      YMM   ""`   MMM     YM  MMM     YM MMM \"\"\"\"YUMMM MMMM   "W"     ""YUMMMP"  \"\"\"\"YUMMM MMMM   "W"  MMMM   "W" MMM
"""
    cprint(logo,'yellow')
    cprint("a.k.a. Wannier19",'red')
    figlet("    by Stepan Tsirkin",font='straight',col='green')



    cprint("""
Tutorial at  Electronic Structure Workshop  was recorded.
Video: https://uzh.zoom.us/rec/share/y84qFIzs8WlIY53g-UGYdfUCB6DUaaa80SUZ-fJZy-GyE37OpaVGSfwDqVj43hk
Input files: https://www.dropbox.com/sh/8lt0rznh7zetagp/AABGrVWr6-1b9kMR3Wo8H92Na?dl=0
""", 'yellow', attrs=['bold'])

    cprint("""
User manual under construction may be viewed here: https://www.overleaf.com/read/kbxxtfbnjvxx
""",'magenta' )


    cprint( "\nVersion: {}\n".format( __version__),'cyan', attrs=['bold'])
#    print_options()

#for font in ['twopoint','contessa','tombstone','thin','straight','stampatello','slscript','short','pepper']:
#    __figlet("by Stepan Tsirkin",font=font,col='green')

    

def check_option(quantities,avail,tp):
    for opt in quantities:
      if opt not in avail:
        raise RuntimeError("Quantity {} is not available for {}. Available options are : \n{}\n".format(opt,tp,avail) )


## TODO: Unify the two methids, to do everything in one shot

def integrate(system,grid,Efermi=None,omega=None, Ef0=0,
                        smearEf=10,smearW=10,quantities=[],adpt_num_iter=0,
                        fout_name="wberri",restart=False,numproc=0,fftlib='fftw',suffix="",file_Klist="Klist",parameters={}):
    """
    Integrate 

    Parameters
    ----------
    system : :class:`~wannierberri.System`
        System under investigation
    grid : :class:`~wannierberri.Grid`
        initial grid for integration
    Efermi : numpy.array
        The list of Fermi levels to be scanned (for Fermi-sea or Fermi-surface properties)
    omega : numpy.array
        The list of ferequencies levels to be scanned (for optical properties)
    Ef0 : float
        a single  Fermi level for optical properties
    smearEf : float
        smearing over Fermi levels (in Kelvin)
    smearW : float
        smearing over frequencies (in Kelvin)
    quantities : list of str
        quantities to be integrated. See :ref:`sec-capabilities`
    adpt_num_iter : int 
        number of recursive adaptive refinement iterations. See :ref:`sec-refine`
    num_proc : int 
        number of parallel processes. If <=0  - serial execution without `multiprocessing` module.
   
    Returns
    --------
    dictionary of  :class:`~wannierberri.EnergyResult` 

    Notes
    -----
    Results are also printed to ASCII files

    """
    cprint ("\nIntegrating the following qantities: "+", ".join(quantities)+"\n",'green', attrs=['bold'])
    check_option(quantities,integrate_options,"integrate")
    smoothEf = None if Efermi is None else smoother(Efermi,smearEf) # smoother for functions of Fermi energy
    smoothW= None if omega is None else smoother(omega,smearW) # smoother for functions of frequency
    eval_func=functools.partial( __integrate.intProperty, Efermi=Efermi, omega=omega, smootherEf=smoothEf, smootherOmega=smoothW,
            quantities=quantities, parameters=parameters )
    res=evaluate_K(eval_func,system,grid,nparK=numproc,fftlib=fftlib,
            adpt_num_iter=adpt_num_iter,adpt_nk=1,
                fout_name=fout_name,suffix=suffix,
                restart=restart,file_Klist=file_Klist)
    cprint ("Integrating finished successfully",'green', attrs=['bold'])
    return res



def tabulate(system,grid, quantities=[],
                  fout_name="wberri",ibands=None,suffix="",numproc=0,Ef0=0.,parameters={}):
    """
    Tabulate quantities to be plotted

    Parameters
    ----------
    system : :class:`~wannierberri.System`
        System under investigation
    grid : :class:`~wannierberri.Grid`
        initial grid for integration
    Ef0 : float
        a single  Fermi level. all energies are given with respect to Ef0
    quantities : list of str
        quantities to be integrated. See :ref:`sec-capabilities`
    num_proc : int 
        number of parallel processes. If <=0  - serial execution without `multiprocessing` module.
   
    Returns
    --------
    list of :class:`~wannierberri.__tabulate.TABresult`

    Notes
    -----
    Results are also printed to text files, ready to plot by for `FermiSurfer <https://fermisurfer.osdn.jp/>`_

    """

    assert grid.GammaCentered , "only Gamma-centered grids are allowed for tabulation"
    cprint ("\nTabulating the following qantities: "+", ".join(quantities)+"\n",'green', attrs=['bold'])
    check_option(quantities,tabulate_options,"tabulate")
    eval_func=functools.partial(  __tabulate.tabXnk, ibands=ibands,quantities=quantities,parameters=parameters )

    res=evaluate_K(eval_func,system,grid,nparK=numproc,
            adpt_num_iter=0 , restart=False,suffix=suffix,file_Klist=None)
            
    res=res.to_grid(grid.dense)
        
    open("{0}_E.frmsf".format(fout_name),"w").write(
         res.fermiSurfer(quantity=None,efermi=Ef0) )
    
    for Q in quantities:
#     for comp in ["x","y","z","sq","norm"]:
     for comp in ["x","y","z","xx","yy","zz","xy","yx","xz","zx","yz","zy"]:
        try:
            txt=res.fermiSurfer(quantity=Q,component=comp,efermi=Ef0)
            open("{2}_{1}-{0}.frmsf".format(comp,Q,fout_name),"w").write(txt)
        except NoComponentError:
            pass

    cprint ("Tabulating finished successfully",'green', attrs=['bold'])
    return res



