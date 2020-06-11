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
from .__evaluate import evaluate_K, determineNK
from .__utility import smoother 
from . import __integrate 
from . import __tabulate  
from . import __symmetry as symmetry

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
.::    .   .:::  :::.     :::.    :::.:::.    :::. :::.,::::::  :::::::..       :::::::.  .,::::::  :::::::..   :::::::..   :::
';;,  ;;  ;;;'   ;;`;;    `;;;;,  `;;;`;;;;,  `;;; ;;;;;;;''''  ;;;;``;;;;       ;;;'';;' ;;;;''''  ;;;;``;;;;  ;;;;``;;;;  ;;;
 '[[, [[, [['   ,[[ '[[,    [[[[[. '[[  [[[[[. '[[ [[[ [[cccc    [[[,/[[['       [[[__[[\. [[cccc    [[[,/[[['   [[[,/[[['  [[[
   Y$c$$$c$P   c$$$cc$$$c   $$$ "Y$c$$  $$$ "Y$c$$ $$$ $$\"\"\"\"    $$$$$$c         $$\"\"\"\"Y$$ $$\"\"\"\"    $$$$$$c     $$$$$$c    $$$
    "88"888     888   888,  888    Y88  888    Y88 888 888oo,__  888b "88bo,    _88o,,od8P 888oo,__  888b "88bo, 888b "88bo,888
     "M "M"     YMM   ""`   MMM     YM  MMM     YM MMM \"\"\"\"YUMMM MMMM   "W"     ""YUMMMP"  \"\"\"\"YUMMM MMMM   "W"  MMMM   "W" MMM
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


def integrate(system,NK=None,NKdiv=None,NKFFT=None,Efermi=None,omega=None, Ef0=0,
                        smearEf=10,smearW=10,quantities=[],adpt_num_iter=0,
                        fout_name="wberri",symmetry_gen=[],
                GammaCentered=True,restart=False,numproc=0,suffix="",file_Klist="Klist",parameters={}):

    cprint ("\nIntegrating the following qantities: "+", ".join(quantities)+"\n",'green', attrs=['bold'])
    check_option(quantities,integrate_options,"integrate")
    smooth=smoother(Efermi,smearEf)
    eval_func=functools.partial(  __integrate.intProperty, Efermi=Efermi, smootherEf=smooth,quantities=quantities,parameters=parameters )
    res=evaluate_K(eval_func,system,NK=NK,NKdiv=NKdiv,NKFFT=NKFFT,nproc=numproc,
            adpt_num_iter=adpt_num_iter,adpt_nk=1,
                fout_name=fout_name,symmetry_gen=symmetry_gen,suffix=suffix,
                GammaCentered=GammaCentered,restart=restart,file_Klist=file_Klist)
    cprint ("Integrating finished successfully",'green', attrs=['bold'])
    return res



def tabulate(system,NK=None,NKdiv=None,NKFFT=None,omega=None, quantities=[],symmetry_gen=[],
                  fout_name="wberri",ibands=None,suffix="",numproc=0,Ef0=0.,parameters={}):

    cprint ("\nTabulating the following qantities: "+", ".join(quantities)+"\n",'green', attrs=['bold'])
    NKdiv,NKFFT=determineNK(NKdiv,NKFFT,NK,system.NKFFTmin)
    NK=NKdiv*NKFFT
    check_option(quantities,tabulate_options,"tabulate")
    eval_func=functools.partial(  __tabulate.tabXnk, ibands=ibands,quantities=quantities,parameters=parameters )

    res=evaluate_K(eval_func,system,NK=NK,NKdiv=NKdiv,NKFFT=NKFFT,nproc=numproc,
            adpt_num_iter=0 ,symmetry_gen=symmetry_gen,  GammaCentered=True ,restart=False,suffix=suffix,file_Klist=None)
            
    res=res.to_grid(NKFFT*NKdiv)
        
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



