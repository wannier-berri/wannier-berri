
            # ###   ###   #####  ###
            # #  #  #  #  #      #  #
            # ###   ###   ###    ###
            # #  #  #  #  #      #
            # #   # #   # #####  #


##################################################################
## This file is distributed as part of                           #
## "IrRep" code and under terms of GNU General Public license v3 #
## see LICENSE file in the                                       #
##                                                               #
##  Written by Stepan Tsirkin, University of Zurich.             #
##  e-mail: stepan.tsirkin@physik.uzh.ch                         #
##################################################################


import numpy as np

def str2list(string):
    return np.hstack( [ np.arange( *(np.array(s.split("-"),dtype=int)+np.array([0,1]) )) if "-" in s else np.array([int(s)])  for s in string.split(",")] )


def compstr(string):
    if "i" in string:
        if "+" in string:
            return float(string.split("+")[0])+1j*float(string.split("+")[1].strip("i"))
        elif "-" in string:
            return float(string.split("-")[0])+1j*float(string.split("-")[1].strip("i"))
    else:
        return  float(string)



def str2list_space(string):
#    print ("str2list  <{0}> ".format(string))
    res=np.hstack( [ np.arange( *(np.array(s.split("-"),dtype=int)+np.array([0,1]) )) if "-" in s else np.array([int(s)])  for s in string.split()] )
#    print ("str2list  <{0}> -> <{1}>".format(string,res))
    return res 


def str2bool(v):
    if v[0] in "fF" :
        return False
    elif v[0] in "tT":
        return True
    else :
        raise RuntimeError(" unrecognized value of bool parameter :{0}".format(v) )


def str_(x):
   return str(round(x,5))
