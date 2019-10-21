__debug = False

import inspect
import numpy as np

def print_my_name_start():
    if __debug: 
        print("DEBUG: Running {} ..".format(inspect.stack()[1][3]))


def print_my_name_end():
    if __debug: 
        print("DEBUG: Running {} - done ".format(inspect.stack()[1][3]))




def einsumk(*args):
#   self._HHUU_K=np.einsum("kmi,kmn,knj->kij",self.UUC_K,_HH_K,self.UU_K).real
    left,right=args[0].split("->")
    left=left.split(",")
    for s in left + [right]:
        if s[0]!='k':
            raise RuntimeError("the first index should be 'k', found '{1}".format(s[0]))
    string_new=",".join(s[1:]for s in left)+"->"+right[1:]
    print ("string_new"+"  ".join(str(a.shape) for a in args[1:]))
    nmat=len(args)-1
    assert(len(left)==nmat)
    if nmat==2:
        return np.array([np.einsum(string_new,a,b) for a,b in zip(args[1],args[2])])
    elif nmat==3:
#        return np.array([np.einsum(string_new,a,b,c) for a,b,c in zip(args[1],args[2],args[3])])
        return np.array([a.dot(b).dot(c) for a,b,c in zip(args[1],args[2],args[3])])
    elif nmat==4:
        return np.array([np.einsum(string_new,a,b,c,d) for a,b,c,d in zip(args[1],args[2],args[3],args[4])])
    else:
        raise RuntimeError("einsumk is not implemented for number of matrices {}".format(nmat))
    