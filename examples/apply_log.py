import numpy as np
import sys

try:
    x0=float(sys.argv[2])
except:
    x0=10.

def funlog(x):
    res=np.zeros(x.shape)
    res[x>= 0]= np.log( 1+x[x>=0]/x0)
    res[x<= 0]=-np.log( 1-x[x<=0]/x0)
    return res


fin=open(sys.argv[1]).readlines()


NK=np.prod(np.array(fin[0].split(),dtype=int))
NB=int(fin[2])
fout=open(sys.argv[1]+"log{:.1f}".format(x0),"w")
fout.write("".join(fin[:NK*NB+6]))
X=np.array(fin[NK*NB+6:2*NK*NB+6],dtype=float)
X=funlog(X)
np.savetxt(fout,X)
fout.close()


    
