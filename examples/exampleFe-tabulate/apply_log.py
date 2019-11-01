import numpy as np
import sys

x0=10

def funlog(x):
    res=np.zeros(x.shape)
    res[abs(x)<x0]=x[abs(x)< x0]/x0
    res[x>= x0]=np.log( x[x>= x0]/x0)+1
    res[x<=-x0]=np.log(-x[x<=-x0]/x0)-1
    return res



fin=open(sys.argv[1]).readlines()

#print (fin)

NK=np.prod(np.array(fin[0].split(),dtype=int))
NB=int(fin[2])
print ("NK={}, NB={}".format(NK,NB))
print(len(fin))
fout=open(sys.argv[1]+"log{}".format(x0),"w")
fout.write("".join(fin[:NK*NB+6]))
X=np.array(fin[NK*NB+6:2*NK*NB+6],dtype=float)
#print (X)
X=funlog(X)
np.savetxt(fout,X)
fout.close()


    
