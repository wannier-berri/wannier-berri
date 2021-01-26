from .__utility import find_degen

import itertools
import numpy as np
from lazy_property import LazyProperty



class FiniteDifferences():

    def __init__(self,recip_lattice,FFT):
        self.FFT=FFT
        self.recip_lattice=recip_lattice
        self.wk,self.bki,self.neighbours = get_neighbours_FFT(self.recip_lattice,self.FFT)
        self.bk_cart=self.bki.dot(self.basis)

    @LazyProperty
    def basis(self):
        return np.array(self.recip_lattice)/np.array(self.FFT)


def find_shells(basis,isearch=3):
        """returns the weights of the bk vectors, and the bk vectors to the corresponding neighbour points"""
        if isearch>6:
            raise RuntimeError('Failed to sattisfy (B1) criteria of PRB 56, 12847 (1997) upto {} cells . Must be smth wrong'.format(6))
        search=np.arange(-isearch,isearch+1)
        bki=np.array(np.meshgrid(search,search,search)).reshape(3,-1,order='F').T
#        print (bki.shape,basis.shape)
        bk=bki.dot(basis)
        leng=np.linalg.norm(bk,axis=1)
        srt=np.argsort(leng)
        bk=bk[srt]
        bki=bki[srt]
        leng=leng[srt]
#        print ("leng=",leng[:50])
        shells=find_degen(leng,1e-8)[1:]   # omit the first "0" shell
        shell_mat=np.array([ bk[b1:b2].T.dot(bk[b1:b2])  for b1,b2 in shells])
        
#       now select the proper shells
        selected_shells=[]
        selected_ishells=[]
        for ishell_try,shell_try in enumerate(shells[:50]):
            if not check_parallel(bk,selected_shells,shell_try):
                continue
            accept,checkB1,weights=check_B1(shell_mat,selected_ishells+[ishell_try])
            if accept:
                selected_shells.append(shell_try)
                selected_ishells.append(ishell_try)
            if checkB1:
                break
        wk =np.array([w      for w,shell in zip(weights,selected_shells) for i in range(shell[0],shell[1]) if abs(w)>1e-8 ])
        bki=np.array([bki[i] for w,shell in zip(weights,selected_shells) for i in range(shell[0],shell[1]) if abs(w)>1e-8 ])
        return wk,bki
  

def check_parallel(bk,selected_shells,shell_try):
    for sh in selected_shells:
        for i in range(sh[0],sh[1]):
            for j in range(shell_try[0],shell_try[1]):
                if np.linalg.norm(np.cross(bk[i],bk[j]))/(np.linalg.norm(bk[i] ) * np.linalg.norm(bk[j]) ) < 1e-6:
                    return False
    return True

def check_B1(shell_mat,selected_shells):
    "returns accept,B1true,weights "
#    print ("selected shells try: ",selected_shells)
    selected_shells=np.array(selected_shells)
    shell_mat=shell_mat[np.array(selected_shells)]
    shell_mat_line=shell_mat.reshape(-1,9)
    u,s,v=np.linalg.svd(shell_mat_line,full_matrices=False)
    if np.any(abs(s)<1e-7):
#        print ("rejecting last shell of {} due to small singular value(s) {}".format(selected_shells,s))
        return False,False,None
    else:
        s=1./s
        weight_shell=np.eye(3).reshape(1,-1).dot(v.T.dot(np.diag(s)).dot(u.T)).reshape(-1)
        check_eye=sum(w*m for w,m in zip(weight_shell,shell_mat))
        tol=np.linalg.norm(check_eye-np.eye(3))
        if tol>1e-5 :
#            print("The following matrix :\n {} \n from shells {} is not identity identity by an error of {}. Searchong for more shells\n ".format( #   Further debug informstion :  \n bk={} \n bki={} \n leng={}\nshells={}\nshell_mat={}\weight_shell={}\n".format(
#                              check_eye,selected_shells,tol)) #, bk,bki,leng,shells,shell_mat,weight_shell) )
            return True,False,None
        else : 
            return True,True,weight_shell

def get_neighbours_FFT(recip_lattice,FFT):
        """returns the weights of the bk vectors, and the corresponding neighbour points in the flattened array of k-points"""
        NFFT_tot=np.prod(FFT)
        wk,bki=find_shells(np.array(recip_lattice)/np.array(FFT))
        kindex=np.arange(NFFT_tot)
        ki=np.array([ kindex//(FFT[1]*FFT[2]),
                      (kindex//FFT[2])%FFT[1],
                      kindex%FFT[2]]).T
#        print ("kpoint indices are :",(ki[:,0]*FFT[1]     + ki[:,1]  )*FFT[2]  +   ki[:,2] )
        neigh=np.array([ (ki+b[None,:])%FFT  for b in bki])
        neighbours=(neigh[:,:,0]*FFT[1]     + neigh[:,:,1]  )*FFT[2]  +   neigh[:,:,2]
#        print ("neigh=",neigh)
#        print ("the weights are:",wk)
#        print ("the bki are:",bki)
#        print ("the neighbours of 0th kp are:",neighbours[:,0])
#        for i in np.array(np.random.random(10)*NFFT_tot,dtype=int):
#            print ("the neighbours of {}th kp are: {}".format(i,neighbours[:,i]))
        return wk,bki,neighbours



if __name__ == "__main__":
    from sys import argv
    c=float(argv[1])
    FD=FiniteDifferences(recip_lattice=[[1,0,0],[-1/2,np.sqrt(3)/2,0],[0,0,c]],FFT=[4,4,5])
    
