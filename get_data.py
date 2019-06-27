import numpy as np
from scipy.io import FortranFile as FF
from aux import str2bool

class Data():

    def __init__(self,seedname="wannier90",getAA=False,getBB=False,getCC=False,getSS=False):
        f=open(seedname+"_HH_save.info","r")
        l=f.readline().split()[:3]
        self.seedname=seedname
        self.num_wann,self.nRvec,self.spinors=int(l[0]),int(l[1]),str2bool(l[2])
        self.real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        iRvec=np.array([f.readline().split()[:4] for i in range(self.nRvec)],dtype=int)
        f.close()
        self.cell_volume=np.linalg.det(self.real_lattice)
        self.Ndegen=iRvec[:,3]
        self.iRvec=iRvec[:,:3]
        self.cRvec=self.iRvec.dot(self.real_lattice)

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Real-space lattice:\n",self.real_lattice)
        #print ("R - points and dege=neracies:\n",iRvec)
        
        self.HH_R=self.__getMat('HH')
        
        if getAA:
            self.AA_R=self.__getMat('AA')
            
        if getBB:
            self.BB_R=self.__getMat('BB')

        if getCC:
            self.CC_R=self.__getMat('CC')

        if getSS:
            self.SS_R=self.__getMat('SS')
            

    def __getMat(self,suffix):

        f=FF(self.seedname+"_" + suffix+"_R.dat")
        MM_R=np.array([[np.array(f.read_record('2f8'),dtype=float) for m in range(self.num_wann)] for n in range(self.num_wann)])
        MM_R=MM_R[:,:,:,0]+1j*MM_R[:,:,:,1]
        f.close()
        ncomp=MM_R.shape[2]/self.nRvec
        if ncomp==1:
            return MM_R/self.Ndegen[None,None,:]
        elif ncomp==3:
            return MM_R.reshape(self.num_wann, self.num_wann, 3, self.nRvec).transpose(0,1,3,2)/self.Ndegen[None,None,:,None]
        elif ncomp==9:
            return MM_R.reshape(self.num_wann, self.num_wann, 3,3, self.nRvec).transpose(0,1,4,3,2)/self.Ndegen[None,None,:,None,None]
        

