import numpy as np


from time import time

class ws_dist_map():
    def __init__(self,iRvec,num_wann,file_ws=None):
        nRvec=iRvec.shape[0]
        self.num_wann=num_wann
        self._iRvec_new=dict()
        t0=time()
        if file_ws is None:
            for ir in range(nRvec):
                self._iRvec_new[tuple(iRvec[ir])]=dict({ir:np.ones((num_wann,num_wann))})
        else:
            f=open(file_ws,"r")
            f.readline()
            for ir in range(nRvec): 
                for iw in range(num_wann):
                    for jw in range(num_wann):
                        l=np.array(f.readline().split(),dtype=int)
                        assert(l[3]==iw+1)
                        assert(l[4]==jw+1)
                        irvec_old=l[:3]
                        assert(  tuple(iRvec[ir])==tuple(irvec_old) )
                        ndeg=int(f.readline())
                        for ideg in range(ndeg):
                            irvec_new=np.array( f.readline().split(),dtype=int )+irvec_old
                            self._add(ir,irvec_new,iw,jw,ndeg)
        t1=time()
        self._iRvec_ordered=sorted(self._iRvec_new)
        for ir  in range(nRvec):
            chsum=0
            for irnew in self._iRvec_new:
                if ir in self._iRvec_new[irnew]:
                    chsum+=self._iRvec_new[irnew][ir]
            chsum=np.abs(chsum-np.ones( (num_wann,num_wann) )).sum() 
            if chsum>1e-12: print "WARNING: Check sum for ",ir," : ",chsum
        t2=time()
        print ("time for reading wsvec: {0}, for chsum: {1}".format(t1-t0,t2-t1))

    def _add(self,ir,irvec_new,iw,jw,ndeg):
        irvec_new=tuple(irvec_new)
        if not (irvec_new in self._iRvec_new):
             self._iRvec_new[irvec_new]=dict()
        if not ir in self._iRvec_new[irvec_new]:
             self._iRvec_new[irvec_new][ir]=np.zeros((self.num_wann,self.num_wann),dtype=float)
        self._iRvec_new[irvec_new][ir][iw,jw]+=1./ndeg
        
    def __call__(self,matrix):
        ndim=len(matrix.shape)-3
        num_wann=matrix.shape[0]
        reshaper=(num_wann,num_wann)+(-1,)*ndim
        matrix_new=np.array([ sum(matrix[:,:,ir]*self._iRvec_new[irvecnew][ir].reshape(reshaper)
                                  for ir in self._iRvec_new[irvecnew] ) 
                                       for irvecnew in self._iRvec_ordered]).transpose( (1,2,0)+tuple(range(3,3+ndim)) )
        assert ( np.abs(matrix_new.sum(axis=2)-matrix.sum(axis=2)).max()<1e-12)
        return matrix_new
             

#    def mapmat0d(self,matrix):
#        return self.mapmat(matrix)
#        matrix_new= np.array([ sum(matrix[:,:,ir]*self._iRvec_new[irvecnew][ir]
#                                  for ir in self._iRvec_new[irvecnew] ) 
#                                       for irvecnew in self._iRvec_ordered]).transpose( (1,2,0)  )        
#        assert ( np.abs(matrix_new.sum(axis=2)-matrix.sum(axis=2)).max()<1e-12)
#        return matrix_new
