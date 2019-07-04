import numpy as np


class ws_dist_map():
    def __init__(self,iRvec,num_wann,file_ws=None):
        nRvec=iRvec.shape[0]
        self.num_wann=num_wann
        self._iRvec_new=dict()
        if file_ws is None:
            for ir in range(nRvec):
                self._iRvec_new[tuple(iRvec[ir])]=dict({ir:np.ones((num_wann,num_wann))})
        else:
            f=open(file_ws,"r")
            f.readline()
            for ir in range(nRvec): 
                for iw in range(num_wann):
                    for jw in range(num_wann):
                        irvec_old=np.array(f.readline().split()[:3],dtype=int)
#                        print iRvec[ir],irvec_old
                        assert(  tuple(iRvec[ir])==tuple(irvec_old) )
                        ndeg=int(f.readline().split()[0])
                        for ideg in range(ndeg):
                            irvec_new=np.array(f.readline().split()[:3],dtype=int)+irvec_old
                            self._add(ir,irvec_new,iw,jw,ndeg)
        
        self._iRvec_ordered=sorted(self._iRvec_new)
        print ir," old vectors read ",len(self._iRvec_new)," new vectors"
        print "control sum: \n",sum(A for vecsold in self._iRvec_new.values() for A in vecsold.values())/nRvec
#        print self._iRvec_new

    def _add(self,ir,irvec_new,iw,jw,ndeg):
        irvec_new=tuple(irvec_new)
        if not (irvec_new in self._iRvec_new):
             self._iRvec_new[irvec_new]=dict()
        if not ir in self._iRvec_new[irvec_new]:
             self._iRvec_new[irvec_new][ir]=np.zeros((self.num_wann,self.num_wann),dtype=float)
        self._iRvec_new[irvec_new][ir][iw,jw]+=1./ndeg
        
    def mapmat(self,matrix):
        ndim=len(matrix.shape)-3
        num_wann=matrix.shape[0]
        reshaper=(num_wann,num_wann)+(-1,)*ndim
        return np.array([ sum(matrix[:,:,ir]*self._iRvec_new[irvecnew][ir].reshape(reshaper)
                                  for ir in self._iRvec_new[irvecnew] ) 
                                       for irvecnew in self._iRvec_ordered]).transpose( (1,2,0)+tuple(range(3,3+ndim)) )
             


if __name__ == '__main__':
    import cProfile
    cProfile.run('ws_dist_map("Fe_wsvec.dat",597,18)')
             
#wsdist=ws_dist_mapping("Fe_wsvec.dat",18)


"""        
      write (file_unit, '(A)') trim(header)

      do irpt = 1, nrpts
        do iw = 1, num_wann
          do jw = 1, num_wann
            write (file_unit, '(5I5)') irvec(:, irpt), iw, jw
            write (file_unit, '(I5)') wdist_ndeg(iw, jw, irpt)
            do ideg = 1, wdist_ndeg(iw, jw, irpt)
              write (file_unit, '(5I5,2F12.6,I5)') irdist_ws(:, ideg, iw, jw, irpt) - &
                irvec(:, irpt)
            end do
          end do
        end do
      end do
"""