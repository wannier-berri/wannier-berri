import numpy as np


class ws_dist_data():
    def __init__(self,iRvec,num_wann,file_ws=None):
        nRvec=iRvec.shape[0]
        self.nRvec=iRvec.shape[0]
        self.num_wann=num_wann
        self.wsdist_ndeg=np.zeros(self.num_wann,self.num_wann,self.nRvec)
        self.irdist_ws=np.zeros(3,8,self.num_wann,self.num_wann,self.nRvec)
        
        
        if True:
            f=open(file_ws,"r")
            f.readline()
            for ir in range(self.nRvec): 
                for iw in range(self.num_wann):
                    for jw in range(self.num_wann):
                        l=np.array(f.readline().split(),dtype=int)
                        assert(l[3]==iw+1)
                        assert(l[4]==jw+1)
                        irvec_old=l[:3]
#                        print iRvec[ir],irvec_old
                        assert(  tuple(iRvec[ir])==tuple(irvec_old) )
                        self.wsdist_ndeg[iw,jw,ir]=int(f.readline())
                        for ideg in range(self.wsdist_ndeg[iw,jw,ir]):
                            self.irdist_ws[:,ideg,iw,jw,ir]=np.array( f.readline().split(),dtype=int )+irvec_old

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