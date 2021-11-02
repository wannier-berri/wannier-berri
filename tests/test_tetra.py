import numpy as np
from wannierberri.__tetrahedron import weights_tetra
from pytest import approx



def check_tetra_derivatives(E,
                            Efermi=np.linspace(-0.6,0.6,100001),
                            acc1=1e-5,acc2=1e-3,plot=False):
        E=E-0.5
        Ecenter=E[0]
        Ecorner=E[1:].reshape(2,2,2)
        Ecorn=E[1:4]

        occ_sea    = weights_tetra (Efermi,Ecenter,Ecorn[0],Ecorn[1],Ecorn[2],der=0)
        occ_der1   = weights_tetra (Efermi,Ecenter,Ecorn[0],Ecorn[1],Ecorn[2],der=1)
        occ_der2   = weights_tetra (Efermi,Ecenter,Ecorn[0],Ecorn[1],Ecorn[2],der=2)
        norm1 = np.max(occ_der1)
        norm2 = np.max(occ_der2)
        occ_der1/=norm1
        occ_der2/=norm2

        occ_der1_fd=occ_sea*0
        occ_der1_fd[1:-1]=(occ_sea[2:]-occ_sea[:-2])/(Efermi[2]-Efermi[0])/norm1
        occ_der2_fd=occ_sea*0
        occ_der2_fd[2:-2]=(occ_sea[4:]+occ_sea[:-4]-2*occ_sea[2:-2])/(Efermi[2]-Efermi[0])**2/norm2

        print (norm1,max(abs(occ_der1 - occ_der1_fd)) , norm2,max(abs(occ_der2 - occ_der2_fd))  )
        error = None
        try :
            assert (occ_der1_fd==approx(occ_der1,abs=acc1)) , (
                "finite-diff first derivative of weights_tetra did not match he analytic expression by"+
                "{} > {}".format(max(abs(occ_der1 - occ_der1_fd)),acc1)+
                f"the energies were {Ecenter}, {Ecorn}"  )
            assert (occ_der2_fd==approx(occ_der2,abs=acc2)) , (
                "finite-diff second derivative of weights_tetra did not match he analytic expression by"+
                "{} > {}".format(max(abs(occ_der2 - occ_der2_fd)),acc2) +
                f"the energies were {Ecenter}, {Ecorn}"  )
        except Exception as err:
            plot = True
            error = err
            
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(Efermi,occ_sea   ,c='blue')
            plt.scatter(Efermi,occ_der1_fd ,c='green')
            plt.scatter(Efermi,occ_der2_fd ,c='red')
            plt.plot(Efermi,occ_der1 , c='yellow')
            plt.plot(Efermi,occ_der2 , c='cyan')

            for x in E[1:4]:
                plt.axvline(x,c='blue')
            plt.axvline(Ecenter,c='red')
            plt.xlim(-0.6,0.6)
            plt.savefig("tetra_weight.pdf")
        
        if error is not None:
            raise error

    
def test_tetra_derivatives():
    E=np.array([[0.2984, 0.4595, 0.5829, 0.7502, 0.7196, 0.9911, 0.7955, 0.1899, 0.3873],
       [0.1452, 0.0907, 0.4754, 0.4508, 0.3953, 0.459 , 0.3443, 0.5388, 0.559 ],
       [0.339 , 0.6026, 0.3444, 0.0816, 0.5054, 0.4948, 0.9665, 0.0043, 0.079 ],
       [0.8992, 0.8343, 0.2925, 0.6543, 0.9982, 0.796 , 0.7424, 0.3358, 0.0717],
       [0.214 , 0.9312, 0.8754, 0.137 , 0.6876, 0.9805, 0.8198, 0.7867, 0.8013],
       [0.6402, 0.6796, 0.5923, 0.3457, 0.2366, 0.629 , 0.5247, 0.4212, 0.3966],
       [0.9593, 0.2697, 0.2101, 0.2543, 0.5695, 0.1169, 0.1485, 0.6536, 0.0675],
       [0.3641, 0.1229, 0.7018, 0.0084, 0.5121, 0.0491, 0.2748, 0.0281, 0.9587],
       [0.1439, 0.5332, 0.9988, 0.4064, 0.1984, 0.4195, 0.9107, 0.0021, 0.6467],
       [0.7869, 0.3651, 0.6513, 0.8564, 0.216 , 0.6372, 0.8625, 0.5177, 0.247 ]])
    for e in E:
        check_tetra_derivatives(E=e)
