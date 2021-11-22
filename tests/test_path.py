import wannierberri as wb
import pythtb as ptb
import numpy as np

def test_path():
    def HaldanePTB(delta,t1,hop2,phi):
        lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
        orb=[[1./3.,1./3.],[2./3.,2./3.]]
        haldane=ptb.tb_model(2,2,lat,orb)
        haldane.set_onsite([-delta,delta])
        haldane.set_hop(t1, 0, 1, [ 0, 0])

        return haldane


    haldane = HaldanePTB(2,1,1/3,np.pi/10)
    syst=wb.System_PythTB(haldane,berry=True,morb=True)

    k_nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    pp = wb.Path(syst, k_nodes=k_nodes, nk=3)

    assert pp.labels == {0: '1', 2: '2'}, "pp.labels is wrong"
    assert np.all(pp.K_list == np.array([[0,  0., 0.], [0.25, 0.25 ,0.25], [0.5, 0.5, 0.5]])), "pp.K_list is wrong"

