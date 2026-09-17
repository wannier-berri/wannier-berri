from wannierberri.w90files.wandata_soc import WannierDataSOC
from wannierberri.system import SystemSOC

wandata = WannierDataSOC.from_npz("wannier_soc", files=["mmn", "eig", "symmetrizer", "chk" ,"amn"])
wandata.wannierise(num_iter=100, sitesym=True,
                    froz_min=-1000, froz_max=15)
system = SystemSOC.from_wannierdata(wandata)
