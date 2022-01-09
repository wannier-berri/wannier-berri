from . import fermiocean


class calculator():
    def __init__(self,degen_thresh=1e-4,degen_Kramers=False,save_mode="bin+txt"):
        self.degen_thresh = degen_thresh
        self.degen_Kramers = degen_Kramers
        self.save_mode = save_mode

class AHC(calculator):
    def __init__(self,Efermi,tetra=False,kwargs_formula={},**kwargs):
        self.Efermi = Efermi
        self.tetra  = tetra
        self.kwargs_formula = kwargs_formula
        super().__init__(**kwargs)
    
    def __call__(self,data_K):
        res =  fermiocean.AHC(data_K,self.Efermi,tetra=self.tetra,degen_thresh=self.degen_thresh,degen_Kramers=self.degen_Kramers,**self.kwargs_formula)
        res.set_save_mode(self.save_mode)
        return res
