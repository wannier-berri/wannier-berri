import numpy as np

def E_to_DOS(E,sigma=0.1,emin=None,emax=None,de=None,divsigma=10,nsigma=5):
    if emin is None: emin=E.min()-nsigma*sigma
    if emax is None: emax=E.max()+nsigma*sigma
    if de is None:de=sigma/2
    E1=E.reshape(-1)
    print (E.shape)
    select=np.where(  (E1>emin-nsigma*sigma)*(E1<emax+nsigma*sigma) )[0]
    E1=E1[select]
    
    de_small=sigma/divsigma
    sigma_fun_x=np.linspace(-nsigma*sigma,nsigma*sigma,divsigma*2*nsigma+1)
    sigma_fun_y=np.exp(-(sigma_fun_x/sigma)**2)/(sigma*np.sqrt(np.pi))

    icenter=np.array((E1-emin)/sigma*divsigma,dtype=int)
    start=icenter-nsigma*divsigma
    end=icenter+nsigma*divsigma+1


    DOS=np.zeros(int((emax-emin)/de_small))
    n_x=DOS.shape[0]
    start[start<0]=0
    end[end>n_x]=n_x
    start1=divsigma*nsigma-icenter+start
    end1=divsigma*nsigma-icenter+end
    
    for s,e,s1,e1 in zip(start,end,start1,end1):
        if (e>s) and (e1>s1): DOS[s:e]+=sigma_fun_y[s1:e1]
    

    return(emin+de_small*np.arange(n_x),DOS)

