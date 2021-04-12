import numpy as np
from .__utility import  alpha_A,beta_A

#####################################################
#####################################################

def Omega(data_K,op=None,ed=None):
        "an attempt for a faster implementation"
        # first give our matrices short names
        NB= data_K.nbands
        A = data_K.A_Hbar[op:ed]
        D = data_K.D_H[op:ed]
        O = np.einsum('knna->kna',data_K.Omega_Hbar[op:ed]).real
        # now define the "alpha" and "beta" components
        A_,D_={},{}
        for var in 'A','D':
            for c in 'alpha','beta':
                locals()[var+"_"][c]=locals()[var][:,:,:,globals()[c+'_A']]
        # This is the formula to be implemented:
        formula =  TraceFormula ( [ ('n', O ) ],ndim=1,TRodd=True,Iodd=False)
        formula.add_term( ('nl,ln',D_['alpha'],-2j*D_['beta' ] ) )
        formula.add_term( ('nl,ln',D_['alpha'],-2 *A_['beta' ] ) )
        formula.add_term( ('nl,ln',D_['beta' ], 2 *A_['alpha'] ) )
#       or equivalently:
#        for s,a,b in ( +1.,'alpha','beta'),(-1.,'beta','alpha'):
#            formula.add_term( ('nl,ln',D_[a],- 2*A_[b]*s ) )
#            formula.add_term( ('nl,ln',D_[a],-1j*D_[b]*s ) )
        return formula


def derOmega(data_K,op=None,ed=None):
        "an attempt for a faster implementation"
        # first give our matrices short name
        #print ("using kpoint [{}:{}]".format(op,ed))
        A  = data_K.A_Hbar[op:ed]
        dA = data_K.A_Hbar_der[op:ed]
#        print ("dA=",dA)
        _D = data_K.D_H[op:ed]
        _V = data_K.V_H[op:ed]
        O  = data_K.Omega_Hbar[op:ed,:,:,:,None]
        dO = data_K.Omega_bar_der_rediag[op:ed]
        W  = data_K.del2E_H[op:ed]


        Acal= (-(A+1j*_D)*data_K.dEig_inv[op:ed,:,:,None])[:,:,:,:,None]
        A  =  A[:,:,:,:,None]
        D  = _D[:,:,:,:,None]
        Dd = _D[:,:,:,None,:]
        V  = _V[:,:,:,:,None]
        Vd = _V[:,:,:,None,:]

        del _D,_V

        # now define the "alpha" and "beta" components
        A_,D_,W_,V_,Acal_,dA_={},{},{},{},{},{}
        for var in 'A','D','Acal','W','V','dA':
            for c in 'alpha','beta':
#                print (var,c,locals()[var].shape)
                locals()[var+"_"][c]=locals()[var][:,:,:,globals()[c+'_A']]
        # This is the formula to be implemented:
        # orange terms
        formula =  TraceFormula ( [ ('n', dO ) ],ndim=2,TRodd=False,Iodd=True)
        formula.add_term  ( ('nl,ln',Dd, -2*O ) )
        for s,a,b in ( +1.,'alpha','beta'),(-1.,'beta','alpha'):
            #  blue terms
            formula.add_term( ('nl,ln',    Acal_ [a] ,  2*W_[b] *s ) )
            formula.add_term( ('nl,lp,pn', Acal_ [a] ,  2*V_[b] *s, Dd ) )
            formula.add_term( ('nl,lp,pn', Acal_ [a] ,  2*Vd    *s, D_[b] ) )
            formula.add_term( ('nl,lm,mn', Acal_ [a] , -2*D_[b] *s, Vd ) )
            formula.add_term( ('nl,lm,mn', Acal_ [a] , -2*Dd    *s, V_[b] ) )
            #  green terms
            formula.add_term( ('nl,ln',       D_ [a] , -2*dA_[b]*s ) )
            formula.add_term( ('nl,lp,pn',    D_ [a] , -2*A_[b] *s, Dd ) )
            formula.add_term( ('nl,lm,mn',    D_ [a] ,  2*Dd    *s, A_[b] ) )
        return formula




class TraceFormula():
    """a class to write a formula as trace of matrix product,
     where inner summation is either over 'inner'/'occupied'  or 'outer'/'unoccupied' states
    as input takes a list of entriies like 
    ('nm,ml,ln',A,B,C)  
    which will mean :math:`\sum_nm&{\rm in}\sum_l^{\rm out} A){nm}B_{ml}C_{ln}`
    by convention indices m,n,o correspond to 'inner' states, l,p,q  -- to 'outer' states
    indices  of neighbouring matrices should match (i.e. 'mn,qn' is forbidden), 
    first and last index shopuld be 'n'

    automatic grouping of terms by first matrix will be implemented, if it is the same and 
    has same type of indices. But onl;y if is really the same, not a copy ( `A1 is A2 ==True `)
    """
    def __init__(self,trace_list,TRodd=False,Iodd=False,ndim=0):
        self.TRodd=TRodd
        self.Iodd=Iodd
        self.ndim=ndim
        self.trace_list=[self.__input_to_inner(term) for term in trace_list]

    def add_term(self,trace):
        self.trace_list.append(self.__input_to_inner(trace) )

    def __len__(self):
        return len(self.trace_list)

    def group_terms(self):
#        print ("grouping {} terms".format(len(self)))
        for i,a in enumerate(self.trace_list):
            if len(a)>2:
                for j in range(len(self)-1,i,-1): 
                    b=self.trace_list[j]
                    if a[0]==b[0]  and a[1] is b[1] :
                        for c in b[2]:
                            a[2].append(c)
                        del self.trace_list[j]
#        print ("after grouping : {} terms".format(len(self)))
                

    def __input_to_inner(self,inp):
        "transform input trace formula to the internal storage format, used by FermiOcean class"
        line=inp[0].replace(' ','')  # remove spaces if appear
        line=line.replace('.',',') # allow to use dot instead of comma
        if len(line)==0: 
            return tuple()
        # now goes a long boring check that the string is valid
        inner_ind="mno"
        outer_ind="lpq"
        indices=inner_ind+outer_ind
        try : 
            allowed_symbols=indices+","
            forbidden_symbols=set(line)-set(allowed_symbols) # everything that is not allowed is forbidden
            if len(forbidden_symbols) > 0 :
                raise ValueError("Unrecognized symbols ( {} ) were found in string {} ".format(
                        " , ".join("'{}'".format(s) for s in forbidden_symbols), line)  )
            assert  line[0]==line[-1]=='n' , "string should start and end with 'n'"
            terms=line.split(',')
            num_mat=len(terms)
            assert num_mat==len(inp)-1 , "the strig contains {} terms, but {} arrays are given".format(len(terms),len(inp)-1)
            for term,mat in zip(terms,inp[1:]):
                assert mat.ndim==self.ndim+1+len(term) ,  ( 
                          "shape of matrix {} does not correspond to index '{}' and dinetsion of the result {}".format(
                               mat.shape,term,self.ndim)  )
            if num_mat==1:
                assert terms[0] in ('n','nn','mn') , (
                        "a single term should be either trace('nn') or sum-over-states('n') "+
                        "or sum of the whole matrix ('mn' ) . found '{}'".format(terms[0])    )
                return  (terms[0],inp[1])
            for i,t in enumerate(terms):
                assert len(t)==2 , "term '{}' does not have two indices".format(t)
            for i in range(len(terms)-1):
                t1,t2=terms[i],terms[i+1]
                assert  t1[1]==t2[0] , "inner dimensions of terms '{}' and '{}' do not match".format(t1,t2)
            for a in inp[1:]:
                assert isinstance(a,np.ndarray) , "inputs should be arrays"
            for l in indices:
                assert line.count(l) <= 2 , "index '{}' occures more then twice".format(l)
        except ( ValueError, AssertionError ) as err:
            raise ValueError("the string '{}' or input arrays [ ] are not valid: {}".format(
                     line,err , " , ".join("{}".format(A.shape) for A in inp[1:]  ) ) )
        # now we can re4lax and process the input, assuming a single-term trace is already handled
        # now translate all indices into 'n' and 'l' 
        translate = {l:'l' for l in outer_ind}.update({l:'n' for l in inner_ind})
        lineright=terms[1]+"".join(t[1] for t in terms[2:])
        return terms[0],inp[1],[ (lineright,)+tuple(inp[2:]) ]
