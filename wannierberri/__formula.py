import numpy as np
from .__utility import  alpha_A,beta_A
from numba import njit

#####################################################
#####################################################

INNER_IND="lmnpqrsto"
OUTER_IND="LMNPRRSTP"
INDICES=INNER_IND+OUTER_IND


def inner_outer(ind,inner,outer):
    if   ind in INNER_IND:
       return inner
    elif ind in OUTER_IND:
       return outer
    else:
       raise ValueError("unknown type of index : '{}'".formta(ind))

class Formula():
    """a class to write a formula as trace of matrix product,
     where inner summation is either over 'inner'/'occupied'  or 'outer'/'unoccupied' states
    as input takes a list of entriies like 
    ('nm,mL,L,Ln',A,B,C,D)  
    which will mean :math:`\sum_nm&{\rm in}\sum_L^{\rm out} A){nm}B_{mL}C_{L}D_{Ln}`
    by convention lower case indices correspond to 'inner' states, upper case --- to 'outer' states
    indices  of neighbouring matrices should match (i.e. 'mn,qn' is forbidden), 
    first and last index should be different, if more then one matrix is given.
    """
    def __init__(self,TRodd=False,Iodd=False,ndim=0):
        self.TRodd=TRodd
        self.Iodd=Iodd
        self.ndim=ndim
        self.term_list=[]
        self.hermitian=hermitian

    def __call__(self,ik,ib_start,ib_end,trace):
        res = sum( term(ik,ib_in_start,ib_in_end) for term in self.term_list )
        if trace: 
            return np.einsum(ind[0]+'...->...',m).real
        else:
            if self.Hermitian:
                res=0.5*(res+swapaxes(res,0,1).conj())
            return np.array(res,dtype=complex)

    def add_term(self,ind_string ,mat_list ,mult=1.):
        self.term_list.append( 
                  MatrixProductTerm(term,ind_string , mat_list ,self.ndim , mult)
                             )


class MatrixProductTerm():
    """ a term containing product of matrices, to be evaluated as a matrix or as a trace
        mat_list should contain matrices of shape [ik,ib1,ib2,a,b,...] or [ik,ib,a,b,...]
        same index should appear only on neighbouring positions (or frst and last)
    """

    def __init__(self,ind_string ,mat_list,ndim,mult):
        self.check_bounds   # defined in a child class
        self.nb=mat_list[0].shape[1]
        self.ndim=ndim
        self.mult=mult
        self.check_bounds(line)

        line=ind_string.replace(' ','')  # remove spaces if appear
        line=line.replace('.',',') # allow to use dot instead of comma
        # now goes a long boring check that the string is valid
 
        allowed_symbols=INDICES+",. "
        forbidden_symbols=set(line)-set(allowed_symbols) # everything that is not allowed is forbidden
        if len(forbidden_symbols) > 0 :
            raise FormulaIndicesError(ind_string,'any',"Unrecognized symbols ( {} ) ".format(
                        " , ".join("'{}'".format(s) for s in forbidden_symbols) )  )

        terms=line.split(',')
        if len(terms>1) and term[0][1]==term[-1][-1] :
            raise FormulaIndicesError(ind_string,'first and last indices should be different')
        if term[0][0] not in INNER_IND:
            raise FormulaIndicesError(ind_string,"first index should be innner i.e. in '{}'".format(INNER_IND))
        if term[-1][-1] not in INNER_IND:
            raise FormulaIndicesError(ind_string,"last  index should be innner i.e. in '{}'".format(INNER_IND))
        if len(terms)==0: 
            raise FormulaIndicesError(ind_string,'any',"does not contain any terms")
        if len(terms)!=len(mat_list) 
            raise FormulaIndicesError(ind_string,'any',"number of terms in the string {} is not equal to number of matrices supplied {}".format(len(line),len(mat_list)))
        for ind1,ind2 in zip(terms,terms[1:]):
            if ind1[-1]!=ind2[0]:
                raise FormulaIndicesError(ind_string,'any',"matching indices {} and {} are different".format(ind1,ind2))
        self.mat_list=[]
        self.ind_list=[]
        
        for term,mat in zip(terms,mat_list):
            assert isinstance(mat,np.ndarray) , "inputs should be arrays"
            if len(term)>2 : 
                raise FormulaIndicesError(ind_string,'any',"more then two indices in '{}' ".format(term))
            assert mat.ndim==self.ndim+1+len(term) ,  ( 
                          "shape of matrix {} does not correspond to index '{}' and dimension of the result {}".format(
                               mat.shape,term,self.ndim)  )
            if len(term)==2:
                if term[0]==tern[1]:
                    mat=mat[:,self.ibrange,self.ibrange]
            self.mat_list.append(mat)
            self.term_list.apped(term)
            

    @lazy_property.LazyProperty
    def ibrange(self):
        return np.arange(self.nb)


    def __call__(self,ik,ib_in_start,ib_in_end):
        inner=np.arange(ib_in_start,ib_in_end)
        outer=np.concatenate( (np.arange(0,ib_in_start),np.arange(ib_in_end,self.nb)) )
        get_index = lambda i : inner_outer(i,inner,outer)
        mat_list=[]
        ind_list=[]
        for ind,m in zip(ind_str_list,mat_list_in):
            m1=m[ik][get_index(ind[0])]
            if len(ind)==2 :
                m1=m1[get_index(inf[1])]
            mat_list.append(m1)
            ind_list.append(ind)
        while len(mat_list)>1:
            ind1=ind_list[-2]
            ind2=ind_list[-1]
            ind_new=ind1[:-1]+ind2[1:]
            mat_list_new[-2] = np.einsum(ind1+'...,'+ind2+'...->'+ind_new+'...', mat_list_new[-2],mat_list_new[-1],optimize=True)
            del mat_list[-1]
            del ind_list[-1]
            ind_list[-1]=ind_new
        return mat_list[0]
        return m*self.mult


class FormulaIndicesError(ValueError):
    def __init__(self, ind_string,reason):
        # Call the base class constructor with the parameters it needs
        super().__init__(  "The following string of indices '{}' sre not valid".format(ind_string)+
            +"because '{}' ".format(reason) )
