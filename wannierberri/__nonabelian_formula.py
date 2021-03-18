import numpy as np
from .__utility import  alpha_A,beta_A
from .__trace_formula import TraceFormula
#####################################################
#####################################################

def identity(data_K,op=None,ed=None):
        "an identity operator (to compute DOS)"
        # first give our matrices short names
        NB= data_K.nbands
        # This is the formula to be implemented:
        ones=np.ones( (ed-op,NB),dtype=float )
        formula =  TraceFormula ( [ ('n', ones ) ],ndim=0,TRodd=False,Iodd=False)
        return formula





class NonabelianFormula(TraceFormula):
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

    def __input_to_inner(self,inp):
        "transform input nonabelian formula to the internal storage format, used by FermiOcean class"
        line=inp[0].replace(' ','')  # remove spaces if appear
        line=line.replace('.',',') # allow to use dot instead of comma
        if len(line)==0: 
            return tuple()
        # now goes a long boring check that the string is valid
        inner_ind="kmno"
        outer_ind="lpqr"
        indices=inner_ind+outer_ind
        try : 
            allowed_symbols=indices+","
            forbidden_symbols=set(line)-set(allowed_symbols) # everything that is not allowed is forbidden
            if len(forbidden_symbols) > 0 :
                raise ValueError("Unrecognized symbols ( {} ) were found in string {} ".format(
                        " , ".join("'{}'".format(s) for s in forbidden_symbols), line)  )
            assert  line[ 0] == 'm' , "string should start with 'm'"
            assert  line[-1] == 'n' , "string should end   with 'n'"
            terms=line.split(',')
            num_mat=len(terms)
            assert num_mat==len(inp)-1 , "the strig contains {} terms, but {} arrays are given".format(len(terms),len(inp)-1)
            for term,mat in zip(terms,inp[1:]):
                assert mat.ndim==self.ndim+1+len(term) ,  ( 
                          "shape of matrix {} does not correspond to index '{}' and dimension of the result {}".format(
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
