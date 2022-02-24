import numpy as np
import abc



"""some basic classes to construct formulae for evaluation"""

class Formula_ln(abc.ABC):

    @abc.abstractmethod
    def __init__(self,data_K,
                internal_terms      = True,
                external_terms      = True,
                correction_wcc = False ,
                dT_wcc = False
                ):
        self.internal_terms = internal_terms
        self.external_terms = external_terms
        self.correction_wcc = correction_wcc
        if self.correction_wcc:
            if not (self.external_terms and self.internal_terms):
                raise ValueError(f"correction_wcc makes sense only with all terms, but called with "
                    f"internal:{self.internal_terms}"
                    f"external:{self.external_terms}"
                    )
            self.T_wcc = data_K.covariant('T_wcc')
            if dT_wcc:
                self.dT_wcc = data_K.covariant('T_wcc',gender=1)

    @abc.abstractmethod
    def ln(self,ik,inn,out):
        pass

    @abc.abstractmethod
    def nn(self,ik,inn,out):
        pass

    def nl(self,ik,inn,out):
        return self.ln(ik,out,inn)

    def ll(self,ik,inn,out):
        return self.nn(ik,out,inn)

    @property
    def additive(self):
        """ if Trace_A+Trace_B = Trace_{A+B} holds.
        needs override for quantities that do not obey this rule (e.g. Orbital magnetization)
        """
        return True

    def trace(self,ik,inn,out):
        return np.einsum("nn...->...",self.nn(ik,inn,out)).real



class Matrix_ln(Formula_ln):
    "anything that can be called just as elements of a matrix"

    def __init__(self,matrix,TRodd = None, Iodd = None):
        self.matrix=matrix
        self.ndim=len(matrix.shape)-3
        if TRodd is not None:
            self.TRodd = TRodd
        if Iodd is not None:
            self.Iodd = Iodd

    def ln(self,ik,inn,out):
        return self.matrix[ik][out][:,inn]

    def nn(self,ik,inn,out):
        return self.matrix[ik][inn][:,inn]


class Matrix_GenDer_ln(Formula_ln):
    "generalized erivative of MAtrix_ln"
    def __init__(self,matrix,matrix_comader,D,TRodd = None, Iodd = None):
        self.A  = matrix
        self.dA = matrix_comader
        self.D  =  D
        self.ndim=matrix.ndim+1
        if TRodd is not None:
            self.TRodd = TRodd
        if Iodd is not None:
            self.Iodd = Iodd

    def nn(self,ik,inn,out):
        summ=self.dA.nn(ik,inn,out)
        summ -= np.einsum( "mld,ln...->mn...d" , self.D.nl(ik,inn,out) , self.A.ln(ik,inn,out) )
        summ += np.einsum( "ml...,lnd->mn...d" , self.A.nl(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ

    def ln(self,ik,inn,out):
        summ= self.dA.ln(ik,inn,out)
        summ -= np.einsum( "mld,ln...->mn...d" , self.D.ln(ik,inn,out) , self.A.nn(ik,inn,out) )
        summ += np.einsum( "ml...,lnd->mn...d" , self.A.ll(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ



class FormulaProduct(Formula_ln):

    """a class to store a product of several formulae"""
    def __init__(self,formula_list,name="unknown",hermitian=False,transpose = None):
        if type(formula_list) not in (list,tuple):
            formula_list=[formula_list]
        self.TRodd=bool(sum(f.TRodd for f in formula_list)%2)
        self.Iodd =bool(sum(f.Iodd for f in formula_list)%2)
        self.name=name
        self.formulae=formula_list
        self.hermitian = hermitian
        self.transpose = (0,1)+(i+2 for i in transpose) if transpose is not None else None
        ndim_list = [f.ndim for f in formula_list]
        self.ndim = sum(ndim_list)
        self.einsumlines=[]
        letters = "abcdefghijklmnopqrstuvw"
        dim = ndim_list[0]
        for d in ndim_list[1:]:
            self.einsumlines.append( "LM"+letters[:dim]+",MN"+letters[dim:dim+d]+"->LN"+letters[:dim+d])
            dim+=d

    def nn(self,ik,inn,out):
        matrices = [frml.nn(ik,inn,out) for frml in self.formulae ]
        res=matrices[0]
        for mat,line in zip(matrices[1:],self.einsumlines):
            res=np.einsum(line,res,mat)
        if self.hermitian:
            res=0.5*(res+res.swapaxes(0,1).conj())
        if self.transpose is not None:
            res = res.transpose(self.transpose)
        return np.array(res,dtype=complex)

    def ln(self,ik,inn,out):
        raise NotImplementedError()



class FormulaSum(Formula_ln):

    """a class to store a sum of several formulae"""
    def __init__(self,formula_list,coefs=None,name="unknown",hermitian=False):
        if type(formula_list) not in (list,tuple):
            formula_list=[formula_list]
        if coefs is None:
            coefs = [1 for f in formula_list]
        self.TRodd=formula_list[0].TRodd  
        assert np.all([f.TRodd==self.TRodd for f in formula_list]) 
        self.Iodd=formula_list[0].Iodd  
        assert np.all([f.Iodd==self.Iodd for f in formula_list]) 
        self.ndim=formula_list[0].ndim
        assert np.all([f.ndim==self.ndim for f in formula_list]) 
        self.name=name
        self.formulae=formula_list
        self.coefs=coefs
        self.hermitian = hermitian


    def nn(self,ik,inn,out):
        res = sum(frml.nn(ik,inn,out)*x for frml,x in zip(self.formulae,self.coefs))
        if self.hermitian:
            res=0.5*(res+res.swapaxes(0,1).conj())
        return np.array(res,dtype=complex)

    def ln(self,ik,inn,out):
        raise NotImplementedError()


