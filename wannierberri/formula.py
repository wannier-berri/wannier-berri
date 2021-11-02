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
                    "internal:{self.internal_terms}"
                    "external:{self.external_terms}"
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
        summ -= np.einsum( "mld,lnb...->mnb...d" , self.D.nl(ik,inn,out) , self.A.ln(ik,inn,out) )
        summ += np.einsum( "mlb...,lnd->mnb...d" , self.A.nl(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ

    def ln(self,ik,inn,out):
        summ= self.dA.ln(ik,inn,out)
        summ -= np.einsum( "mld,lnb...->mnb...d" , self.D.ln(ik,inn,out) , self.A.nn(ik,inn,out) )
        summ += np.einsum( "mlb...,lnd->mnb...d" , self.A.ll(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ



class FormulaProduct(Formula_ln):

    """a class to store a product of several formulae"""
    def __init__(self,formula_list,name="unknown",hermitian=False):
        if type(formula_list) not in (list,tuple):
            formula_list=[formula_list]
        self.TRodd=bool(sum(f.TRodd for f in formula_list)%2)
        self.Iodd =bool(sum(f.Iodd for f in formula_list)%2)
        self.name=name
        self.formulae=formula_list
        self.hermitian = hermitian
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
        for mat,line  in zip(matrices[1:],self.einsumlines):
            res=np.einsum(line,res,mat)
        if self.hermitian:
            res=0.5*(res+res.swapaxes(0,1).conj())
        return np.array(res,dtype=complex)

    def ln(self,ik,inn,out):
        raise NotImplementedError()

class FormulaProduct_2(Formula_ln):
    """a class to store a product of several formulae (when have same index)"""
    def __init__(self,formula_list,index_list,name="unknown",hermitian=False,dot = False):
        if type(formula_list) not in (list,tuple):
            formula_list=[formula_list]
        self.TRodd=bool(sum(f.TRodd for f in formula_list)%2)
        self.Iodd =bool(sum(f.Iodd for f in formula_list)%2)
        self.name=name
        self.formulae=formula_list
        self.index=index_list
        self.hermitian = hermitian
        self.einsumlines=[]
        if dot:
            self.einsumlines.append( "LM"+self.index[0]+",MN"+self.index[1]+"->LN" )
            self.ndim = 0
        else:
            save_index = self.index[0]
            for i in range(len(self.index)-1):
                result_index = "".join(dict.fromkeys(save_index+self.index[i+1]))
                self.einsumlines.append( "LM"+save_index+",MN"+self.index[i+1]+"->LN"+result_index )
                save_index = result_index
            self.ndim = len(save_index)
        matrices = [frml.nn(1,[1],[1]) for frml in self.formulae ]
    def nn(self,ik,inn,out):
        matrices = [frml.nn(ik,inn,out) for frml in self.formulae ]
        res=matrices[0]
        for mat,line  in zip(matrices[1:],self.einsumlines):
            res=np.einsum(line,res,mat)
        if self.hermitian:
            res=0.5*(res+res.swapaxes(0,1).conj())
        return np.array(res,dtype=complex)

    def ln(self,ik,inn,out):
        raise NotImplementedError()

class ProductDelta(Formula_ln):
    """ """
    def __init__(self,matrix,index_list):
        self.delta = np.eye(3)
        self.TRodd=matrix.TRodd
        self.Iodd=matrix.Iodd
        self.matrix = matrix
        self.index = index_list
        self.ndim = len("".join(dict.fromkeys(self.index[0]+self.index[1])))

    def nn(self,ik,inn,out):
        matrix = self.matrix.nn(ik,inn,out)
        res = np.einsum(""+self.index[0]+",ML"+self.index[1]+"->ML"+"".join(dict.fromkeys(self.index[0]+self.index[1])),self.delta,matrix)
        return np.array(res,dtype=complex)
    
    def ln(self,ik,inn,out):
        matrix = self.matrix.ln(ik,inn,out)
        res = np.einsum(""+self.index[0]+",ML"+self.index[1]+"->ML"+"".join(dict.fromkeys(self.index[0]+self.index[1])),self.delta,matrix)
        return np.array(res,dtype=complex)
