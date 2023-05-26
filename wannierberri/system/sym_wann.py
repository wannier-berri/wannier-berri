import numpy as np
import spglib
from .sym_wann_orbitals import Orbitals
from irrep.spacegroup import SymmetryOperation
from collections import defaultdict
import lazy_property
from time import time

class SymWann():

    default_parameters = {
            'soc':False,
            'magmom':None,
            'DFT_code':'qe'}

    __doc__ = """
    Symmetrize wannier matrices in real space: Ham_R, AA_R, BB_R, SS_R,...

    Parameters
    ----------
    num_wann: int
        Number of wannier functions.
    lattice: array
        Unit cell lattice constant.
    positions: array
        Positions of each atom.
    atom_name: list
        Name of each atom.
    proj: list
        Should be the same with projections card in relative Wannier90.win.

        eg: ``['Te: s','Te:p']``

        If there is hybrid orbital, grouping the other orbitals.

        eg: ``['Fe':sp3d2;t2g]`` Plese don't use ``['Fe':sp3d2;dxz,dyz,dxy]``

            ``['X':sp;p2]`` Plese don't use ``['X':sp;pz,py]``
    iRvec: array
        List of R vectors.
    XX_R: dic
        Matrix before symmetrization.
    soc: bool
        Spin orbital coupling. Default: ``{soc}``
    magmom: 2D array
        Magnetic momentom of each atoms. Default ``{magmom}``
    DFT_code: str
        ``'qe'`` or ``'vasp'``   Default: ``{DFT_code}``
        vasp and qe have different orbitals arrangement with SOC.

    Return
    ------
    Dictionary of matrix after symmetrization.
    Updated list of R vectors.

    """.format(**default_parameters)

    def __init__(
            self,
            positions,
            atom_name,
            projections,
            num_wann,
            lattice,
            iRvec,
            XX_R,
            **parameters):

        for param in self.default_parameters:
            if param in parameters:
                vars(self)[param] = parameters[param]
            else:
                vars(self)[param] = self.default_parameters[param]
        self.iRvec = [tuple(R) for R in iRvec]
        self.iRvec_index = {r:i for i,r in enumerate(self.iRvec)}
        self.nRvec = len(self.iRvec)
        self.num_wann = num_wann
        self.lattice = lattice
        self.positions = positions
        self.atom_name = atom_name
        self.possible_matrix_list = ['Ham','AA', 'SS', 'BB', 'CC']  #['AA','BB','CC','SS','SA','SHA','SR','SH','SHR']
        # This is confusing, actually the I-odd vectors have "+1" here, because the minus is already in the rotation matrix
        # but Ham is a scalar, so +1
        #TODO: change it
        self.parity_I = {
            'Ham':1,
            'AA': 1,
            'BB': 1,
            'CC': -1,
            'SS': -1
        }  #{'AA':1,'BB':1,'CC':1,'SS':-1,'SA':1,'SHA':1,'SR':1,'SH':1,'SHR':1}
        self.parity_TR = {
            'Ham':1,
            'AA': 1,
            'BB': 1,
            'CC': -1,
            'SS': -1
        }  #{'AA':1,'BB':1,'CC':1,'SS':-1,'SA':1,'SHA':1,'SR':1,'SH':1,'SHR':1}
        self.ndimv = {
            'Ham':0,
            'AA':1, 'BB':1, 'CC':1,'SS':1
                    }
        self.orbitals = Orbitals()

        num_atom = len(self.atom_name)

        #=============================================================
        #Generate wannier_atoms_information list and H_select matrices
        #=============================================================
        '''
        Wannier_atoms_information is a list of informations about atoms which contribute projections orbitals.
        Form: (number [int], name_of_element [str],position [array], orbital_index [list] ,
                starting_orbital_index_of_each_orbital_quantum_number [list],
                ending_orbital_index_of_each_orbital_quantum_number [list]  )
        Eg: (1, 'Te', array([0.274, 0.274, 0.   ]), 'sp', [0, 1, 6, 7, 8, 9, 10, 11], [0, 6], [2, 12])

        H_select matrices is bool matrix which can select a subspace of Hamiltonian between one atom and it's
        equivalent atom after symmetry operation.
        '''
        times = []
        times.append(time())
        proj_dic = defaultdict(lambda : [])
        orbital_index = 0
        orbital_index_list = [[] for i in range(num_atom)]
        for proj in projections:
            name_str = proj.split(":")[0].split()[0]
            orb_str = proj.split(":")[1].strip('\n').strip().split(';')
            proj_dic[name_str] += orb_str
            for iatom,atom_name in enumerate(self.atom_name):
                if atom_name == name_str:
                    for iorb in orb_str:
                        num_orb = self.orbitals.num_orbitals(iorb)
                        orb_list = [orbital_index + i for i in range(num_orb)]
                        if self.soc:
                            orb_list += [i + self.num_wann // 2 for i in orb_list]
                        orbital_index += num_orb
                        orbital_index_list[iatom].append(orb_list)
        times.append(time())

        self.wann_atom_info = []
        for atom,name in enumerate(self.atom_name):
            if name in proj_dic:
                projection = proj_dic[name]
                self.wann_atom_info.append( WannAtomInfo(iatom=atom+1,  atom_name=self.atom_name[atom],
                        position=self.positions[atom], projection=projection, orbital_index=orbital_index_list[atom], soc=self.soc,
                        magmom=self.magmom[atom] if self.magmom is not None else None) )
        self.num_wann_atom = len (self.wann_atom_info)
        times.append(time())

        self.H_select = np.zeros((self.num_wann_atom, self.num_wann_atom, self.num_wann, self.num_wann), dtype=bool)
        for a,atom_a in enumerate(self.wann_atom_info):
            orb_list_a = atom_a.orbital_index
            for b,atom_b in enumerate(self.wann_atom_info):
                orb_list_b = atom_b.orbital_index
                for oa_list in orb_list_a:
                    for oia in oa_list:
                        for ob_list in orb_list_b:
                            for oib in ob_list:
                                self.H_select[a, b, oia, oib] = True
        times.append(time())

        self.matrix_dict_list = {}
        for k,v in XX_R.items():
            self.spin_reorder(v)
            self.matrix_dict_list[k] = _matrix_to_dict(v, self.H_select, self.wann_atom_info)
            if k not in self.possible_matrix_list:
                raise ValueError(f" symmetrization of matrix {k} is not implemented yet")#, so it will not be symmetrized, but passed as it. Use on your own risk")
        times.append(time())

        print('Wannier atoms info')
        for item in self.wann_atom_info:
            print(item)
        times.append(time())

        numbers = []
        names = list(set(self.atom_name))
        for name in self.atom_name:
            numbers.append(names.index(name) + 1)
        cell = (self.lattice, self.positions, numbers)
        print("[get_spacegroup]")
        print("  Spacegroup is %s." % spglib.get_spacegroup(cell))
        dataset = spglib.get_symmetry_dataset(cell)
        all_symmetry_operations = [
                SymmetryOperation_loc(rot, dataset['translations'][i], cell[0], ind=i + 1, spinor=self.soc)
                for i,rot in enumerate(dataset['rotations'])
                                   ]
        times.append(time())
        self.nrot=0
        self.symmetry_operations = []
        for symop in all_symmetry_operations:
            symop.rot_map, symop.vec_shift, symop.sym_only, symop.sym_T = self.atom_rot_map(symop)
            if symop.sym_T or symop.sym_only:
                self.symmetry_operations.append(symop)
                if symop.sym_only: self.nrot += 1
                if symop.sym_T: self.nrot += 1
                symop.p_mat_atom = []
                symop.p_mat_atom_dagger = []
                for atom in range(self.num_wann_atom):
                    p_mat_, p_mat_dagger_ = self.atom_p_mat(self.wann_atom_info[atom], symop)
                    symop.p_mat_atom.append(p_mat_)
                    symop.p_mat_atom_dagger.append(p_mat_dagger_)
                if symop.sym_T:
                    symop.p_mat_atom_T = []
                    symop.p_mat_atom_dagger_T = []
                    for atom in range(self.num_wann_atom):
                        ul = self.wann_atom_info[atom].ul
                        ur = self.wann_atom_info[atom].ur
                        symop.p_mat_atom_T.append(symop.p_mat_atom[atom].dot(ur))
                        symop.p_mat_atom_dagger_T.append(ul.dot(symop.p_mat_atom_dagger[atom]))
        times.append(time())

        self.nsymm = len(self.symmetry_operations)
        has_inv = np.any([(s.inversion and s.angle==0) for s in self.symmetry_operations])  # has inversion or not
        if has_inv:
            print('====================\nSystem has inversion symmetry\n====================')

        self.show_symmetry()
        times.append(time())
        print ("Timing of SymWann.__init__():", [t1-t0 for t0,t1 in zip(times,times[1:])], "total:",times[-1]-times[0])


#        self.find_irreducible_Rab()

    #==============================
    #Find space group and symmetres
    #==============================
    def show_symmetry(self):
        for i, symop  in enumerate(self.symmetry_operations):
            rot = symop.rotation
            trans = symop.translation
            rot_cart = symop.rotation_cart
            trans_cart = symop.translation_cart
            det = symop.det_cart
            print("  --------------- %4d ---------------" % (i + 1))
            print(" det = ", det)
            print("  rotation:                    cart:")
            for x in range(3):
                print(
                    "     [%2d %2d %2d]                    [%3.2f %3.2f %3.2f]" %
                    (rot[x, 0], rot[x, 1], rot[x, 2], rot_cart[x, 0], rot_cart[x, 1], rot_cart[x, 2]))
            print("  translation:")
            print(
                "     (%8.5f %8.5f %8.5f)  (%8.5f %8.5f %8.5f)" %
                (trans[0], trans[1], trans[2], trans_cart[0], trans_cart[1], trans_cart[2]))



    def atom_rot_map(self, symop):
        '''
        rot_map: A map to show which atom is the equivalent atom after rotation operation.
        vec_shift_map: Change of R vector after rotation operation.
        '''
        wann_atom_positions = [self.wann_atom_info[i].position for i in range(self.num_wann_atom)]
        rot_map = []
        vec_shift_map = []
        for atomran in range(self.num_wann_atom):
            atom_position = np.array(wann_atom_positions[atomran])
            new_atom = np.dot(symop.rotation, atom_position) + symop.translation
            for atom_index in range(self.num_wann_atom):
                old_atom = np.array(wann_atom_positions[atom_index])
                diff = np.array(new_atom - old_atom)
                if np.all(abs((diff + 0.5) % 1 - 0.5) < 1e-5):
                    match_index = atom_index
                    vec_shift = np.array(
                        np.round(new_atom - np.array(wann_atom_positions[match_index]), decimals=2), dtype=int)
                else:
                    if atom_index == self.num_wann_atom - 1:
                        assert atom_index != 0, (
                            f'Error!!!!: no atom can match the new atom after symmetry operation {symop.ind},\n'
                            + f'Before operation: atom {atomran} = {atom_position},\n'
                            + f'After operation: {atom_position},\nAll wann_atom: {wann_atom_positions}')
            rot_map.append(match_index)
            vec_shift_map.append(vec_shift)
        #Check if the symmetry operator respect magnetic moment.
        #TODO opt magnet code
        if self.soc:
            sym_only = True
            sym_T = True
            if self.magmom is not None:
                for i in range(self.num_wann_atom):
                    if sym_only or sym_T:
                        magmom = self.wann_atom_info[i].magmom
                        new_magmom = np.dot(symop.rotation_cart , magmom)*(-1 if symop.inversion else 1)
                        if abs(np.linalg.norm(magmom - new_magmom)) > 0.0005:
                            sym_only = False
                        if abs(np.linalg.norm(magmom + new_magmom)) > 0.0005:
                            sym_T = False
                if sym_only:
                    print('Symmetry operator {} respects magnetic moment'.format(symop.ind))
                if sym_T:
                    print('Symmetry operator {}*T respects magnetic moment'.format(symop.ind))
        else:
            sym_only = True
            sym_T = False
        return np.array(rot_map, dtype=int), np.array(vec_shift_map, dtype=int), sym_only, sym_T


    def atom_p_mat(self, atom_info, symop):
        '''
        Combining rotation matrix of Hamiltonian per orbital_quantum_number into per atom.  (num_wann,num_wann)
        '''
        orbitals = atom_info.projection
        orb_position_dic = atom_info.orb_position_on_atom_dic
        num_wann_on_atom =atom_info.num_wann
        p_mat = np.zeros((num_wann_on_atom, num_wann_on_atom), dtype=complex)
        p_mat_dagger = np.zeros_like(p_mat)
        for orb_name in orbitals:
            rot_orbital = self.orbitals.rot_orb(orb_name, symop.rotation_cart)
            if self.soc:
                rot_orbital = np.kron(symop.spinor_rotation, rot_orbital)
            orb_position = orb_position_dic[orb_name]
            p_mat[orb_position] = rot_orbital.flatten()
            p_mat_dagger[orb_position] = np.conj(np.transpose(rot_orbital)).flatten()
        return p_mat, p_mat_dagger


    def index_R(self,R):
        try:
            return self.iRvec_index[tuple(R)]
        except KeyError:
            return -1

    def find_irreducible_Rab(self):
        print ("searching irreducible Rvectors for pairs of a,b")

        R_list = np.array(self.iRvec, dtype=int)
        irreducible = np.ones((self.nRvec,self.num_wann_atom,self.num_wann_atom),dtype=bool)
        t0=time()
        for symop in self.symmetry_operations:
            if symop.sym_only or symop.sym_T:
                print('symmetry operation  ', symop.ind)
                R_map = np.dot(R_list, np.transpose(symop.rotation))
                atom_R_map = R_map[:, None, None, :] - symop.vec_shift[None, :, None, :] + symop.vec_shift[None, None, :, :]
                for a in range(self.num_wann_atom):
                    a1 = symop.rot_map[a]
                    for b in range(self.num_wann_atom):
                        b1 = symop.rot_map[b]
                        for iR in range(self.nRvec):
                            if irreducible[iR,a,b]:
                                iR1 = self.index_R(atom_R_map[iR, a, b])
                                if iR1>=0 and not (a,b,iR) == (a1,b1,iR1):
                                        irreducible[iR1,a1,b1]=False
        print (f"Found {np.sum(irreducible)} sets of (R,a,b) out of the total {self.nRvec*self.num_wann_atom**2} ({self.nRvec}*{self.num_wann_atom}^2)")
        t1=time()
        dic =  {(a,b):set([iR for iR in range(self.nRvec)  if irreducible[iR,a,b]])
                for a in range(self.num_wann_atom)  for b in range(self.num_wann_atom)}
        t2=time()
        res =  {k:v for k,v in dic.items() if len(v)>0}
        t3=time()
#        print ("timing search irred",t1-t0,t2-t1,t3-t2)
        return res


    def average_H_irreducible(self, iRab_new, matrix_dict_in, iRvec_new, mode):
        assert mode in ["sum","single"]
        iRvec_new_array = np.array(iRvec_new, dtype=int)

        matrix_dict_list_res = {k:defaultdict( lambda:defaultdict(lambda: 0))  for k,v in self.matrix_dict_list.items() }

        iRab_all = defaultdict( lambda:set() )

        for symop in self.symmetry_operations:
            if symop.sym_only or symop.sym_T:
                print('symmetry operation  ', symop.ind)

                R_map = np.dot(iRvec_new_array, np.transpose(symop.rotation))
                atom_R_map = R_map[:, None, None, :] - symop.vec_shift[None, :, None, :] + symop.vec_shift[None, None, :, :]
                iR0_old = self.index_R((0, 0, 0))

                #TODO try numba
                for (atom_a,atom_b),iR_new_list in iRab_new.items():
                    exclude_set = set()
                    for iR in iR_new_list:
                        new_Rvec = tuple(atom_R_map[iR, atom_a, atom_b])
                        iRab_all[(symop.rot_map[atom_a], symop.rot_map[atom_b])].add(new_Rvec)
                        new_Rvec_index = self.index_R(new_Rvec)
                        if new_Rvec_index>=0 :
                            '''
                            H_ab_sym = P_dagger_a dot H_ab dot P_b
                            H_ab_sym_T = ul dot H_ab_sym.conj() dot ur
                            '''
                            for X in matrix_dict_list_res:
                                if new_Rvec_index in matrix_dict_in[X][(symop.rot_map[atom_a], symop.rot_map[atom_b])]:
                                    if mode == "single":
                                        exclude_set.add(iR)
                                    #X_L: only rotation wannier centres from L to L' before rotating orbitals.
                                    XX_L = matrix_dict_in[X][(symop.rot_map[atom_a], symop.rot_map[atom_b])][
                                                               new_Rvec_index]
                                    #special even with R == [0,0,0] diagonal terms.
                                    if iR == iR0_old and atom_a == atom_b:
                                        if X in ['AA','BB']:
                                            v_tmp = (symop.vec_shift[atom_a] - symop.translation).dot(self.lattice)
                                            m_tmp = np.zeros_like(XX_L)
                                            for i in range(self.wann_atom_info[atom_a].num_wann):
                                                m_tmp[i,i,:]=v_tmp
                                        if X == 'AA':
                                            XX_L += m_tmp
                                        elif X == 'BB':
                                            XX_L += (m_tmp
                                                *self.matrix_dict_list['Ham'][(symop.rot_map[atom_a], symop.rot_map[atom_b])][
                                                    new_Rvec_index][:,:,None] )
                                    if XX_L.ndim==3:
                                        #X_all: rotating vector.
                                        XX_L = np.tensordot( XX_L, symop.rotation_cart, axes=1).reshape( XX_L.shape )
                                    elif XX_L.ndim>3:
                                        raise ValueError("transformation of tensors is not implemented")
                                    if symop.inversion:
                                        XX_L*=self.parity_I[X]
                                    if symop.sym_only:
                                        matrix_dict_list_res[X][(atom_a,atom_b)][iR] += _rotate_matrix(XX_L,symop.p_mat_atom_dagger[atom_a], symop.p_mat_atom[atom_b])
                                    if symop.sym_T:
                                        matrix_dict_list_res[X][(atom_a,atom_b)][iR] += _rotate_matrix(XX_L,symop.p_mat_atom_dagger_T[atom_a], symop.p_mat_atom_T[atom_b]).conj() * self.parity_TR[X]
                    # in single mode we need to determine it only once
                    if mode == "single":
                        iR_new_list-=exclude_set

        if mode == "single":
            for (atom_a,atom_b),iR_new_list in iRab_new.items():
                assert len(iR_new_list)==0, f"for atoms ({atom_a},{atom_b}) some R vectors were not set : {iR_new_list}"


        if mode == "sum":
            for x in matrix_dict_list_res.values():
                for d in x.values():
                    for k,v in d.items():
                        v /= self.nrot
        print('number of symmetry oprations == ', self.nrot)
        return matrix_dict_list_res, iRab_all


    def symmetrize(self):

        #========================================================
        #symmetrize existing R vectors and find additional R vectors
        #========================================================
        print('##########################')
        print('Symmetrizing Started')
        t0=time()
        iRab_irred = self.find_irreducible_Rab()
        t1=time()
        matrix_dict_list_res, iRvec_ab_all = self.average_H_irreducible( iRab_new=iRab_irred, matrix_dict_in=self.matrix_dict_list, iRvec_new=self.iRvec, mode="sum")
        t2=time()
        iRvec_new_set = set.union(*iRvec_ab_all.values())
        iRvec_new_set.add( (0,0,0) )
        iRvec_new = list(iRvec_new_set)
        nRvec_new = len(iRvec_new)
        t2a=time()
        iRvec_new_index = {r:i for i,r in enumerate(iRvec_new)}
        iRab_new = {k:set([iRvec_new_index[irvec] for irvec in v]) for k,v in iRvec_ab_all.items()}
        t3=time()
        matrix_dict_list_res, iRab_all_2 = self.average_H_irreducible( iRab_new=iRab_new, matrix_dict_in=matrix_dict_list_res, iRvec_new=iRvec_new, mode="single")
        t4=time()

        return_dic = {}
        for k,v in matrix_dict_list_res.items():
            return_dic[k] = _dict_to_matrix(v, H_select=self.H_select, wann_atom_info=self.wann_atom_info, nRvec=nRvec_new, ndimv=self.ndimv[k] )
            self.spin_reorder(return_dic[k], back=True)
        t5=time()

        print('Symmetrizing Finished')
        print ("Timing:",t1-t0,t2-t1,t2a-t2,t3-t2a,t4-t3,t5-t4,"total:",t5-t0)

        return return_dic, np.array(iRvec_new)

    def spin_reorder(self,Mat_in,back=False):
        """ rearranges the spins of the Wannier functions
            back=False : from interlacing spins to spin blocks
            back=True : from spin blocks to interlacing spins
        """
        if not self.soc:
            return
        elif self.DFT_code.lower() == 'vasp':
            return
        elif self.DFT_code.lower() in ['qe', 'quantum_espresso', 'espresso']:
            Mat_out = np.zeros(np.shape(Mat_in), dtype=complex)
            nw2 = self.num_wann // 2
            for i in 0,1:
                for j in 0,1:
                    if back:
                        Mat_out[i:self.num_wann:2,j:self.num_wann:2, ...] = Mat_in[i*nw2:(i+1)*nw2, j*nw2:(j+1)*nw2,...]
                    else:
                        Mat_out[i*nw2:(i+1)*nw2, j*nw2:(j+1)*nw2,...] = Mat_in[i:self.num_wann:2,j:self.num_wann:2, ...]
            Mat_in[...]=Mat_out[...]
            return
        else :
            raise ValueError(f"does not work for DFT_code  '{self.DFT_code}' so far")

class WannAtomInfo():

    def __init__(self,   iatom, atom_name, position, projection, orbital_index,  magmom=None, soc=False):
        self.iatom = iatom
        self.atom_name = atom_name
        self.position = position
        self.projection = projection
        self.orbital_index = orbital_index
#        self.orb_position_dic = orb_position_dic
        self.magmom = magmom
        self.soc = soc
        self.num_wann = len(sum(self.orbital_index, []))  #number of orbitals of atom_a
        allindex = sorted(sum(self.orbital_index ,[]))
        print ("allindex",allindex)
        self.orb_position_on_atom_dic = {}
        for pr,ind in zip(projection,orbital_index):
            indx = [allindex.index(i) for i in ind]
            print (pr,":",ind,":",indx)
            orb_select = np.zeros((self.num_wann, self.num_wann), dtype=bool)
            for oi in indx:
                for oj in indx:
                    orb_select[oi, oj] = True
            self.orb_position_on_atom_dic[pr]=orb_select

        #====Time Reversal====
        #syl: (sigma_y)^T *1j, syr: sigma_y*1j
        if self.soc:
            base_m = np.eye(self.num_wann // 2)
            syl = np.array([[0.0, -1.0], [1.0, 0.0]])
            syr = np.array([[0.0, 1.0], [-1.0, 0.0]])
            self.ul = np.kron(syl, base_m)
            self.ur = np.kron(syr, base_m)


    def __str__(self):
        return "; ".join(f"{key}:{value}" for key,value in self.__dict__.items() if key not in ["orb_position_on_atom_dic","ul","ur"] )


# TODO : move to irrep?
class SymmetryOperation_loc(SymmetryOperation):

    @lazy_property.LazyProperty
    def rotation_cart(self):
        return np.dot(np.dot(self._lattice_T, self.rotation), self._lattice_inv_T)

    @lazy_property.LazyProperty
    def translation_cart(self):
        return np.dot(np.dot(self._lattice_T, self.translation), self._lattice_inv_T)

    @lazy_property.LazyProperty
    def det_cart(self):
        return np.linalg.det(self.rotation_cart)

    @lazy_property.LazyProperty
    def det(self):
        return np.linalg.det(self.rotation)

    @lazy_property.LazyProperty
    def _lattice_inv_T(self):
        return np.linalg.inv(np.transpose(self.Lattice))

    @lazy_property.LazyProperty
    def _lattice_T(self):
        return np.transpose(self.Lattice)

def _rotate_matrix(X,L,R):
    if X.ndim==2:
        return L.dot(X).dot(R)
    elif X.ndim==3:
        X_shift = X.transpose(2, 0, 1)
        tmpX = L.dot(X_shift).dot(R)
        return tmpX.transpose( 0, 2, 1).reshape(X.shape)
    else:
        raise ValueError()

def _matrix_to_dict( mat, H_select, wann_atom_info):
    """transforms a matrix X[m,n,iR,...] into a dictionary like
        {(a,b): {iR: np.array(num_w_a.num_w_b,...)}}
    """
    result = {}
    for a,atom_a in enumerate(wann_atom_info):
        num_w_a = atom_a.num_wann  #number of orbitals of atom_a
        for b,atom_b in enumerate(wann_atom_info):
            num_w_b = atom_b.num_wann  #number of orbitals of atom_a
            result_ab = {}
            X = mat[ H_select[a, b] ]
            X = X.reshape( (num_w_a,num_w_b)+mat.shape[2:] )
            for iR in range(mat.shape[2]):
                x = X[:,:,iR]
                if np.max(abs(x))>1e-15:
                    result_ab[iR]=x
            if len(result_ab)>0:
                result[(a,b)] = result_ab
    return result


def _dict_to_matrix(dic,H_select,wann_atom_info,nRvec,ndimv):
    num_wann = H_select.shape[2]
    mat = np.zeros( (num_wann,num_wann,nRvec)+(3,)*ndimv, dtype=complex )
    for (a,b), irX in dic.items():
        for iR,X in irX.items():
            mat[H_select[a,b],iR] = X.reshape( (-1,)+X.shape[2:])
    return mat
