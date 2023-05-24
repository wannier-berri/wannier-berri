import numpy as np
import spglib
from .sym_wann_orbitals import Orbitals
from ..__utility import get_angle
from irrep.spacegroup import SymmetryOperation

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
            proj,
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
        self.Ham_R = XX_R['Ham']
        self.iRvec = iRvec.tolist()
        self.nRvec = len(iRvec)
        self.num_wann = num_wann
        self.lattice = lattice
        self.positions = positions
        self.atom_name = atom_name
        self.proj = proj
        self.possible_matrix_list = ['AA', 'SS', 'BB', 'CC']  #['AA','BB','CC','SS','SA','SHA','SR','SH','SHR']
        self.matrix_list = {k: v for k, v in XX_R.items() if k in self.possible_matrix_list}
        self.parity_I = {
            'AA': 1,
            'BB': 1,
            'CC': -1,
            'SS': -1
        }  #{'AA':1,'BB':1,'CC':1,'SS':-1,'SA':1,'SHA':1,'SR':1,'SH':1,'SHR':1}
        self.parity_TR = {
            'AA': 1,
            'BB': 1,
            'CC': -1,
            'SS': -1
        }  #{'AA':1,'BB':1,'CC':1,'SS':-1,'SA':1,'SHA':1,'SR':1,'SH':1,'SHR':1}
        self.orbitals = Orbitals()

        self.wann_atom_info = []

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
        proj_dic = {}
        orbital_index = 0
        orbital_index_list = []
        for i in range(num_atom):
            orbital_index_list.append([])
        for iproj in self.proj:
            name_str = iproj.split(":")[0].split()[0]
            orb_str = iproj.split(":")[1].strip('\n').strip().split(';')
            if name_str in proj_dic:
                proj_dic[name_str] = proj_dic[name_str] + orb_str
            else:
                proj_dic[name_str] = orb_str
            for iatom in range(num_atom):
                if self.atom_name[iatom] == name_str:
                    for iorb in orb_str:
                        num_orb = self.orbitals.num_orbitals(iorb)
                        orb_list = [orbital_index + i for i in range(num_orb)]
                        if self.soc:
                            orb_list += [orbital_index + i + int(self.num_wann / 2) for i in range(num_orb)]
                        orbital_index += num_orb
                        orbital_index_list[iatom].append(orb_list)

        self.wann_atom_info = []
        for atom in range(num_atom):
            name = self.atom_name[atom]
            if name in proj_dic:
                projection = proj_dic[name]
                orb_position_dic = {}
                for i in range(len(projection)):
                    orb_select = np.zeros((self.num_wann, self.num_wann), dtype=bool)
                    for oi in orbital_index_list[atom][i]:
                        for oj in orbital_index_list[atom][i]:
                            orb_select[oi, oj] = True
                    orb_position_dic[projection[i]] = orb_select
                self.wann_atom_info.append( WannAtomInfo(iatom=atom+1, atom_name=self.atom_name[atom], 
                        position=self.positions[atom], projection=projection, orbital_index=orbital_index_list[atom],
                        orb_position_dic=orb_position_dic, magmom=self.magmom[atom] if self.magmom is not None else None) )
        self.num_wann_atom = len (self.wann_atom_info)

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

        print('Wannier atoms info')
        for item in self.wann_atom_info:
            print(item)

        numbers = []
        names = list(set(self.atom_name))
        for name in self.atom_name:
            numbers.append(names.index(name) + 1)
        cell = (self.lattice, self.positions, numbers)
        #print(cell)
        print("[get_spacegroup]")
        print("  Spacegroup is %s." % spglib.get_spacegroup(cell))
        dataset = spglib.get_symmetry_dataset(cell)
        self.symmetry_operations = [
            SymmetryOperation(rot, dataset['translations'][i], cell[0], ind=i + 1, spinor=self.soc) 
                for i,rot in enumerate(dataset['rotations']) 
                          ]
        self.nsymm = len(self.symmetry_operations)
        self.show_symmetry()
        has_inv = np.any([(s.inversion and s.angle==0) for s in   self.symmetry_operations])  # has inversion or not
        if has_inv:
            print('====================\nSystem has inversion symmetry\n====================')

        self.spin_reorder(self.Ham_R)
        for X in self.matrix_list:
            self.spin_reorder(self.matrix_list[X])

    #==============================
    #Find space group and symmetres
    #==============================
    def show_symmetry(self):
        rot_c = []
        for i, symop  in enumerate(self.symmetry_operations):
            rot = symop.rotation
            trans = symop.translation
            rot_cart = np.dot(np.dot(np.transpose(self.lattice), rot), np.linalg.inv(np.transpose(self.lattice)))
            trans_cart = np.dot(np.dot(np.transpose(self.lattice), trans), np.linalg.inv(np.transpose(self.lattice)))
            det = np.linalg.det(rot_cart)
            symop.rotation_cart = rot_cart
            symop.translation_cart = trans_cart
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
        return rot_c

    def Part_P(self, rot_sym_glb, orb_symbol):
        '''
        Rotation matrix of orbitals.

        Without SOC Part_P = rotation matrix of orbital
        With SOC Part_P = Kronecker product of rotation matrix of orbital and rotation matrix of spin
        '''
        if abs(np.dot(np.transpose(rot_sym_glb), rot_sym_glb) - np.eye(3)).max() > 1.0E-4:
            raise RuntimeError('rot_sym is not orthogomal \n {}'.format(rot_sym_glb))
        rmat = np.linalg.det(rot_sym_glb) * rot_sym_glb
        select = np.abs(rmat) < 0.01
        rmat[select] = 0.0
        select = rmat > 0.99
        rmat[select] = 1.0
        select = rmat < -0.99
        rmat[select] = -1.0
        rot_orbital = self.orbitals.rot_orb(orb_symbol, rot_sym_glb)
        if self.soc:
            if np.abs(rmat[2, 2]) < 1.0:
                beta = np.arccos(rmat[2, 2])
                cos_gamma = -rmat[2, 0] / np.sin(beta)
                sin_gamma = rmat[2, 1] / np.sin(beta)
                gamma = get_angle(sin_gamma, cos_gamma)
                cos_alpha = rmat[0, 2] / np.sin(beta)
                sin_alpha = rmat[1, 2] / np.sin(beta)
                alpha = get_angle(sin_alpha, cos_alpha)
            else:
                beta = 0.0
                if rmat[2, 2] == -1.: beta = np.pi
                gamma = 0.0
                alpha = np.arccos(rmat[1, 1])
                if rmat[0, 1] > 0.0: alpha = -1.0 * alpha
            # euler_angle = np.array([alpha, beta, gamma])
            dmat = np.zeros((2, 2), dtype=complex)
            dmat[0, 0] = np.exp(-(alpha + gamma) / 2.0 * 1j) * np.cos(beta / 2.0)
            dmat[0, 1] = -np.exp(-(alpha - gamma) / 2.0 * 1j) * np.sin(beta / 2.0)
            dmat[1, 0] = np.exp((alpha - gamma) / 2.0 * 1j) * np.sin(beta / 2.0)
            dmat[1, 1] = np.exp((alpha + gamma) / 2.0 * 1j) * np.cos(beta / 2.0)
            rot_orbital = np.kron(dmat, rot_orbital)
        return rot_orbital

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
        rot_sym_glb = symop.rotation_cart 
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
                    print('Symmetry operator {} respects magnetic moment'.format(symop.ind + 1))
                if sym_T:
                    print('Symmetry operator {}*T respects magnetic moment'.format(symop.ind + 1))
        else:
            sym_only = True
            sym_T = False

        return np.array(rot_map, dtype=int), np.array(vec_shift_map, dtype=int), sym_only, sym_T

    def full_p_mat(self, atom_index, symop):
        '''
        Combining rotation matrix of Hamiltonian per orbital_quantum_number into per atom.  (num_wann,num_wann)
        '''
        orbitals = self.wann_atom_info[atom_index].projection
        orb_position_dic = self.wann_atom_info[atom_index].orb_position_dic
        p_mat = np.zeros((self.num_wann, self.num_wann), dtype=complex)
        p_mat_dagger = np.zeros((self.num_wann, self.num_wann), dtype=complex)
        for orb_name in orbitals:
            tmp = self.Part_P(symop.rotation_cart, orb_name)
            orb_position = orb_position_dic[orb_name]
            p_mat[orb_position] = tmp.flatten()
            p_mat_dagger[orb_position] = np.conj(np.transpose(tmp)).flatten()
        return p_mat, p_mat_dagger

    def average_H(self, iRvec):
        #If we can make if faster, respectively is the better choice. Because XX_all matrix are supper large.(eat memory)
        nrot = 0
        R_list = np.array(iRvec, dtype=int)
        nRvec = len(R_list)
        tmp_R_list = []
        Ham_res = np.zeros((self.num_wann, self.num_wann, nRvec), dtype=complex)

        matrix_list_res = {
            k: np.zeros((self.num_wann, self.num_wann, nRvec, 3), dtype=complex)
            for k in self.matrix_list
        }
        # print (f"iRvec ({nRvec}):\n  {self.iRvec}")

        for irot,symop in enumerate(self.symmetry_operations):
            rot_map, vec_shift, sym_only, sym_T = self.atom_rot_map(symop)
            if sym_only + sym_T == 0:
                pass
            else:
                print('irot = ', irot + 1)
                if sym_only: nrot += 1
                if sym_T: nrot += 1
                p_map = np.zeros((self.num_wann_atom, self.num_wann, self.num_wann), dtype=complex)
                p_map_dagger = np.zeros((self.num_wann_atom, self.num_wann, self.num_wann), dtype=complex)
                for atom in range(self.num_wann_atom):
                    p_map[atom], p_map_dagger[atom] = self.full_p_mat(atom, symop)
                R_map = np.dot(R_list, np.transpose(symop.rotation))
                atom_R_map = R_map[:, None, None, :] - vec_shift[None, :, None, :] + vec_shift[None, None, :, :]
                Ham_all = np.zeros(
                    (nRvec, self.num_wann_atom, self.num_wann_atom, self.num_wann, self.num_wann), dtype=complex)
                matrix_list_all = {
                    X: np.zeros(
                        (nRvec, self.num_wann_atom, self.num_wann_atom, self.num_wann, self.num_wann, 3), dtype=complex)
                    for X in self.matrix_list
                }

                #TODO try numba
                for iR in range(nRvec):
                    for atom_a in range(self.num_wann_atom):
                        num_w_a = self.wann_atom_info[atom_a].num_wann  #number of orbitals of atom_a
                        for atom_b in range(self.num_wann_atom):
                            new_Rvec = list(atom_R_map[iR, atom_a, atom_b])
                            if new_Rvec in self.iRvec:
                                new_Rvec_index = self.iRvec.index(new_Rvec)
                                Ham_all[iR, atom_a, atom_b,
                                        self.H_select[atom_a, atom_b]] = self.Ham_R[self.H_select[rot_map[atom_a],
                                                                                                  rot_map[atom_b]],
                                                                                    new_Rvec_index]

                                for X in self.matrix_list:
                                    if X in ['AA', 'BB', 'SS', 'CC', 'FF']:
                                        num_w_a = self.wann_atom_info[atom_a].num_wann
                                        num_w_b = self.wann_atom_info[atom_b].num_wann
                                        #X_L: only rotation wannier centres from L to L' before rotating orbitals.
                                        XX_L = self.matrix_list[X][self.H_select[rot_map[atom_a], rot_map[atom_b]],
                                                                   new_Rvec_index, :].reshape(num_w_a, num_w_b, 3)
                                        #special even with R == [0,0,0] diagonal terms.
                                        if iR == self.iRvec.index([0, 0, 0]) and atom_a == atom_b:
                                            if X in ['AA','BB']:
                                                v_tmp = (vec_shift[atom_a] - symop.translation).dot(self.lattice)
                                                m_tmp = np.zeros_like(XX_L)
                                                for i in range(num_w_a):
                                                    m_tmp[i,i,:]=v_tmp
                                            if X == 'AA':
                                                XX_L += m_tmp
                                            elif X == 'BB':
                                                XX_L += (m_tmp
                                                    *self.Ham_R[self.H_select[rot_map[atom_a], rot_map[atom_b]],
                                                        new_Rvec_index].reshape(num_w_a, num_w_b)[:, :, None])
                                        #X_all: rotating vector.
                                        matrix_list_all[X][iR, atom_a, atom_b,
                                                           self.H_select[atom_a, atom_b], :] = np.tensordot(
                                                                XX_L, symop.rotation_cart, axes = 1 ).reshape(-1, 3)
                                    else:
                                        print(f"WARNING: Symmetrization of {X} is not implemented")
                            else:
                                if new_Rvec in tmp_R_list:
                                    pass
                                else:
                                    tmp_R_list.append(new_Rvec)

                for atom_a in range(self.num_wann_atom):
                    for atom_b in range(self.num_wann_atom):
                        '''
                        H_ab_sym = P_dagger_a dot H_ab dot P_b
                        H_ab_sym_T = ul dot H_ab_sym.conj() dot ur
                        '''
                        tmp = np.dot(np.dot(p_map_dagger[atom_a], Ham_all[:, atom_a, atom_b]), p_map[atom_b])
                        if sym_only:
                            Ham_res += tmp.transpose(0, 2, 1)

                        if sym_T:
                            tmp_T = self.ul.dot(tmp.transpose(1, 0, 2)).dot(self.ur).conj()
                            Ham_res += tmp_T.transpose(0, 2, 1)

                        for X in self.matrix_list:  # vector matrix
                            X_shift = matrix_list_all[X].transpose(0, 1, 2, 5, 3, 4)
                            tmpX = np.dot(np.dot(p_map_dagger[atom_a], X_shift[:, atom_a, atom_b]), p_map[atom_b])
                            if symop.inversion:
                                parity_I = self.parity_I[X]
                            else:
                                parity_I = 1
                            if sym_only:
                                matrix_list_res[X] += tmpX.transpose(0, 3, 1, 2) * parity_I
                            if sym_T:
                                tmpX_T = self.ul.dot(tmpX.transpose(1, 2, 0, 3)).dot(self.ur).conj()
                                matrix_list_res[X] += tmpX_T.transpose(0, 3, 1, 2) * parity_I * self.parity_TR[X]

        for k in matrix_list_res:
            matrix_list_res[k] /= nrot
        res_dic = matrix_list_res
        res_dic['Ham'] = Ham_res / nrot

        print('number of symmetry oprations == ', nrot)

        return res_dic, tmp_R_list

    def symmetrize(self):
        #====Time Reversal====
        #syl: (sigma_y)^T *1j, syr: sigma_y*1j
        if self.soc:
            base_m = np.eye(self.num_wann // 2)
            syl = np.array([[0.0, -1.0], [1.0, 0.0]])
            syr = np.array([[0.0, 1.0], [-1.0, 0.0]])
            self.ul = np.kron(syl, base_m)
            self.ur = np.kron(syr, base_m)

        #========================================================
        #symmetrize existing R vectors and find additional R vectors
        #========================================================
        print('##########################')
        print('Symmetrizing Start')
        return_dic, iRvec_add = self.average_H(self.iRvec)
        nRvec_add = len(iRvec_add)
        print('nRvec_add =', nRvec_add)
        if nRvec_add > 0:
            return_dic_add, iRvec_add_0 = self.average_H(iRvec_add)
            for X in return_dic_add.keys():
                return_dic[X] = np.concatenate((return_dic[X], return_dic_add[X]), axis=2)

        self.spin_reorder(return_dic['Ham'], back=True)
        for X in self.matrix_list:
            self.spin_reorder(return_dic[X], back=True)

        print('Symmetrizing Finished')

        return return_dic, np.array(self.iRvec + iRvec_add)


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

    def __init__(self,  iatom, atom_name, position, projection, orbital_index,  orb_position_dic, magmom=None):
        self.iatom = iatom
        self.atom_name = atom_name
        self.position = position
        self.projection = projection
        self.orbital_index = orbital_index
        self.orb_position_dic = orb_position_dic
        self.magmom = magmom
        self.num_wann = len(sum(self.orbital_index, []))  #number of orbitals of atom_a

    def __str__(self):
        return "; ".join(f"{key}:{value}" for key,value in self.__dict__.items() if key!="orb_position_dic" )