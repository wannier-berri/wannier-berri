import numpy as np
import spglib
import numpy.linalg as la
import sympy as sym

#np.set_printoptions(threshold=np.inf,linewidth=500)
np.set_printoptions(suppress=True,precision=4,threshold=np.inf,linewidth=500)


class sym_wann():
    '''
    Symmetrize wannier matrixes in real space: Ham_R, AA_R, BB_R, SS_R,...  
    
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
        eg: ['Te: s','Te:p']
        If there is hybrid orbital, grouping the other orbitals.
        eg: ['Fe':sp3d2;t2g] Plese don't use ['Fe':sp3d2;dxz,dyz,dxy]
            ['X':sp;p2] Plese don't use ['X':sp;pz,py]
    iRvec: array
        List of R vectors.
    XX_R: dic
        Matrix before symmetrization. {'Ham':self.Ham_R,'AA':self.AA_R,......}
    soc: bool
        Spin orbital coupling.
    magmom: 2D array
        Magnetic momentom of each atoms.
    Return
    ------
    Dictionary of matrix after symmetrization.
    Updated list of R vectors.
    '''
    def __init__(self,num_wann=None,lattice=None,positions=None,atom_name=None,proj=None,iRvec=None,
            XX_R=None,soc=False,TR=False,magmom=None):

        self.soc=soc
        self.TR=TR
        self.Ham_R =XX_R['Ham']
        self.iRvec = iRvec.tolist()
        self.nRvec = len(iRvec)
        self.num_wann = num_wann
        self.lattice = lattice
        self.positions = positions
        self.atom_name = atom_name
        self.proj = proj
        self.matrix_list = ['AA','SS','BB','CC']#['AA','BB','CC','SS','SA','SHA','SR','SH','SHR']
        self.parity_I =  {'AA':1,'BB':1,'CC':1,'SS':-1}#{'AA':1,'BB':1,'CC':1,'SS':-1,'SA':1,'SHA':1,'SR':1,'SH':1,'SHR':1} 
        self.parity_TR =  {'AA':1,'BB':1,'CC':1,'SS':-1}#{'AA':1,'BB':1,'CC':1,'SS':-1,'SA':1,'SHA':1,'SR':1,'SH':1,'SHR':1}
        self.matrix_bool = {}
        self.magmom=magmom
        for X in self.matrix_list:
            try: 
                vars(self)[X+'_R'] = XX_R[X]
                self.matrix_bool[X] = True
            except KeyError:
                self.matrix_bool[X] = False
        
        self.orbital_dic = {"s":1,"p":3,"d":5,"f":7,"sp3":4,"sp2":3,"pz":1,"sp":2,"p2":2,
                "sp3d":5,"sp3d2":6,'t2g':3, 'dx2-y2':1, 'eg':2}
        self.wann_atom_info = []

        num_atom = len(self.atom_name)

        #=============================================================
        #Generate wannier_atoms_information list and H_select matrixes
        #=============================================================
        '''
        Wannier_atoms_information is a list of informations about atoms which contribute projections orbitals. 
        Form: (number [int], name_of_element [str],position [array], orbital_index [list] , 
                starting_orbital_index_of_each_orbital_quantum_number [list],
                ending_orbital_index_of_each_orbital_quantum_number [list]  )
        Eg: (1, 'Te', array([0.274, 0.274, 0.   ]), 'sp', [0, 1, 6, 7, 8, 9, 10, 11], [0, 6], [2, 12])

        H_select matrixex is bool matrix which can select a subspace of Hamiltonian between one atom and it's 
        equivalent atom after symmetry operation.  
        '''
        proj_dic = {}
        orbital_index = 0
        orbital_index_list = []
        for i in range(num_atom):
            orbital_index_list.append([])
        for iproj in self.proj:
            name_str = iproj.split(":")[0].split()[0]
            print(name_str)
            orb_str = iproj.split(":")[1].strip('\n').strip().split(';')
            print(orb_str)
            if name_str in proj_dic:
                proj_dic[name_str]=proj_dic[name_str]+orb_str
            else:
                proj_dic[name_str]=orb_str
            for iatom in range(num_atom):
                if self.atom_name[iatom] == name_str:
                    for iorb in orb_str:
                        num_orb =  self.orbital_dic[iorb]
                        orb_list = [ orbital_index+i for i in range(num_orb)]
                        if self.soc:
                            orb_list += [ orbital_index+i+int(self.num_wann/2) for i in range(num_orb)]
                        orbital_index+= num_orb
                        orbital_index_list[iatom].append(orb_list) 

        self.wann_atom_info = []
        self.num_wann_atom = 0
        for atom in range(num_atom):
            name = self.atom_name[atom]
            if name in proj_dic:
                projection=proj_dic[name]	
                self.num_wann_atom +=1
                orb_position_dic={}
                for i in range(len(projection)):
                    orb_select=np.zeros((self.num_wann,self.num_wann),dtype=bool)
                    for oi in orbital_index_list[atom][i]: 
                        for oj in orbital_index_list[atom][i]:
                            orb_select[oi,oj] = True
                    orb_position_dic[projection[i]] = orb_select
                if self.magmom is None:
                    self.wann_atom_info.append((atom+1,self.atom_name[atom],self.positions[atom],projection,
                        orbital_index_list[atom],orb_position_dic))
                else:
                    self.wann_atom_info.append((atom+1,self.atom_name[atom],self.positions[atom],projection,
                        orbital_index_list[atom],self.magmom[atom],orb_position_dic))
        
        self.H_select=np.zeros((self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann),dtype=bool)
        for atom_a in range(self.num_wann_atom):
            for atom_b in range(self.num_wann_atom):
                orb_name_a = self.wann_atom_info[atom_a][3] #list of orbital type
                orb_name_b = self.wann_atom_info[atom_b][3] #...
                orb_list_a = self.wann_atom_info[atom_a][4] #list of orbital index
                orb_list_b = self.wann_atom_info[atom_b][4] #...
                for oa_list in orb_list_a:
                    for oia in oa_list:
                        for ob_list in orb_list_b:
                            for oib in ob_list:
                                self.H_select[atom_a,atom_b,oia,oib]=True
        
        print('Wannier atoms info')
        for item in self.wann_atom_info:
            print(item[:-1])
        
        #==============================
        #Find space group and symmetres
        #==============================
        def show_symmetry(symmetry):
            for i in range(symmetry['rotations'].shape[0]):
                rot = symmetry['rotations'][i]
                trans = symmetry['translations'][i]
                rot_cart = np.dot(np.dot(np.transpose(self.lattice),rot),np.linalg.inv(np.transpose(self.lattice)) )
                trans_cart = np.dot(np.dot(np.transpose(self.lattice),trans),np.linalg.inv(np.transpose(self.lattice)) )
                det = np.linalg.det(rot_cart)
                print("  --------------- %4d ---------------" % (i + 1))
                print(" det = ",det)
                print("  rotation:                    cart:")
                for x in range(3):
                    print("     [%2d %2d %2d]                    [%3.2f %3.2f %3.2f]" % (rot[x,0], rot[x,1], rot[x,2], 
                        rot_cart[x,0], rot_cart[x,1], rot_cart[x,2]))
                print("  translation:")
                print("     (%8.5f %8.5f %8.5f)  (%8.5f %8.5f %8.5f)" % (trans[0], trans[1], trans[2],
                    trans_cart[0], trans_cart[1], trans_cart[2]))
        numbers = []
        names = list(set(self.atom_name))
        for name in self.atom_name:
            numbers.append(names.index(name)+1)
        cell = (self.lattice,self.positions,numbers)
        #print(cell)
        print("[get_spacegroup]")
        print("  Spacegroup is %s." %spglib.get_spacegroup(cell))
        self.symmetry = spglib.get_symmetry_dataset(cell)
        self.nsymm = self.symmetry['rotations'].shape[0]
        show_symmetry(self.symmetry)
        self.Inv = (self.symmetry['rotations'][1] == -1*np.eye(3)).all() #inversion or not
        if self.Inv: print('====================\nSystem have inversion symmetry\n====================')

    def get_angle(self,sina,cosa):
        '''Get angle in radian from sin and cos.'''
        if abs(cosa) > 1.0:
            cosa = np.round(cosa,decimals=1)
        alpha = np.arccos(cosa)
        if sina < 0.0:
            alpha = 2.0 * np.pi - alpha
        return alpha

    def rot_orb(self,orb_symbol,rot_glb):
        ''' Get rotation matrix of orbitals in each orbital quantum number '''
        x = sym.Symbol('x')
        y = sym.Symbol('y')
        z = sym.Symbol('z')
        ss = lambda x,y,z : 1+0*x
        px = lambda x,y,z : x
        py = lambda x,y,z : y
        pz = lambda x,y,z : z
        dz2 = lambda x,y,z : (2*z*z-x*x-y*y)/(2*sym.sqrt(3.0))
        #dz2 = lambda x,y,z : (3*z*z-1)/sym.sqrt(3)/2
        dxz = lambda x,y,z : x*z
        dyz = lambda x,y,z : y*z
        dx2_y2 = lambda x,y,z : (x*x-y*y)/2
        dxy = lambda x,y,z : x*y
        fz3 = lambda x,y,z : z*(2*z*z-3*x*x-3*y*y)/(2*sym.sqrt(15.0))
        fxz2 = lambda x,y,z : x*(4*z*z-x*x-y*y)/(2*sym.sqrt(10.0))
        fyz2 = lambda x,y,z : y*(4*z*z-x*x-y*y)/(2*sym.sqrt(10.0))
        fzx2_zy2 = lambda x,y,z : z*(x*x-y*y)/2
        fxyz = lambda x,y,z : x*y*z
        fx3_3xy2 = lambda x,y,z : x*(x*x-3*y*y)/(2*sym.sqrt(6.0))
        f3yx2_y3 = lambda x,y,z : y*(3*x*x-y*y)/(2*sym.sqrt(6.0))
        
        sp_1 = lambda x,y,z: 1/sym.sqrt(2) * x
        sp_2 = lambda x,y,z: -1/sym.sqrt(2) * x

        sp2_1 = lambda x,y,z: - 1/sym.sqrt(6)*x + 1/sym.sqrt(2)*y
        sp2_2 = lambda x,y,z: - 1/sym.sqrt(6)*x - 1/sym.sqrt(2)*y
        sp2_3 = lambda x,y,z: 2/sym.sqrt(6)*x

        sp3_1 = lambda x,y,z : 0.5*(x + y + z)
        sp3_2 = lambda x,y,z : 0.5*(x - y - z)
        sp3_3 = lambda x,y,z : 0.5*(- x + y - z)
        sp3_4 = lambda x,y,z : 0.5*(- x - y + z)

        sp3d2_1 = lambda x,y,z: -1/sym.sqrt(2)*x 
        sp3d2_2 = lambda x,y,z: 1/sym.sqrt(2)*x 
        sp3d2_3 = lambda x,y,z: -1/sym.sqrt(2)*y 
        sp3d2_4 = lambda x,y,z: 1/sym.sqrt(2)*y 
        sp3d2_5 = lambda x,y,z: -1/sym.sqrt(2)*z 
        sp3d2_6 = lambda x,y,z: 1/sym.sqrt(2)*z 
        
        #sp3d2_plus_1 = lambda x,y,z: - 1/sym.sqrt(12)*(2*z*z-x*x-y*y)/sym.sqrt(3)/2 + 0.5*(x*x-y*y)/2
        #sp3d2_plus_2 = lambda x,y,z: - 1/sym.sqrt(12)*(2*z*z-x*x-y*y)/sym.sqrt(3)/2 + 0.5*(x*x-y*y)/2
        #sp3d2_plus_3 = lambda x,y,z: - 1/sym.sqrt(12)*(2*z*z-x*x-y*y)/sym.sqrt(3)/2 - 0.5*(x*x-y*y)/2
        #sp3d2_plus_4 = lambda x,y,z: - 1/sym.sqrt(12)*(2*z*z-x*x-y*y)/sym.sqrt(3)/2 - 0.5*(x*x-y*y)/2
        #sp3d2_plus_5 = lambda x,y,z: + 1/sym.sqrt(3)*(2*z*z-x*x-y*y)/sym.sqrt(3)/2
        #sp3d2_plus_6 = lambda x,y,z: + 1/sym.sqrt(3)*(2*z*z-x*x-y*y)/sym.sqrt(3)/2
        
        sp3d2_plus_1 = lambda x,y,z: - 1/sym.sqrt(12)*(3*z*z-1)/sym.sqrt(3)/2 + 0.5*(x*x-y*y)/2
        sp3d2_plus_2 = lambda x,y,z: - 1/sym.sqrt(12)*(3*z*z-1)/sym.sqrt(3)/2 + 0.5*(x*x-y*y)/2
        sp3d2_plus_3 = lambda x,y,z: - 1/sym.sqrt(12)*(3*z*z-1)/sym.sqrt(3)/2 - 0.5*(x*x-y*y)/2
        sp3d2_plus_4 = lambda x,y,z: - 1/sym.sqrt(12)*(3*z*z-1)/sym.sqrt(3)/2 - 0.5*(x*x-y*y)/2
        sp3d2_plus_5 = lambda x,y,z: + 1/sym.sqrt(3)*(3*z*z-1)/sym.sqrt(3)/2
        sp3d2_plus_6 = lambda x,y,z: + 1/sym.sqrt(3)*(3*z*z-1)/sym.sqrt(3)/2

        orb_function_dic={'s':[ss],
                          'p':[pz,px,py],
                          'd':[dz2,dxz,dyz,dx2_y2,dxy],
                          'f':[fz3,fxz2,fyz2,fzx2_zy2,fxyz,fx3_3xy2,f3yx2_y3],
                         'sp':[sp_1,sp_2],
                         'p2':[pz,py],
                        'sp2':[sp2_1,sp2_2,sp2_3],
                         'pz':[pz],
                        'sp3':[sp3_1,sp3_2,sp3_3,sp3_4],
                      'sp3d2':[sp3d2_1,sp3d2_2,sp3d2_3,sp3d2_4,sp3d2_5,sp3d2_6],
                 'sp3d2_plus':[sp3d2_plus_1,sp3d2_plus_2,sp3d2_plus_3,sp3d2_plus_4,sp3d2_plus_5,sp3d2_plus_6],
                        't2g':[dxz,dyz,dxy],
                         'eg':[dxz,dyz,dxy],
                      }
        orb_chara_dic={'s':[x],
                       'p':[z,x,y],
                       'd':[z*z,x*z,y*z,x*x,x*y,y*y],
                       'f':[z*z*z,x*z*z,y*z*z,z*x*x,x*y*z,x*x*x,y*y*y  ,z*y*y,x*y*y,y*x*x],
                      'sp':[x,y,z],
                      'p2':[z,y,x],
                     'sp2':[x,y,z],
                      'pz':[z,x,y],
                     'sp3':[x,y,z],
                   'sp3d2':[x,y,z],
              'sp3d2_plus':[z*z,x*x,y*y,x*y,x*z,y*z],
                     't2g':[x*z,y*z,x*y,z*z,x*x,y*y], 
                      'eg':[z*z,x*x,y*y,x*z,y*z,x*y], 
                }
        orb_dim = self.orbital_dic[orb_symbol]
        orb_rot_mat = np.zeros((orb_dim,orb_dim),dtype=float)
        xp = np.dot(np.linalg.inv(rot_glb)[0],np.transpose([x,y,z]))
        yp = np.dot(np.linalg.inv(rot_glb)[1],np.transpose([x,y,z]))
        zp = np.dot(np.linalg.inv(rot_glb)[2],np.transpose([x,y,z]))
        OC = orb_chara_dic[orb_symbol]
        OC_len = len(OC)
        if orb_symbol == 'sp3d2':
            OC_plus = orb_chara_dic[orb_symbol+'_plus']
            OC_plus_len = len(OC_plus)
        for i in range(orb_dim):
            subs = []
            equation = (orb_function_dic[orb_symbol][i](xp,yp,zp)).expand()
            for j in range(OC_len):
                for j_add in range(OC_len):
                    if j_add == 0:
                        eq_tmp = equation.subs(OC[j],1)
                    else:
                        eq_tmp = eq_tmp.subs(OC[(j+j_add)%OC_len],0)
                subs.append(eq_tmp)
            if orb_symbol in ['sp3d2']:
                subs_plus = []
                equation_plus = (orb_function_dic[orb_symbol+'_plus'][i](xp,yp,zp)).expand()
                for k in range(OC_plus_len):
                    for k_add in range(OC_plus_len):
                        if k_add == 0:
                            eq_tmp = equation_plus.subs(OC_plus[k],1)
                        else:
                            eq_tmp = eq_tmp.subs(OC_plus[(k+k_add)%OC_plus_len],0)
                    subs_plus.append(eq_tmp)
            
            if orb_symbol in ['s','pz']:
                orb_rot_mat[0,0] = subs[0].evalf()
            elif orb_symbol == 'p':
                orb_rot_mat[0,i] = subs[0].evalf()
                orb_rot_mat[1,i] = subs[1].evalf()
                orb_rot_mat[2,i] = subs[2].evalf()
            elif orb_symbol == 'd':
                orb_rot_mat[0,i] = (2*subs[0]-subs[3]-subs[5])/sym.sqrt(3.0)
                orb_rot_mat[1,i] = subs[1].evalf()
                orb_rot_mat[2,i] = subs[2].evalf()
                orb_rot_mat[3,i] =  (subs[3]-subs[5]).evalf()
                orb_rot_mat[4,i] = subs[4].evalf()
            elif orb_symbol == 'f':
                orb_rot_mat[0,i] = (subs[0]*sym.sqrt(15.0)).evalf()
                orb_rot_mat[1,i] = (subs[1]*sym.sqrt(10.0)/2).evalf()
                orb_rot_mat[2,i] = (subs[2]*sym.sqrt(10.0)/2).evalf()
                orb_rot_mat[3,i] = (2*subs[3]+3*subs[0]).evalf()
                orb_rot_mat[4,i] = subs[4].evalf()
                orb_rot_mat[5,i] = ((2*subs[5]+subs[1]/2)*sym.sqrt(6.0)).evalf()
                orb_rot_mat[6,i] = ((-2*subs[6]-subs[2]/2)*sym.sqrt(6.0)).evalf()
            elif orb_symbol == 'sp':
                orb_rot_mat[0,i] = 1/2-1/sym.sqrt(2)*subs[0]
                orb_rot_mat[1,i] = 1/2-1/sym.sqrt(2)*subs[0]
            elif orb_symbol == 'p2':
                orb_rot_mat[0,i] = subs[0]
                orb_rot_mat[1,i] = subs[1]
            elif orb_symbol == 'sp2':
                orb_rot_mat[0,i] = 1/3-1/sym.sqrt(6)*subs[0] +1/sym.sqrt(2)*subs[1]
                orb_rot_mat[1,i] = 1/3-1/sym.sqrt(6)*subs[0] -1/sym.sqrt(2)*subs[1]
                orb_rot_mat[2,i] = 1/3+2/sym.sqrt(6)*subs[0]
            elif orb_symbol == 'sp3':
                orb_rot_mat[0,i] = 0.5*(subs[0]+subs[1]+subs[2] + 0.5)
                orb_rot_mat[1,i] = 0.5*(subs[0]-subs[1]-subs[2] + 0.5)
                orb_rot_mat[2,i] = 0.5*(subs[1]-subs[0]-subs[2] + 0.5)
                orb_rot_mat[3,i] = 0.5*(subs[2]-subs[1]-subs[0] + 0.5)
            elif orb_symbol == 'sp3d2':
                orb_rot_mat[0,i] = 1/6 -1/sym.sqrt(2)*subs[0] #- 1/sym.sqrt(12)*subs[0]*sym.sqrt(3.0) + 0.5*(subs[1]-subs[2])
                orb_rot_mat[1,i] = 1/6 +1/sym.sqrt(2)*subs[0] #- 1/sym.sqrt(12)*subs[0]*sym.sqrt(3.0) + 0.5*(subs[1]-subs[2])
                orb_rot_mat[2,i] = 1/6 -1/sym.sqrt(2)*subs[1] #- 1/sym.sqrt(12)*subs[0]*sym.sqrt(3.0) - 0.5*(subs[1]-subs[2])
                orb_rot_mat[3,i] = 1/6 +1/sym.sqrt(2)*subs[1] #- 1/sym.sqrt(12)*subs[0]*sym.sqrt(3.0) - 0.5*(subs[1]-subs[2])
                orb_rot_mat[4,i] = 1/6 -1/sym.sqrt(2)*subs[2] #+ 1/sym.sqrt(3)*subs[0]*sym.sqrt(3.0)
                orb_rot_mat[5,i] = 1/6 +1/sym.sqrt(2)*subs[2] #+ 1/sym.sqrt(3)*subs[0]*sym.sqrt(3.0)
                
                orb_rot_mat[0,i] +=  - 1/sym.sqrt(12)*subs_plus[0]*sym.sqrt(3.0)  + 0.5*(subs_plus[1]-subs_plus[2])
                orb_rot_mat[1,i] +=  - 1/sym.sqrt(12)*subs_plus[0]*sym.sqrt(3.0)  + 0.5*(subs_plus[1]-subs_plus[2])
                orb_rot_mat[2,i] +=  - 1/sym.sqrt(12)*subs_plus[0]*sym.sqrt(3.0)  - 0.5*(subs_plus[1]-subs_plus[2])
                orb_rot_mat[3,i] +=  - 1/sym.sqrt(12)*subs_plus[0]*sym.sqrt(3.0)  - 0.5*(subs_plus[1]-subs_plus[2])
                orb_rot_mat[4,i] +=  + 1/sym.sqrt(3)*subs_plus[0]*sym.sqrt(3.0)
                orb_rot_mat[5,i] +=  + 1/sym.sqrt(3)*subs_plus[0]*sym.sqrt(3.0)
            elif orb_symbol == 't2g':
                orb_rot_mat[0,i] = subs[0].evalf()
                orb_rot_mat[1,i] = subs[1].evalf()
                orb_rot_mat[2,i] = subs[2].evalf()
            elif orb_symbol == 'eg':
                orb_rot_mat[0,i] = (2*subs[0]-subs[1]-subs[2])/sym.sqrt(3.0)
                orb_rot_mat[2,i] =  (subs[1]-subs[2]).evalf()
                
        assert  np.abs(np.linalg.det(orb_rot_mat)) > 0.99,'ERROR!!!!: Your crystal symmetry does not allow {} orbital exist.'.format(orb_symbol)
        
        return orb_rot_mat

	
    def Part_P(self,rot_sym_glb,orb_symbol):
        ''' 
        Rotation matrix of Hamiltonian.

        Without SOC Part_P = rotation matrix of orbital
        With SOC Part_P = Kronecker product of rotation matrix of orbital and rotation matrix of spin 
        '''
        if abs(np.dot(np.transpose(rot_sym_glb),rot_sym_glb) - np.eye(3)).sum() >1.0E-4:
            print('rot_sym is not orthogomal \n {}'.format(rot_sym_glb))
        rmat = np.linalg.det(rot_sym_glb)*rot_sym_glb
        select = np.abs(rmat) < 0.01
        rmat[select] = 0.0 
        select = rmat > 0.99 
        rmat[select] = 1.0 
        select = rmat < -0.99 
        rmat[select] = -1.0 
        if self.soc:
            if np.abs(rmat[2,2]) < 1.0:
                beta = np.arccos(rmat[2,2])
                cos_gamma = -rmat[2,0] / np.sin(beta)
                sin_gamma =  rmat[2,1] / np.sin(beta)
                gamma = self.get_angle(sin_gamma, cos_gamma)
                cos_alpha = rmat[0,2] / np.sin(beta)
                sin_alpha = rmat[1,2] / np.sin(beta)
                alpha = self.get_angle(sin_alpha, cos_alpha)
            else:
                beta = 0.0
                if rmat[2,2] == -1. :beta = np.pi
                gamma = 0.0
                alpha = np.arccos(rmat[1,1])
                if rmat[0,1] > 0.0:alpha = -1.0*alpha
            euler_angle = np.array([alpha,beta,gamma])
            dmat = np.zeros((2,2),dtype=complex)
            dmat[0,0] =  np.exp(-(alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0)
            dmat[0,1] = -np.exp(-(alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0)
            dmat[1,0] =  np.exp( (alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0)
            dmat[1,1] =  np.exp( (alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0)
        rot_orbital = self.rot_orb(orb_symbol,rot_sym_glb)
        if self.soc:
            rot_orbital = np.kron(dmat,rot_orbital)
            rot_imag = rot_orbital.imag
            rot_real = rot_orbital.real
            rot_orbital = np.array(rot_real + 1j*rot_imag,dtype=complex)
        return rot_orbital

    def atom_rot_map(self,sym):
        '''
        rot_map: A map to show which atom is the equivalent atom after rotation operation.
        vec_shift_map: Change of R vector after rotation operation.
        '''
        wann_atom_positions = [self.wann_atom_info[i][2] for i in range(self.num_wann_atom)]
        rot_map=[]
        vec_shift_map=[]
        for atomran in range(self.num_wann_atom):
            atom_position=np.array(wann_atom_positions[atomran])
            new_atom = np.dot(self.symmetry['rotations'][sym],atom_position) + self.symmetry['translations'][sym]
            for atom_index in range(self.num_wann_atom):
                old_atom= np.array(wann_atom_positions[atom_index])
                diff = np.array(new_atom-old_atom)
                if np.all( abs((diff+0.5)%1-0.5)<1e-5 ):
                    match_index=atom_index
                    vec_shift= np.array(np.round(new_atom-np.array(wann_atom_positions[match_index]),decimals=2),dtype=int)
                else:
                    if atom_index==self.num_wann_atom-1:
                        assert atom_index != 0,'Error!!!!: no atom can match the new one Rvec = {}, atom_index = {}'.format(self.iRvec[ir],atom_index)
            rot_map.append(match_index)
            vec_shift_map.append(vec_shift)
        #Check if the symmetry operator respect to magnetic moment.
        #TODO opt magnet code
        rot_sym = self.symmetry['rotations'][sym]
        rot_sym_glb = np.dot(np.dot(np.transpose(self.lattice),rot_sym),np.linalg.inv(np.transpose(self.lattice)) )
        if self.soc:
            if self.magmom is not None:
                for i in range(self.num_wann_atom):
                    magmom = np.round(self.wann_atom_info[i][-2],decimals=4)
                    new_magmom =np.round( np.dot(rot_sym_glb,magmom),decimals=4)
                    if abs(np.linalg.norm(magmom - np.linalg.det(rot_sym_glb)*new_magmom)) > 0.0005:
                        sym_only = False
                    else:
                        sym_only = True
                        print('Symmetry operator {} is respect to magnetic moment'.format(sym+1) )
                    if abs(np.linalg.norm(magmom + np.linalg.det(rot_sym_glb)*new_magmom)) > 0.0005:
                        sym_T = False
                    else:
                        sym_T = True
                        print('Symmetry operator {}*T is respect to magnetic moment'.format(sym+1) )
                    if sym_T+sym_only == 0:
                        break

            else:
                sym_only = True
                sym_T = True
        else:
            sym_only = True
            sym_T = False

        return np.array(rot_map,dtype=int),np.array(vec_shift_map,dtype=int),sym_only,sym_T

    def full_p_mat(self,atom_index,rot):
        '''
        Combianing rotation matrix of Hamiltonian per orbital_quantum_number into per atom.  (num_wann,num_wann)
        '''
        orbitals = self.wann_atom_info[atom_index][3]
        orb_position_dic = self.wann_atom_info[atom_index][-1]
        p_mat = np.zeros((self.num_wann,self.num_wann),dtype = complex)
        p_mat_dagger = np.zeros((self.num_wann,self.num_wann),dtype = complex)
        rot_sym = self.symmetry['rotations'][rot]
        rot_sym_glb = np.dot(np.dot(np.transpose(self.lattice),rot_sym),np.linalg.inv(np.transpose(self.lattice)) )
        for orb in range(len(orbitals)):
            orb_name = orbitals[orb]
            tmp = self.Part_P(rot_sym_glb,orb_name)
            orb_position = orb_position_dic[orb_name]
            p_mat[orb_position] = tmp.flatten() 	
            p_mat_dagger[orb_position] = np.conj(np.transpose(tmp)).flatten()
        return p_mat,p_mat_dagger


    def average_H(self,iRvec,keep_New_R=True):
        #If we can make if faster, respectively is the better choice. Because XX_all matrix are supper large.(eat memory)  
        nrot = 0 
        R_list = np.array(iRvec,dtype=int)
        nRvec=len(R_list)
        tmp_R_list = []
        Ham_res = np.zeros((self.num_wann,self.num_wann,nRvec),dtype=complex)
        for X in self.matrix_list:
            if self.matrix_bool[X]:
                vars()[X+'_res'] = np.zeros((self.num_wann,self.num_wann,nRvec,3),dtype=complex)
        for rot in range(self.nsymm):
            rot_cart = np.dot(np.dot(np.transpose(self.lattice), self.symmetry['rotations'][rot]),np.linalg.inv(np.transpose(self.lattice)) )
            rot_map,vec_shift,sym_only,sym_T = self.atom_rot_map(rot)
            if sym_only+sym_T == 0:
                pass
            else:
                print('rot = ',rot+1)
                if sym_only: nrot+= 1
                if sym_T: nrot+= 1
                p_map = np.zeros((self.num_wann_atom,self.num_wann,self.num_wann),dtype=complex)
                p_map_dagger = np.zeros((self.num_wann_atom,self.num_wann,self.num_wann),dtype=complex)
                for atom in range(self.num_wann_atom):
                    p_map[atom],p_map_dagger[atom] = self.full_p_mat(atom,rot)
                R_map = np.dot(R_list,np.transpose(self.symmetry['rotations'][rot]))
                atom_R_map = R_map[:,None,None,:] - vec_shift[None,:,None,:] + vec_shift[None,None,:,:]
                Ham_all = np.zeros((nRvec,self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann),dtype=complex)
                for X in self.matrix_list:
                    if self.matrix_bool[X]:
                        vars()[X+'_all'] = np.zeros((nRvec,self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann,3),dtype=complex)
                
                #TODO try numba
                for iR in range(nRvec):
                    for atom_a in range(self.num_wann_atom):
                        for atom_b in range(self.num_wann_atom):
                            new_Rvec=list(atom_R_map[iR,atom_a,atom_b])
                            if new_Rvec in self.iRvec:
                                new_Rvec_index = self.iRvec.index(new_Rvec)
                                Ham_all[iR,atom_a,atom_b,self.H_select[atom_a,atom_b]] = self.Ham_R[self.H_select[rot_map[atom_a],
                                    rot_map[atom_b]],new_Rvec_index]
                                for X in self.matrix_list:
                                    if self.matrix_bool[X]:
                                        #if X in ['SS','AA','BB','CC']:
                                        if X in ['SS']:
                                            vars()[X+'_all'][iR,atom_a,atom_b,self.H_select[atom_a,atom_b],:] = vars(self)[X+'_R'][
                                            self.H_select[rot_map[atom_a],rot_map[atom_b]],new_Rvec_index,:].dot(np.transpose(
                                            np.linalg.inv(rot_cart)))
                                        if X in ['AA']:
                                            if iR == self.iRvec.index([0,0,0]) and atom_a == atom_b:
                                                cR_shift =np.einsum('i,ij->j',vec_shift[atom_a],self.lattice)
                                                cR_m = np.zeros((self.num_wann,self.num_wann,3))
                                                for i in self.wann_atom_info[atom_a][4]: 
                                                    cR_m[i,i,:] += cR_shift
                                                vars()[X+'_all'][iR,atom_a,atom_b,self.H_select[atom_a,atom_b],:] = (
                                                        vars(self)[X+'_R'][self.H_select[rot_map[atom_a],rot_map[atom_b]],new_Rvec_index,:] 
                                                        + cR_m[self.H_select[atom_a,atom_b],:]
                                                        ).dot(np.transpose(np.linalg.inv(rot_cart) ) )#
                                            else:
                                                vars()[X+'_all'][iR,atom_a,atom_b,self.H_select[atom_a,atom_b],:] = vars(self)[X+'_R'][
                                                self.H_select[rot_map[atom_a],rot_map[atom_b]],new_Rvec_index,:].dot(np.transpose(
                                                    np.linalg.inv(rot_cart) ) )
                                       
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
                        tmp = np.dot(np.dot(p_map_dagger[atom_a],Ham_all[:,atom_a,atom_b]),p_map[atom_b])
                        if sym_only: 
                            Ham_res += tmp.transpose(0,2,1)
                        
                        if sym_T:
                            tmp_T = self.ul.dot(tmp.transpose(1,0,2)).dot(self.ur).conj()
                            Ham_res += tmp_T.transpose(0,2,1)

                        for X in self.matrix_list:  # vector matrix
                            if self.matrix_bool[X]:
                                vars()[X+'_shift'] = vars()[X+'_all'].transpose(0,1,2,5,3,4)
                                tmpX= np.dot(np.dot(p_map_dagger[atom_a],vars()[X+'_shift'][:,atom_a,atom_b]),p_map[atom_b])
                                if np.linalg.det(self.symmetry['rotations'][rot]) < 0:
                                        parity_I = self.parity_I[X]
                                else: parity_I = 1

                                if sym_only:
                                    vars()[X+'_res'] += tmpX.transpose(0,3,1,2)*parity_I
                                if sym_T:
                                    tmpX_T = self.ul.dot(tmpX.transpose(1,2,0,3)).dot(self.ur).conj()
                                    vars()[X+'_res'] += tmpX_T.transpose(0,3,1,2)*parity_I*self.parity_TR[X]


        res_dic = {}
        res_dic['Ham'] = Ham_res/nrot
        

        for X in self.matrix_list:
            if self.matrix_bool[X]:
                X_res = X+'_res'
                res_dic[X] = vars()[X_res]/nrot
        
        print('number of symmetry oprator == ',nrot)
        

        if keep_New_R:
                return res_dic , tmp_R_list
        else:
                return res_dic			

    def symmetrize(self):
        #====Time Reversal====
        #syl: (sigma_y)^T *1j, syr: sigma_y*1j
        if self.soc:
            base_m = np.eye(self.num_wann//2)
            syl=np.array([[0.0,-1.0],[1.0,0.0]])
            syr=np.array([[0.0,1.0],[-1.0,0.0]])
            self.ul=np.kron(syl,base_m)
            self.ur=np.kron(syr,base_m)
        
        #========================================================
        #symmetrize exist R vectors and find additional R vectors 
        #========================================================
        print('##########################')
        print('Symmetrizing Start')
        res_dic_1, iRvec_add =  self.average_H(self.iRvec,keep_New_R=True)
        nRvec_add = len(iRvec_add)
        print('Symmetrizing Finish')
        
        return_dic = {}
        return_dic['Ham'] = res_dic_1['Ham']
        for X in self.matrix_list:
                if self.matrix_bool[X]:
                    return_dic[X] = vars()[X+'_R_final']= res_dic_1[X]
        return  return_dic, np.array(self.iRvec)

