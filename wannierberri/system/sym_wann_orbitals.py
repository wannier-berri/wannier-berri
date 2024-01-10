import numpy as np
import sympy as sym


class Orbitals():

    def __init__(self):
        x = sym.Symbol('x')
        y = sym.Symbol('y')
        z = sym.Symbol('z')
        self.xyz = np.transpose([x, y, z])
        ss = lambda x, y, z: 1 + 0 * x
        px = lambda x, y, z: x
        py = lambda x, y, z: y
        pz = lambda x, y, z: z
        dz2 = lambda x, y, z: (2 * z * z - x * x - y * y) / (2 * sym.sqrt(3.0))
        dxz = lambda x, y, z: x * z
        dyz = lambda x, y, z: y * z
        dx2_y2 = lambda x, y, z: (x * x - y * y) / 2
        dxy = lambda x, y, z: x * y
        fz3 = lambda x, y, z: z * (2 * z * z - 3 * x * x - 3 * y * y) / (2 * sym.sqrt(15.0))
        fxz2 = lambda x, y, z: x * (4 * z * z - x * x - y * y) / (2 * sym.sqrt(10.0))
        fyz2 = lambda x, y, z: y * (4 * z * z - x * x - y * y) / (2 * sym.sqrt(10.0))
        fzx2_zy2 = lambda x, y, z: z * (x * x - y * y) / 2
        fxyz = lambda x, y, z: x * y * z
        fx3_3xy2 = lambda x, y, z: x * (x * x - 3 * y * y) / (2 * sym.sqrt(6.0))
        f3yx2_y3 = lambda x, y, z: y * (3 * x * x - y * y) / (2 * sym.sqrt(6.0))

        sp_1 = lambda x, y, z: 1 / sym.sqrt(2) * x
        sp_2 = lambda x, y, z: -1 / sym.sqrt(2) * x

        sp2_1 = lambda x, y, z: -1 / sym.sqrt(6) * x + 1 / sym.sqrt(2) * y
        sp2_2 = lambda x, y, z: -1 / sym.sqrt(6) * x - 1 / sym.sqrt(2) * y
        sp2_3 = lambda x, y, z: 2 / sym.sqrt(6) * x

        sp3_1 = lambda x, y, z: 0.5 * (x + y + z)
        sp3_2 = lambda x, y, z: 0.5 * (x - y - z)
        sp3_3 = lambda x, y, z: 0.5 * (-x + y - z)
        sp3_4 = lambda x, y, z: 0.5 * (-x - y + z)

        sp3d2_1 = lambda x, y, z: -1 / sym.sqrt(2) * x
        sp3d2_2 = lambda x, y, z: 1 / sym.sqrt(2) * x
        sp3d2_3 = lambda x, y, z: -1 / sym.sqrt(2) * y
        sp3d2_4 = lambda x, y, z: 1 / sym.sqrt(2) * y
        sp3d2_5 = lambda x, y, z: -1 / sym.sqrt(2) * z
        sp3d2_6 = lambda x, y, z: 1 / sym.sqrt(2) * z

        sp3d2_plus_1 = lambda x, y, z: -1 / sym.sqrt(12) * (2 * z * z - x * x - y * y) / (2 * sym.sqrt(3.0)) + 0.5 * (
            x * x - y * y) / 2
        sp3d2_plus_2 = lambda x, y, z: -1 / sym.sqrt(12) * (2 * z * z - x * x - y * y) / (2 * sym.sqrt(3.0)) + 0.5 * (
            x * x - y * y) / 2
        sp3d2_plus_3 = lambda x, y, z: -1 / sym.sqrt(12) * (2 * z * z - x * x - y * y) / (2 * sym.sqrt(3.0)) - 0.5 * (
            x * x - y * y) / 2
        sp3d2_plus_4 = lambda x, y, z: -1 / sym.sqrt(12) * (2 * z * z - x * x - y * y) / (2 * sym.sqrt(3.0)) - 0.5 * (
            x * x - y * y) / 2
        sp3d2_plus_5 = lambda x, y, z: +1 / sym.sqrt(3) * (2 * z * z - x * x - y * y) / (2 * sym.sqrt(3.0))
        sp3d2_plus_6 = lambda x, y, z: +1 / sym.sqrt(3) * (2 * z * z - x * x - y * y) / (2 * sym.sqrt(3.0))

        self.orb_function_dic = {
            's': [ss],
            'p': [pz, px, py],
            'd': [dz2, dxz, dyz, dx2_y2, dxy],
            'f': [fz3, fxz2, fyz2, fzx2_zy2, fxyz, fx3_3xy2, f3yx2_y3],
            'sp': [sp_1, sp_2],
            'p2': [pz, py],
            'sp2': [sp2_1, sp2_2, sp2_3],
            'pz': [pz],
            'sp3': [sp3_1, sp3_2, sp3_3, sp3_4],
            'sp3d2': [sp3d2_1, sp3d2_2, sp3d2_3, sp3d2_4, sp3d2_5, sp3d2_6],
            'sp3d2_plus': [sp3d2_plus_1, sp3d2_plus_2, sp3d2_plus_3, sp3d2_plus_4, sp3d2_plus_5, sp3d2_plus_6],
            't2g': [dxz, dyz, dxy],
            'eg': [dx2_y2, dz2],
        }
        self.orb_chara_dic = {
            's': [x],
            'p': [z, x, y],
            'd': [z * z, x * z, y * z, x * x, x * y, y * y],
            'f': [
                z * z * z, x * z * z, y * z * z, z * x * x, x * y * z, x * x * x, y * y * y, z * y * y, x * y * y,
                y * x * x
            ],
            'sp': [x, y, z],
            'p2': [z, y, x],
            'sp2': [x, y, z],
            'pz': [z, x, y],
            'sp3': [x, y, z],
            'sp3d2': [x, y, z],
            'sp3d2_plus': [z * z, x * x, y * y, x * y, x * z, y * z],
            't2g': [x * z, y * z, x * y, z * z, x * x, y * y],
            'eg': [z * z, x * x, y * y, x * z, y * z, x * y],
        }



    def num_orbitals(self, orb_symbol):
        return len(self.orb_function_dic[orb_symbol])


    def rot_orb(self, orb_symbol, rot_glb):
        ''' Get rotation matrix of orbitals in each orbital quantum number '''
        orb_dim = self.num_orbitals(orb_symbol)
        orb_rot_mat = np.zeros((orb_dim, orb_dim), dtype=float)
        xp, yp, zp = np.dot(np.linalg.inv(rot_glb), self.xyz)
        OC = self.orb_chara_dic[orb_symbol]
        OC_len = len(OC)
        if orb_symbol == 'sp3d2':
            OC_plus = self.orb_chara_dic[orb_symbol + '_plus']
            OC_plus_len = len(OC_plus)
        for i in range(orb_dim):
            subs = []
            equation = (self.orb_function_dic[orb_symbol][i](xp, yp, zp)).expand()
            for j in range(OC_len):
                eq_tmp = equation.subs(OC[j], 1)
                for j_add in range(1, OC_len):
                    eq_tmp = eq_tmp.subs(OC[(j + j_add) % OC_len], 0)
                subs.append(eq_tmp)
            if orb_symbol in ['sp3d2']:
                subs_plus = []
                equation_plus = (self.orb_function_dic[orb_symbol + '_plus'][i](xp, yp, zp)).expand()
                for k in range(OC_plus_len):
                    eq_tmp = equation_plus.subs(OC_plus[k], 1)
                    for k_add in range(1, OC_plus_len):
                        eq_tmp = eq_tmp.subs(OC_plus[(k + k_add) % OC_plus_len], 0)
                    subs_plus.append(eq_tmp)

            if orb_symbol in ['s', 'pz']:
                orb_rot_mat[0, 0] = subs[0].evalf()
            elif orb_symbol == 'p':
                orb_rot_mat[0, i] = subs[0].evalf()
                orb_rot_mat[1, i] = subs[1].evalf()
                orb_rot_mat[2, i] = subs[2].evalf()
            elif orb_symbol == 'd':
                orb_rot_mat[0, i] = (2 * subs[0] - subs[3] - subs[5]) / sym.sqrt(3.0)
                orb_rot_mat[1, i] = subs[1].evalf()
                orb_rot_mat[2, i] = subs[2].evalf()
                orb_rot_mat[3, i] = (subs[3] - subs[5]).evalf()
                orb_rot_mat[4, i] = subs[4].evalf()
            elif orb_symbol == 'f':
                orb_rot_mat[0, i] = (subs[0] * sym.sqrt(15.0)).evalf()
                orb_rot_mat[1, i] = (subs[1] * sym.sqrt(10.0) / 2).evalf()
                orb_rot_mat[2, i] = (subs[2] * sym.sqrt(10.0) / 2).evalf()
                orb_rot_mat[3, i] = (2 * subs[3] + 3 * subs[0]).evalf()
                orb_rot_mat[4, i] = subs[4].evalf()
                orb_rot_mat[5, i] = ((2 * subs[5] + subs[1] / 2) * sym.sqrt(6.0)).evalf()
                orb_rot_mat[6, i] = ((-2 * subs[6] - subs[2] / 2) * sym.sqrt(6.0)).evalf()
            elif orb_symbol == 'sp':
                orb_rot_mat[0, i] = 1 / 2 - 1 / sym.sqrt(2) * subs[0]
                orb_rot_mat[1, i] = 1 / 2 - 1 / sym.sqrt(2) * subs[0]
            elif orb_symbol == 'p2':
                orb_rot_mat[0, i] = subs[0]
                orb_rot_mat[1, i] = subs[1]
            elif orb_symbol == 'sp2':
                orb_rot_mat[0, i] = 1 / 3 - 1 / sym.sqrt(6) * subs[0] + 1 / sym.sqrt(2) * subs[1]
                orb_rot_mat[1, i] = 1 / 3 - 1 / sym.sqrt(6) * subs[0] - 1 / sym.sqrt(2) * subs[1]
                orb_rot_mat[2, i] = 1 / 3 + 2 / sym.sqrt(6) * subs[0]
            elif orb_symbol == 'sp3':
                orb_rot_mat[0, i] = 0.5 * (subs[0] + subs[1] + subs[2] + 0.5)
                orb_rot_mat[1, i] = 0.5 * (subs[0] - subs[1] - subs[2] + 0.5)
                orb_rot_mat[2, i] = 0.5 * (subs[1] - subs[0] - subs[2] + 0.5)
                orb_rot_mat[3, i] = 0.5 * (subs[2] - subs[1] - subs[0] + 0.5)
            elif orb_symbol == 'sp3d2':
                tmp_1 = (2 * subs_plus[0] - subs_plus[1] - subs_plus[2]) / 3
                tmp_2 = (subs_plus[1] - subs_plus[2]) / 2
                orb_rot_mat[0, i] = 1 / 6 - 1 / sym.sqrt(2) * subs[0] - tmp_1 / 2 + tmp_2
                orb_rot_mat[1, i] = 1 / 6 + 1 / sym.sqrt(2) * subs[0] - tmp_1 / 2 + tmp_2
                orb_rot_mat[2, i] = 1 / 6 - 1 / sym.sqrt(2) * subs[1] - tmp_1 / 2 - tmp_2
                orb_rot_mat[3, i] = 1 / 6 + 1 / sym.sqrt(2) * subs[1] - tmp_1 / 2 - tmp_2
                orb_rot_mat[4, i] = 1 / 6 - 1 / sym.sqrt(2) * subs[2] + tmp_1
                orb_rot_mat[5, i] = 1 / 6 + 1 / sym.sqrt(2) * subs[2] + tmp_1
            elif orb_symbol == 't2g':
                orb_rot_mat[0, i] = subs[0].evalf()
                orb_rot_mat[1, i] = subs[1].evalf()
                orb_rot_mat[2, i] = subs[2].evalf()
            elif orb_symbol == 'eg':
                orb_rot_mat[0, i] = (2 * subs[0] - subs[1] - subs[2]) / sym.sqrt(3.0)
                orb_rot_mat[2, i] = (subs[1] - subs[2]).evalf()

        assert np.abs(np.linalg.det(
            orb_rot_mat)) > 0.99, 'ERROR!!!!: Your crystal symmetry does not allow {} orbital exist. {}'.format(
                orb_symbol, np.abs(np.linalg.det(orb_rot_mat)))

        return orb_rot_mat
