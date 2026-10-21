from functools import cached_property
import os
import warnings
import numpy as np

from ..symmetry.point_symmetry import PointGroup

from ..utility import cached_einsum
from ..fourier.rvectors import Rvectors
from ..w90files.soc import SOC

from .system_R import System_R


class System_Slab(System_R):
    """
    A system that builds a SLAB Hamiltonian based on the System_R, by default, it is ALWAYS periodic in: (x',y',z')->(True,True,False)
    where z' is the rotated slab direction
    """

    def __init__(self,
                 system,
                 N_slab=12,#the number of slabs in the Ham
                 surf_vecs=np.array([[1,0,0],[0,1,0]]),#surface vectors, 
                 cleavage_level=None,#[Z_min,Z_max], just the WFs between these values will be accouted, this is a termination dependent calculation
                 atoms=None,write_geometry=None
                 ):
        self.needed_R_matrices = set()
        self.hr=system.get_R_mat('Ham')

        print('PBC Hamiltonian:',self.hr.shape)

        assert isinstance(system, System_R), f"system_up must be an instance of System_R, got {type(system)}"
        print(f'%%%%%%SLAB BUILD HAS STARTED%%%%%%%')
        self.is_phonon = False
        
        self.has_soc = False
        self._XX_R = dict()
        self.nslab=N_slab
        self.cleavage_level=cleavage_level
        self.force_internal_terms_only = any([system.force_internal_terms_only])
        self.surf_vecs=surf_vecs 
        self.old_lattice=system.real_lattice
        self.wannier_centers_primitive,self.num_wann_primitive=system.wannier_centers_cart,system.num_wann
        make_surface_basis(self)

        if atoms is not None:
            self.atoms_symbols_prim = [atom[0] for atom in atoms]
            self.atoms_symbols_slab=self.atoms_symbols_prim*self.nslab
            self.atoms_frac_coords = np.asarray(
                [atom[1:] for atom in atoms],
                dtype=float
            )

            self.atoms_cart_coords_prim = self.atoms_frac_coords @ system.real_lattice
 
            make_wannier_centres_slab(self,atoms=True,num_wann_prim=system.num_wann,write_geometry=write_geometry)
        else:
            make_wannier_centres_slab(self,num_wann_prim=system.num_wann,write_geometry=write_geometry)

        make_Rvec_and_Ham_slab(self,system.rvec.iRvec)




def make_surface_basis(self):
    #self.nslab
    v1,v2=self.surf_vecs[0],self.surf_vecs[1]
    if len(self.surf_vecs)==2:
        v3=np.array([0,0,1])
        U_matrix=np.array([v1,v2,v3])
    else:
        v3=self.surf_vecs[2]
        U_matrix=np.array([v1,v2,v3])
    #this thing with the cell volume I am doing but not so confident about it, should be verified afterwards
    real_lattice=U_matrix@self.old_lattice
    new_volume=np.linalg.det(real_lattice)
    old_volume=np.linalg.det(self.old_lattice)
    if new_volume<0:
        v3*=-1.
        U_matrix[2, :] = -U_matrix[2, :]
    if np.abs(np.abs(new_volume)-np.abs(old_volume))>0.001:
        print(f"cell volume is different-TAKE CARE")
    print(f"New cell's Volume is {new_volume} Ang^3")
    self.real_lattice=real_lattice
    self.U_matrix=U_matrix
    print(f' The 1st vector on surface     :{U_matrix[0]}')
    print(f' The 2nd vector on surface     :{U_matrix[1]}')
    print(f' The 3rd vector out of surface :{U_matrix[2]}')

    print(f"R1'= {self.real_lattice[0]}")
    print(f"R1'= {self.real_lattice[1]}")
    print(f"R1'= {self.real_lattice[2]}")   
    

def make_wannier_centres_slab(self,atoms=False,num_wann_prim=None,write_geometry=False):
    new_centers_pbc_frac = (self.wannier_centers_primitive@np.linalg.inv(self.real_lattice))%1
    new_centers_pbc=new_centers_pbc_frac@self.real_lattice
    #print(new_centers_pbc)
    translation = self.real_lattice[2]

    slab_centers = []

    for i in range(self.nslab):
        slab_centers.append(new_centers_pbc + i * translation)

    slab_centers = np.vstack(slab_centers)
    for i in slab_centers:
        print(f'{i[0]:.8f} {i[1]:.8f} {i[2]:.8f}')

    slab_real_lattice = np.array(
                [self.real_lattice[0, :], 
                 self.real_lattice[1, :], 
                 self.real_lattice[2, :]*self.nslab]
                 )
    print('SLAB LATTICE VECTORS:\n', slab_real_lattice)

    if self.cleavage_level is not None:
        z_min, z_max = self.cleavage_level[0]*slab_real_lattice[2,2], self.cleavage_level[1]*slab_real_lattice[2,2]
        cut_mask = (slab_centers[:,2] < z_min) | (slab_centers[:,2] > z_max)
        self.shift = np.where(cut_mask)[0]
        self.keep_mask = np.logical_not(cut_mask)
        print(f"Cut WF Positions (x, y, z):{self.shift}\n{slab_centers[self.shift]}")
        self.num_wann = int(self.nslab*num_wann_prim-len(self.shift))
        print(f'Your new num_wann is: {self.nslab*num_wann_prim}-{len(self.shift)}={self.num_wann}')

        if atoms is True:
            atoms_frac_rotated=(self.atoms_frac_coords@np.linalg.inv(self.U_matrix))%1
            atoms_cart_rotated=atoms_frac_rotated@self.real_lattice
            pos_cart_slab = []
            for i in range(self.nslab):
                pos_cart_slab.append(atoms_cart_rotated + i* translation)
            pos_cart_slab = np.vstack(pos_cart_slab)
            cut_mask = (pos_cart_slab[:,2] < z_min) | (pos_cart_slab[:,2] > z_max)
            print(f'NUMBER OF ATOMS REMOVED = {np.sum(cut_mask)}, check if this matches with your nem num_wann')
            pos_cart_slab = pos_cart_slab[np.logical_not(cut_mask)]
            self.atoms_symbols_slab = [symbol for symbol, mask in zip(self.atoms_symbols_slab, cut_mask) if not mask]
            print('SLAB CARTESIAN POSITIONS:\n')
            for i in pos_cart_slab:
                print(f'{i[0]:.8f} {i[1]:.8f} {i[2]:.8f}')

    else:
        cut_mask = (slab_centers[:,2] < -np.inf) | (slab_centers[:,2] > +np.inf)
        self.shift = np.where(cut_mask)[0]
        self.keep_mask = np.logical_not(cut_mask)
        #self.shift = np.array([], dtype=int)
        self.num_wann = int(self.nslab*num_wann_prim)
        if atoms is True:
            atoms_frac_rotated=(self.atoms_frac_coords@np.linalg.inv(self.U_matrix))%1
            atoms_cart_rotated=atoms_frac_rotated@self.real_lattice

            pos_cart_slab = []
            for i in range(self.nslab):
                pos_cart_slab.append(atoms_cart_rotated + i * translation)
            pos_cart_slab = np.vstack(pos_cart_slab)
  
            print('SLAB CARTESIAN POSITIONS:\n')
            for i in pos_cart_slab:
                print(f'{i[0]:.8f} {i[1]:.8f} {i[2]:.8f}')
            
    if write_geometry is True:
        write_CIF('SLABERRI.in',slab_real_lattice,self.atoms_symbols_slab,pos_cart_slab,slab_centers)

    self.wannier_centers_cart=slab_centers

def make_Rvec_and_Ham_slab(self,old_iRvec):
    M = np.linalg.inv(self.U_matrix)
    self.new_iRvec = np.rint(
        (M @ old_iRvec.T).T
    ).astype(int)

    if len(self.shift)!=0:
        pass
    print('IRVEC SHAPE=',self.new_iRvec.shape)

    total_uncleaved_wann = self.nslab*self.num_wann_primitive
    unique_R_plane = np.unique(self.new_iRvec[:, :2], axis=0)
    num_R_slab = len(unique_R_plane)
    Ham_slab_full = np.zeros(
        (num_R_slab, total_uncleaved_wann, total_uncleaved_wann), 
        dtype=complex
    )
    R_plane_to_idx = {tuple(r): idx for idx, r in enumerate(unique_R_plane)}

    for r_idx, R_prim in enumerate(old_iRvec):
        R_slab_proj = self.new_iRvec[r_idx]
        Rx, Ry, Rz = R_slab_proj
        
        H_prim = self.hr[r_idx]
        for l1 in range(self.nslab):
            l2 = l1+Rz
            
            if 0 <= l2 < self.nslab:
                if (Rx, Ry) in R_plane_to_idx:
                    r_slab_idx = R_plane_to_idx[(Rx, Ry)]
                    

                    idx1_start = l1 * self.num_wann_primitive
                    idx1_end = (l1 + 1) * self.num_wann_primitive
                    
                    idx2_start = l2 * self.num_wann_primitive
                    idx2_end = (l2 + 1) * self.num_wann_primitive

                    Ham_slab_full[
                        r_slab_idx, 
                        idx1_start:idx1_end, 
                        idx2_start:idx2_end
                    ] += H_prim

    Ham_slab = Ham_slab_full[:, self.keep_mask, :][:, :, self.keep_mask]

    # 6. Construct 3D slab R-vectors (append z=0 for surface/slab symmetry)
    slab_iRvec = np.hstack([unique_R_plane, np.zeros((num_R_slab, 1), dtype=int)])
   # self.hr
   # self.shift
   # self.num_wann

    # FINISHING DEFINITIONS HERE
    self.rvec = Rvectors(lattice=self.real_lattice,
                         iRvec=slab_iRvec,#self.new_iRvec,
                         shifts_left_red=self.wannier_centers_cart)#verify this
    self.periodic = (True,True,False)

    self.set_R_mat('Ham', Ham_slab)


def write_CIF(filename, lattice, symbols, new_atoms,slab_centers):
    with open(filename, 'w') as f:
        # Write 3 lattice vectors
        for vec in lattice:
            f.write(f"lattice_vector {vec[0]:.8f} {vec[1]:.8f} {vec[2]:.8f}\n")

        # Write atomic positions in Cartesian coordinates
        for pos, sym in zip(new_atoms, symbols):
            f.write(f"atom {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f} {sym}\n")
        f.write(f"#HERE ARE YOUR WANNIER FUNCTIONS so that you can cut them :)\n")
        for i in slab_centers:
            f.write(f"# {i[0]:.8f} {i[1]:.8f} {i[2]:.8f}\n")
    print(f"Geometry file written successfully to: {filename}")

