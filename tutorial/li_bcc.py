import matplotlib
matplotlib.use('Agg')
import numpy as np
import pythtb as ptb

import matplotlib.pyplot as plt

import wannierberri as wb
# example extracted from http://www.physics.rutgers.edu/~dhv/pythtb-book-examples/ptb_samples.html
# 3D model of Li on bcc lattice, with s orbitals only
# define lattice vectors
lat=[[-0.5, 0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5,-0.5]]
# define coordinates of orbitals
orb=[[0.0,0.0,0.0]]

# make 3D model
my_model=ptb.tb_model(3,3,lat,orb)

# set model parameters
# lattice parameter implicitly set to a=1
Es= 4.5    # site energy
t =-1.4    # hopping parameter

# set on-site energy
my_model.set_onsite([Es])
# set hoppings along four unique bonds
# note that neighboring cell must be specified in lattice coordinates
# (the corresponding Cartesian coords are given for reference)
for R in ([1,0,0],[0,1,0],[0,0,1],[1,1,1]):
    my_model.set_hop(t, 0, 0, R)    # [-0.5, 0.5, 0.5] cartesian


system=wb.System_PythTB(my_model,getAA=True)
Efermi=np.linspace(-5,5,1000)
generators=[wb.symmetry.Identity]
seedname="pythtbLi"
q_int=["dos","cumdos"]
num_iter=6
wb.integrate(system,
            NK=(50),
            Efermi=Efermi,
            smearEf=10, # 10K
            quantities=q_int,
            numproc=16,
            adpt_num_iter=num_iter,
            fout_name=seedname,
            symmetry_gen=generators,
            restart=False )
q_tab=["V"]
wb.tabulate(system,
             NK=(50),
             quantities=q_tab,
             symmetry_gen=generators,
             fout_name=seedname,
             numproc=16,
             ibands=None,
             Ef0=0)

# read the files
with open(seedname+'-'+q_int[0]+'_iter-{0:04d}.dat'.format(num_iter)) as f:
	array0 = np.loadtxt(f.readlines()[1:])

figdos,axdos=plt.subplots()
axdos.plot(array0[:,0],array0[:,2])
axdos.set_title(q_int[0])
axdos.set_xlabel('Efermi')
figdos.savefig(seedname+'-'+q_int[0]+'_iter-{0:04d}.pdf'.format(num_iter))

with open(seedname+'-'+q_int[1]+'_iter-{0:04d}.dat'.format(num_iter)) as f:
	array1 = np.loadtxt(f.readlines()[1:])

figcdos,axcdos=plt.subplots()
axcdos.plot(array1[:,0],array1[:,2])
axcdos.set_title(q_int[1])
axcdos.set_xlabel('Efermi')
figdos.savefig(seedname+'-'+q_int[1]+'_iter-{0:04d}.pdf'.format(num_iter))