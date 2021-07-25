import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pythtb as ptb
import matplotlib.pyplot as plt
import time
import wannierberri as wb
SYM=wb.symmetry
# Example for the interface PythTB-WannierBerri. Extracted from 
# http://www.physics.rutgers.edu/~dhv/pythtb-book-examples/ptb_samples.html
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
t =-1.4   # hopping parameter

# set on-site energy
my_model.set_onsite([Es])
# set hoppings along four unique bonds
# note that neighboring cell must be specified in lattice coordinates
for R in ([1,0,0],[0,1,0],[0,0,1],[1,1,1]):
    my_model.set_hop(t, 0, 0, R)


system=wb.System_PythTB(my_model,berry=True)
Efermi=np.linspace(-7,16,1000)
# Define the generators of the point group of the crystal (Im-3m)
# generators extracted from Bilbao Crystallographic Center
# (using same notation as https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-point_genpos?w2do=gens&num=32&what=)
# C2z axis (predefined): ITA200z=SYM.Rotation(2,[0,0,1])
# C2y axis (predefined): ITA20y0=SYM.Rotation(2,[0,1,0])
# Inversion (predefined)
ITA3M111=SYM.Rotation(3,[1,1,1])# C3 axis along 111
ITA2xx0=SYM.Rotation(2,[1,1,0])# C2 axis along 110
generators=['C2z','C2y','Inversion',ITA3M111,ITA2xx0]
system.set_symmetry(generators)
seedname='pythtbLi'
q_int=['dos','cumdos','conductivity_ohmic','conductivity_ohmic_fsurf']
num_iter=1
grid=wb.Grid(system,length=200,NKFFT=20)
parallel=wb.Parallel(num_cpus=8,method='ray')
start_int=time.time()
wb.integrate(system,
            grid=grid,
            Efermi=Efermi,
            smearEf=300,
            quantities=q_int,
            parallel=parallel,
            adpt_num_iter=num_iter,
            fout_name=seedname,
            restart=False )
end_int=time.time()
q_tab=["V"]
start_tab=time.time()
wb.tabulate(system,
             grid=grid,
             quantities=q_tab,
             frmsf_name=seedname,
             parallel=parallel,
             ibands=None,
             Ef0=0)
end_tab=time.time()

start_printing=time.time()
# Plot the quantities
dos_file=seedname+'-'+q_int[0]+'_iter-{0:04d}.dat'.format(num_iter)
dos_plot=seedname+'-'+q_int[0]+'_iter-{0:04d}.pdf'.format(num_iter)

with open(dos_file) as dos:
	array = np.loadtxt(dos.readlines()[1:])

figdos,axdos=plt.subplots()
axdos.plot(array[:,0],array[:,1],label='Calculated')
axdos.plot(array[:,0],array[:,2],label='Smoothed')
axdos.set_title(seedname+' '+q_int[0])
axdos.legend()
axdos.set_xlabel('Efermi')
figdos.savefig(dos_plot)

cumdos_file=seedname+'-'+q_int[1]+'_iter-{0:04d}.dat'.format(num_iter)
cumdos_plot=seedname+'-'+q_int[1]+'_iter-{0:04d}.pdf'.format(num_iter)
with open(cumdos_file) as cdos:
	array = np.loadtxt(cdos.readlines()[1:])

figcumdos,axcumdos=plt.subplots()
axcumdos.plot(array[:,0],array[:,1],label='Calculated')
axcumdos.plot(array[:,0],array[:,2],label='Smoothed')
axcumdos.legend()
axcumdos.set_title(seedname+' '+q_int[1])
axcumdos.set_xlabel('Efermi')
figcumdos.savefig(cumdos_plot)

ohmic_file=seedname+'-'+q_int[2]+'_iter-{0:04d}.dat'.format(num_iter)
ohmic_plot=seedname+'-'+q_int[2]+'_iter-{0:04d}.pdf'.format(num_iter)
with open(ohmic_file) as ohm:
	array = np.loadtxt(ohm.readlines()[1:])

figohm,axohm=plt.subplots()
axohm.plot(array[:,0],array[:,10],label='xx smooth')
axohm.plot(array[:,0],array[:,14],label='yy smooth')
axohm.plot(array[:,0],array[:,18],label='zz smooth')
axohm.legend()
axohm.set_title(seedname+' '+q_int[2])
axohm.set_xlabel('Efermi')
figohm.savefig(ohmic_plot)

ohmicfsurf_file=seedname+'-'+q_int[3]+'_iter-{0:04d}.dat'.format(num_iter)
ohmicfsurf_plot=seedname+'-'+q_int[3]+'_iter-{0:04d}.pdf'.format(num_iter)
with open(ohmicfsurf_file) as ohmfsf:
	array = np.loadtxt(ohmfsf.readlines()[1:])

figohmfsf,axohmfsf=plt.subplots()
axohmfsf.plot(array[:,0],array[:,10],label='xx smooth')
axohmfsf.plot(array[:,0],array[:,14],label='yy smooth')
axohmfsf.plot(array[:,0],array[:,18],label='zz smooth')
axohmfsf.legend()
axohmfsf.set_title(seedname+' '+q_int[2])
axohmfsf.set_xlabel('Efermi')
figohmfsf.savefig(ohmicfsurf_plot)

end_printing=time.time()

print('Integration time: ',end_int-start_int)
print('Tabulation time: ',end_tab-start_tab)
print('Printing time: ',end_printing-start_printing)