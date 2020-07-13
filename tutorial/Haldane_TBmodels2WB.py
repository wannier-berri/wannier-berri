import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')
import wannierberri as wb
SYM=wb.symmetry
import tbmodels as tbm # import TB model class
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# Example for the interface TBmodels-WannierBerri. The model is an example 
# of a two-dimensional tight binding model (Haldane model) written in 
# TBmodels.
# Check http://z2pack.ethz.ch/tbmodels/doc/1.3/index.html for more information
# about TBmodels
def HaldaneTB(delta,t1,hop2,phi):
# define lattice vectors
    lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
# define coordinates of orbitals
    orb=[[1./3.,1./3.],[2./3.,2./3.]]

# make two dimensional tight-binding Haldane model
    haldane=tbm.Model(on_site=[-delta,delta],
                      dim=2,
                      pos=orb,
                      uc=np.array(lat))

# set model parameters
    t2=hop2*np.exp(1.j*phi)


# set hoppings (one for each connected pair of orbitals)
# from j in R to i in 0
# (amplitude, i, j, [lattice vector to cell containing j])
    haldane.add_hop(t1, 0, 1, [ 0, 0])
    haldane.add_hop(t1, 1, 0, [ 1, 0])
    haldane.add_hop(t1, 1, 0, [ 0, 1])
# add second neighbour complex hoppings
    haldane.add_hop(t2 , 0, 0, [ 0, -1])
    haldane.add_hop(t2 , 0, 0, [ 1, 0])
    haldane.add_hop(t2 , 0, 0, [ -1, 1])
    haldane.add_hop(t2 , 1, 1, [ -1, 0])

    haldane.add_hop(t2 , 1, 1, [ 1,-1])
    haldane.add_hop(t2 , 1, 1, [ 0, 1])
    return haldane

# Define the model for a fixed set of parameters
haldane=HaldaneTB(2,1,1/3,np.pi/10)
# Call the interface for TBmodels to define the system class
syst=wb.System_TBmodels(haldane,getAA=True)
Efermi=np.linspace(-4,6,1000)
# Define some symmetries
syst.set_symmetry(['C3z'])
# After defining the symmetries, create the grid class
grid=wb.Grid(syst,NK=(200,200,1))
# Define which quantities are going to be integrated
q_int=["dos","ahc"]
seedname="tbmodels_Haldane"
num_iter=10
wb.integrate(syst,
            grid=grid,
            Efermi=Efermi,
            smearEf=300, # 10K
            quantities=q_int,
            numproc=8,
            adpt_num_iter=num_iter,
            fout_name=seedname,
            restart=False )

# Plot the quantities
dos_file=seedname+'-'+q_int[0]+'_iter-{0:04d}.dat'.format(num_iter)
dos_plot=seedname+'-'+q_int[0]+'_iter-{0:04d}.pdf'.format(num_iter)

with open(dos_file) as dos:
	array = np.loadtxt(dos.readlines()[1:])

figdos,axdos=plt.subplots()
axdos.plot(array[:,0],array[:,1],label='Calculated')
axdos.plot(array[:,0],array[:,2],label='Smoothed')
axdos.set_title(seedname+' '+q_int[0])
axdos.set_xlabel('Efermi')
axdos.legend()
figdos.savefig(dos_plot)

ahc_file=seedname+'-'+q_int[1]+'_iter-{0:04d}.dat'.format(num_iter)
ahc_plot=seedname+'-'+q_int[1]+'_iter-{0:04d}.pdf'.format(num_iter)

with open(ahc_file) as ahc:
	array = np.loadtxt(ahc.readlines()[1:])

figahc,axahc=plt.subplots()
axahc.plot(array[:,0],array[:,3],label='Calculated')
axahc.plot(array[:,0],array[:,6],label='Smoothed')
axahc.set_title(seedname+' '+q_int[1]+'z component')
axahc.set_xlabel('Efermi')
axahc.legend()
figdos.savefig(dos_plot)
figahc.savefig(ahc_plot)