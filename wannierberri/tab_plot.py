#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
#------------------------------------------------------------#
# This file is written by                                    #
# Xiaoxiong Liu, University of Zurich                        #
# Email: xxliu@physik.uzh.ch                                 #
#------------------------------------------------------------#
'''This is a script to plot the band with line path (High symmetry line) or plane cut using tabulate calculation result from Wannierberri. 

	NOTE: 

	      1. Please chose the line kpaths which are in the 1st BZ.
	      2. The plane cut figure shows a plane cut of 2x2x2 reciprocal lattice. 
	      3. All k coordinate values in list like parameters should be integer. (means which k point).
		 eg. If k-grid is 12x12x12. kpoint (1/3,2/3,0.5) should be (4,8,6)
	      4. In order to ensure that more high symmetry points have k-grid points, you would better set NK as multiples of 12 when doing tabulate calculation. 

        Usage example:
                line: ::

                    python3 -m wannierberri.tab_plot tab_result.pickle type=Line quantity=True kpath=0,0,0,0,0,40 namelist=G,Z qtype=berry component=z

                plane: ::

    	              python3 -m wannierberri.tab_plot tab_result.pickle type=Plane quantity=True Efermi=-0.5 vec1=1,0,0 vec2=0,1,0 qtype=berry component=z

        Options
            -h 
                | print the help message

            type (String)
                |  Type of plot. 
		|  	Line: line cut
		|  	Plane: plane cut 
                |  Default: None
            quantity (Boolean)
                |  Plot quantity or not.
		|	False: Only plot energy of band.
		|	True: Not only energy of band but also quantity(plot as color dot)  
                |  Default: False
            o_point (list like)
		|  k coordinate of origin.(type=Plane)
		|  Two vectors and one origin can define a plane. 
                |  Default: 0,0,0
            vec1 (list like) 
                |  k coordinate of one of the two vectors. (type=Plane)
		|  Only direction of vector work. (0,0,1) == (0,0,6)
		|  And it is the horizontal axis of plot.  
                |  Default: 0,0,1  
            vec2 (list like) 
                |  k coordinate of one of the two vectors. (type=Plane)
		|  Only direction of vector work. (0,1,0) == (0,6,0)
                |  Default: 0,1,0  
            kpath (list like)
                |  Starting points and ending points of k-path. (type=Plane)
		|  6 elements are one group, the first three elements are k coordinate of starting point and the back three elements are k coordinate of ending point. It should have multiples of 6 elements.
		|  coordinates are given as integers on the grid
		|  Default: 0,0,0,0,0,40 
            Efermi (float) 
                |  Fermi level when (type=Line)
		|  Plotting energy (when type=Plane) 
                |  default: 0.0
            E_min, E_max (float) 
                |  Energy window of plot. (type=Line) 
                |  default: E_min=-2 Emax=2
            qtype (str)
                |  type of quantities (quantity=True)
                |  spin,V,morb,berry,hall_spin,hall_orb
                |  default: None
            component (str)
                |  Cartesian coordinate projection (quantity=True)
                |  x,y,z
                |  default: None

'''



import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import pickle 

def hlp():
	from termcolor import cprint
	cprint ("tab_plot",'green', attrs=['bold'])
	print (__doc__)
	exit()


def main():
	if "-h" in argv[1:]:
		hlp()
	#########  input #############
	filename = argv[1] 
	Line=False
	Plane=False
	quantity=False
	o_point = np.array([0,0,0],dtype=int)    #original point  for kplane
	vec1 = np.array([0,0,1],dtype=int)       #two vectors to determin a plane
	vec2 = np.array([0,1,0],dtype=int)
	kpath=np.array(              #for kline
	[[0,0,0], 	             #path1 start
	[0,0,40]		     #path1 end
	],dtype=int)
	Efermi = 0.0
	E_min = -2    #E_min and E_max energy window for plot
	E_max = 2
	namelist=['A','B','C','D','E','F','G','H','I','J']
	qtype=None
	component=None
	for arg in argv[2:]:
		arg=arg.split("=")
		if arg[0]=="type":
			if arg[1]=="Line": Line=True
			elif arg[1]=="Plane": Plane=True
		if arg[0]=="quantity": 
			if arg[1]=="True": quantity=True
		if arg[0]=='o_point': o_point=np.array([x for x in arg[1].split(',')],dtype=int)
		if arg[0]=='vec1': vec1=np.array([x for x in arg[1].split(',')],dtype=int)
		if arg[0]=='vec2': vec2=np.array([x for x in arg[1].split(',')],dtype=int)
		if arg[0]=='kpath': 
			tmp=np.array([x for x in arg[1].split(',')],dtype=int)
			kpath=tmp.reshape(len(tmp)//3,3)
		if arg[0]=='Efermi': Efermi=float(arg[1])
		if arg[0]=='E_min': E_min=float(arg[1])
		if arg[0]=='E_max': E_max=float(arg[1])
		if arg[0]=='namelist': namelist=[x for x in arg[1]]
		if arg[0]=='component': component = arg[1]
		if arg[0]=='qtype': qtype = arg[1]

	####################################
	readstr  = lambda F : "".join(c.decode('ascii')  for c in F.read_record('c') ).strip()
	print('Reading file ...')
	tab_result=pickle.load(open(filename,"rb"))
	LATTICE = tab_result.recip_lattice
	NK=tab_result.grid
	NB=tab_result.nband
	EIGMAT = np.zeros((NB,NK[0],NK[1],NK[2]),dtype=float)
	if quantity:
 		QMAT = np.zeros((NB,NK[0],NK[1],NK[2]),dtype=float)
	for ib in range(NB):
		EIGMAT[ib] = tab_result.get_data(iband=ib,quantity ='E')
		if quantity:
			QMAT[ib] = tab_result.get_data(iband=ib,quantity = qtype ,component = component)
	EIGMAT=np.kron(np.ones((2,2,2)),EIGMAT)
	if quantity:
		QMAT=np.kron(np.ones((2,2,2)),QMAT)
	print('Finished reading...')
	
	K1 = np.linspace(-NK[0],NK[0],2*NK[0],endpoint=False,dtype=int)
	K2 = np.linspace(-NK[1],NK[1],2*NK[1],endpoint=False,dtype=int)
	K3 = np.linspace(-NK[2],NK[2],2*NK[2],endpoint=False,dtype=int)
	K = np.array(np.meshgrid(K1,K2,K3))

	print('kgrid = ' ,NK)
	print('number of bands = ',NB)
	print('k-vectors')
	print(LATTICE)

	def module(vec):
		return (vec[0]**2+vec[1]**2+vec[2]**2)**0.5

	def kline(nkp):
		n_l = (K-(kpath[2*nkp])[:,None,None,None])[:,:,:,:]
		l_m = (kpath[2*nkp]-kpath[2*nkp+1])[:,None,None,None]
		m_l = (kpath[2*nkp+1]-kpath[2*nkp])[:,None,None,None]
		cross = abs(n_l[1,:,:,:] * l_m[2] - n_l[2,:,:,:] * l_m[1]) + abs(n_l[0,:,:,:] * l_m[2] - n_l[2,:,:,:] * l_m[0]) +  abs(n_l[0,:,:,:] * l_m[1] - n_l[1,:,:,:] * l_m[0])
		dot = n_l[0,:,:,:] * m_l[0] + n_l[1,:,:,:] * m_l[1] + n_l[2,:,:,:] * m_l[2]
		select1 = cross == 0
		select2 = dot >= 0
		select3 = dot <= np.dot(kpath[2*nkp+1] - kpath[2*nkp],kpath[2*nkp+1]-kpath[2*nkp])
		select = select1*select2*select3
		positions = K[:,select]
		print('=_____________________________=')
		print('kpath = ',nkp+1)
		print('num of kpoint = ',len(positions[0]) )
		print(positions.transpose(1,0))
		eig = EIGMAT[:,select]
		kpatheig = np.zeros((NB,len(eig[0]),3))
		if quantity:	
			kpatheig = np.zeros((NB,len(eig[0]),3))
			quan = QMAT[:,select]
			kpatheig[:,:,1] = quan
		else:
			kpatheig = np.zeros((NB,len(eig[0]),2))
		kpatheig[:,:,0] = eig
		for i in range(len(positions[0])):
			dis = KPOP[nkp] + module(np.dot(positions[:,i]-kpath[2*nkp],LATTICE)) 
			kpatheig[:,i,-1] = dis
		if quantity:
			return kpatheig.reshape(NB*len(eig[0]),3)
		else:
			return kpatheig.reshape(NB*len(eig[0]),2)

	def kplane():
		cross_p = np.array([vec1[1] * vec2[2] - vec1[2] * vec2[1], vec1[0] * vec2[2] - vec1[2] * vec2[0] ,vec1[0] * vec2[1] - vec1[1] * vec2[0]])[:,None,None,None]
		Kvec = (K-o_point[:,None,None,None])[:,:,:,:]
		dot_p = Kvec[0,:,:,:]*cross_p[0] + Kvec[1,:,:,:]*cross_p[1] + Kvec[2,:,:,:]*cross_p[2] 
		select_p = dot_p==0
		positions_p = K[:,select_p]
		eig_p = EIGMAT[:,select_p]
		kc = np.dot(positions_p.transpose(1,0)/NK,LATTICE)
		kcv1 = np.dot(vec1,LATTICE)
		kcv2 = np.dot(vec2,LATTICE)
		V = np.dot(np.cross(LATTICE[0],LATTICE[1]),LATTICE[2])
		x=kcv1
		z=np.cross(kcv1,kcv2)
		z=z*V/np.dot(z,z)
		y=np.cross(x,z)
		y=y*V/np.dot(y,y)
		xlen=module(kcv1)
		angle = np.arccos(np.dot(kcv1,kcv2)/(module(kcv1)*module(kcv2)))
		ylen=module(kcv1)*np.sin(angle)
		kx = np.dot(kc,x)
		kx = kx*xlen/max(kx)
		ky = np.dot(kc,y)
		ky = ky*ylen/max(ky)
		if quantity:
			kplaneeig = np.zeros((NB,len(eig_p[0]),4))
			quan_p = QMAT[:,select_p]
			kplaneeig[:,:,1] = quan_p
		else:
			kplaneeig = np.zeros((NB,len(eig_p[0]),3))
		kplaneeig[:,:,0] = eig_p
		kplaneeig[:,:,-2] = kx
		kplaneeig[:,:,-1] = ky
	
		if quantity:
			return kplaneeig.reshape(NB*len(eig_p[0]),4)
		else:
			return kplaneeig.reshape(NB*len(eig_p[0]),3)
	###################### main #############################
	if Line:
		NKP = len(kpath)//2  #number of kpath
		KPOP = []
		op=0.
		KPOP.append(op)  #starting point of each path when ploting 
		for i in range(NKP):
			path=kpath[2*i+1]-kpath[2*i]
			op+=module(np.dot(path,LATTICE))
			KPOP.append(op)
	
		kpatheig=kline(0)
		if NKP > 1:
			for nkp in range(1,NKP):
				kpatheig=np.concatenate((kpatheig,kline(nkp)),axis=0)
	if Plane:
		kplaneeig = kplane()
	###################### plot #############################
	if Line:
		kpatheig = sorted(kpatheig,key=(lambda x:x[-1]))
		kpatheig=np.array(kpatheig)
	
		plt.figure()
		for b in range(NB):
			plt.plot(kpatheig[b::NB,-1],kpatheig[b::NB,0],'0.5')	
		if quantity:
			bar=plt.scatter(kpatheig[:,-1],kpatheig[:,0],c=kpatheig[:,1],s=10,cmap='seismic')
			plt.colorbar(bar)
		for i in range(len(KPOP)):
			plt.vlines(KPOP[i],E_min,E_max)
		kpts_name =[xx for xx in namelist][:len(KPOP)]
		plt.xticks(KPOP,kpts_name,fontsize='small')
		plt.ylim(E_min,E_max)
		plt.savefig('Line_eig.png')
#		plt.show()
	
	if Plane:
		plt.figure()
		psel = abs(kplaneeig[:,0]) < 0.01
		if quantity:
			bar=plt.scatter(kplaneeig[psel,-2],kplaneeig[psel,-1],c=kplaneeig[psel,1],s=10,cmap='seismic')
			plt.colorbar(bar)
		plt.scatter(kplaneeig[psel,-2],kplaneeig[psel,-1],c='k',s=2)
		plt.savefig('Plane_eig.png')
#		plt.show()		

	
if __name__ == "__main__":
	main()		 


