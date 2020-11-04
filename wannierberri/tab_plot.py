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
'''This utility calculates the matrices .uHu and/or .uIu from the .mmn matrices, and also reduces the number of bands in .amn, .mmn, .eig  and .spn files

        Usage example:
                python3 -m wanierberri.mmn2uHu seedname NBout=10 NBsum=100,200  targets=mmn,uHu formatted=uHu
                
        Options
            -h 
                | print the help message
            IBstart
                |  the first band in the output file (counting starts from 1). 
                |  default: 1
            IBstartSum
                |  the first band in the sum         (counting starts from 1). 
                |  default: 1
            NBout 
                |  the number of bands in the output files. 
                |  Default : all bands
            NBsum 
                |  the number of bands in the summation. (one may specify several numbers, usefull to test convergence with the number of bands). 
                |  Default:all bands
            input 
                |  path to the input files. 
                |  Default: ./  
            output 
                |  path to the output files 
            targets 
                |  files to write : ``amn``, ``mmn``, ``spn``, ``uHu``, ``uIu``, ``eig`` 
                |  default: ``amn``,``mmn``,``eig``,``uHu``
            formatted 
                |  files to write as formatted  ``uHu``, ``uIu``, ``spn``, ``spn_in``, ``spn_out``, ``all`` 
                |  default: none

'''



import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import math
from scipy.io import FortranFile

def hlp():
	from termcolor import cprint
	cprint ("tab_plot",'green', attrs=['bold'])
	print (__doc__)


def main():
	hlp()
	#########  input #############
	filename = argv[1] 
	Line=False
	Plane=False
	quantity=False
	formatted = False
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
	for arg in argv[2:]:
		arg=arg.split("=")
		if arg[0]=="type":
			if arg[1]=="Line": Line=True
			elif arg[1]=="Plane": Plane=True
		if arg[0]=="quantity": 
			if arg[1]=="True": quantity=True
		if arg[0]=="formatted": 
			if arg[1]=="True": formatted=True
		if arg[0]=='o_point': o_point=np.array([x for x in arg[1].split(',')],dtype=int)
		if arg[0]=='vec1': vec1=np.array([x for x in arg[1].split(',')],dtype=int)
		if arg[0]=='vec2': vec2=np.array([x for x in arg[1].split(',')],dtype=int)
		if arg[0]=='kpath': 
			tmp=np.array([x for x in arg[1].split(',')],dtype=int)
			kpath=tmp.reshape(len(arg[1].split)//3,3)
		if arg[0]=='Efermi': Efermi=float(arg[1])
		if arg[0]=='E_min': E_min=float(arg[1])
		if arg[0]=='E_max': E_min=float(arg[1])
		if arg[0]=='namelist': namelist=[x for x in arg[1]]

	####################################
	readstr  = lambda F : "".join(c.decode('ascii')  for c in F.read_record('c') ).strip()
	print('Reading file ...')
	if formatted:
		filename=filename+".frmsf"
		NK = np.loadtxt(filename,skiprows=0,max_rows=1,dtype=int)   # number of kpoint [k1,k2,k3]
		NB = int(np.loadtxt(filename,skiprows=2,max_rows=1))        # number of band
		LATTICE = np.loadtxt(filename,skiprows=3,max_rows=3)        # k vectors
		num=NK[0]*NK[1]*NK[2]*NB
		EIG = np.loadtxt(filename,skiprows=6,max_rows=num)#,max_rows=NK[0]*NK[1]*NK[2]*NB)    
		EIGMAT = np.kron(np.ones((2,2,2)),EIG.reshape(NB,NK[0],NK[1],NK[2]))-Efermi            # eigenvalue matrix
		if quantity:
			Q = np.loadtxt(filename,skiprows=num+6)
			QMAT = np.kron(np.ones((2,2,2)),Q.reshape(NB,NK[0],NK[1],NK[2]))            # quantities matrix
	else:
		read_f = FortranFile(filename, 'r')
		header=readstr(read_f).replace('\n', ' ').split()
		LATTICE = read_f.read_record('f8').reshape(3,3)
		NK=np.array([header[0],header[1],header[2]],dtype=int)
		NB=int(header[-1])
		EIG = read_f.read_record('f8')
		EIGMAT = np.kron(np.ones((2,2,2)),EIG.reshape(NK[0],NK[1],NK[2],NB).transpose(3,0,1,2))-Efermi
		if quantity:
			Q = read_f.read_record('f8')
			QMAT = np.kron(np.ones((2,2,2)),Q.reshape(NK[0],NK[1],NK[2],NB).transpose(3,0,1,2))
	print('Finished reading...')
	
	K1 = np.linspace(-NK[0],NK[0],2*NK[0],endpoint=False,dtype=int)
	K2 = np.linspace(-NK[1],NK[1],2*NK[1],endpoint=False,dtype=int)
	K3 = np.linspace(-NK[2],NK[2],2*NK[2],endpoint=False,dtype=int)
	K = np.array(np.meshgrid(K1,K2,K3))

	print('kgrid = ' ,NK)
	print('number of bands = ',NB)
	print('k-vectors')
	print(LATTICE)

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

		 


