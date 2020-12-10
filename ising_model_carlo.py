#===================================================================
# Metropolis algo for generating ising model
# Author: Sanjana Sekhar
# Date: 12/8/20
#===================================================================

import numpy as np
import numba 
from numba import jit
import numpy.random as rng
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time

@jit
def monte_carlo_ising(Q,N,kT,lattice):

	ising = np.zeros((Q,N,N))
	mag = np.zeros((Q,1))
	accept = 0

	for index in range(0,Q):
		E_i,E_f=0,0
		#generate a random no i and j for index of spin to be flipped
		i,j,r = rng.randint(0,N), rng.randint(0,N), rng.uniform(0,1)
		##print('===============================')
		##print(i,j,r)
		test_lattice = np.copy(lattice)
		#flip
		test_lattice[i,j] = -test_lattice[i,j]
		
		#Compute energy for both configs
		#numpy is modifying the original too
		#check right
		if(j!=N-1):
			#print('right')
			E_i+=-(lattice[i,j]*lattice[i,j+1])
			E_f+=-(test_lattice[i,j]*test_lattice[i,j+1])
		elif(j==N-1):
			E_i+=-(lattice[i,j]*lattice[i,0])
			E_f+=-(test_lattice[i,j]*test_lattice[i,0])
		#check left
		if(j!=0):
			#print('left')
			E_i+=-(lattice[i,j]*lattice[i,j-1])
			E_f+=-(test_lattice[i,j]*test_lattice[i,j-1])
		elif(j==0):
			E_i+=-(lattice[i,j]*lattice[i,N-1])
			E_f+=-(test_lattice[i,j]*test_lattice[i,N-1])
		#check top 
		if(i!=0):
			#print('top')
			E_i+=-(lattice[i,j]*lattice[i-1,j])
			E_f+=-(test_lattice[i,j]*test_lattice[i-1,j])
		elif(i==0):
			E_i+=-(lattice[i,j]*lattice[N-1,j])
			E_f+=-(test_lattice[i,j]*test_lattice[N-1,j])
		#check bottom
		if(i!=N-1):
			#print('bottom')
			E_i+=-(lattice[i,j]*lattice[i+1,j])
			E_f+=-(test_lattice[i,j]*test_lattice[i+1,j])
		elif(i==N-1):
			E_i+=-(lattice[i,j]*lattice[0,j])
			E_f+=-(test_lattice[i,j]*test_lattice[0,j])


		#make the choice 
		#print('E_i = ',E_i)
		#print(lattice)
		#print('E_f = ',E_f)
		#print(test_lattice)
		delE = E_f - E_i 
		#print('delE = ',delE)
		if(delE < 0 or (delE >= 0 and r < np.exp(-delE/kT))):
			lattice = np.copy(test_lattice)
			ising[accept] = lattice
			#find magnetization
			#N_plus = np.sum(lattice.clip(0,1))
			#N_minus = N*N - N_plus
			#M = (N_plus - N_minus)/(N*N)
			mag[accept] = (2*np.sum(lattice.clip(0,1))-(N*N))/(N*N)
			accept+=1
	print('accept = ',accept)
	return ising[:accept+1],mag[:accept+1]


N_list = [10,20,30,40]
Q = 1000000
J = 1
date = 'dec9'
end = 0
for N in N_list:
	start = time.clock()
	print("=================== N = %i ==================="%(N))
	#sample from 40 temperatures 
	kT_list = np.linspace(1,3.5,40)

	pp = PdfPages('mag_per_T_N%i_%s.pdf'%(N,date))

	for kT in kT_list:
		print('Generating for T = ',kT)
		#Start off with a random config
		lattice = rng.choice([1, -1], size=(N, N))
		#print(lattice)
		if(kT<1.35):
			Q = 100000000
		elif(kT<2.3):
			Q = 2000000
		else:
			Q = 500000
		ising_config, mag = monte_carlo_ising(Q,N,kT,lattice)

		plt.hist(mag, bins=np.arange(-1,1,0.05), histtype='step', density=True,linewidth=2)
		plt.title('Probability of magnetization for T = %0.2f'%(kT))
		plt.xlabel('magnetization')
		pp.savefig()
		plt.close()

	pp.close()
	print('for N = %i time taken = %0.2f secs'%(N,time.clock()-start))
	end+=time.clock()-start
print('total time taken for MC generation = ',end)


