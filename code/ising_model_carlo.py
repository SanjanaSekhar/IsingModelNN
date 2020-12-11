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
import h5py

@jit
def monte_carlo_ising(Q,N,kT,lattice):

	ising = np.zeros((Q,N,N))
	mag = np.zeros((Q,1))
	accept = 0

	for index in range(0,Q):
		
		if(index%100000==0):
			print(index)
		E_i,E_f=0,0
		#generate a random no i and j for index of spin to be flipped
		i,j,r = rng.randint(0,N), rng.randint(0,N), rng.uniform(0,1)
		#Compute energy for both configs
		#check right
		if(j!=N-1):
			E_i+=-(lattice[i,j]*lattice[i,j+1])
			E_f+=(lattice[i,j]*lattice[i,j+1]) #for a spin flip
		elif(j==N-1):
			E_i+=-(lattice[i,j]*lattice[i,0])
			E_f+=(lattice[i,j]*lattice[i,0]) #for a spin flip
		#check left
		if(j!=0):
			E_i+=-(lattice[i,j]*lattice[i,j-1])
			E_f+=(lattice[i,j]*lattice[i,j-1]) #for a spin flip
		elif(j==0):
			E_i+=-(lattice[i,j]*lattice[i,N-1])
			E_f+=(lattice[i,j]*lattice[i,N-1]) #for a spin flip
		#check top 
		if(i!=0):
			E_i+=-(lattice[i,j]*lattice[i-1,j])
			E_f+=(lattice[i,j]*lattice[i-1,j]) #for a spin flip
		elif(i==0):
			E_i+=-(lattice[i,j]*lattice[N-1,j])
			E_f+=(lattice[i,j]*lattice[N-1,j]) #for a spin flip
		#check bottom
		if(i!=N-1):
			E_i+=-(lattice[i,j]*lattice[i+1,j])
			E_f+=(lattice[i,j]*lattice[i+1,j]) #for a spin flip
		elif(i==N-1):
			E_i+=-(lattice[i,j]*lattice[0,j])
			E_f+=(lattice[i,j]*lattice[0,j]) #for a spin flip


		#make the choice 
		delE = E_f - E_i 
		if(delE < 0 or (delE >= 0 and r < np.exp(-delE/kT))):
			lattice[i,j] = -lattice[i,j]
			ising[accept] = lattice
			#find magnetization
			#N_plus = np.sum(lattice.clip(0,1))
			#N_minus = N*N - N_plus
			#M = (N_plus - N_minus)/(N*N)
			mag[accept] = (2*np.sum(lattice.clip(0,1))-(N*N))/(N*N)
			accept+=1

	print('accept = ',accept)
	return ising[:accept],mag[:accept]

@jit
def generate_data_perN(N,date,n_per_T,n_temps,T_c):

	Q = 10 #for testing

	ising_config = np.zeros((n_per_T*n_temps,N,N))
	mag = np.zeros((n_per_T*n_temps,1))	
	temp = np.zeros((n_per_T*n_temps,1))	
	#label T < T_c = 1, T > T_c = 0
	label = np.zeros((n_per_T*n_temps,1))	

	print("=================== N = %i ==================="%(N))
	#sample from 40 temperatures 
	kT_list = np.linspace(1,3.5,n_temps)

	pp = PdfPages('plots/mag_perT_N%i_%s.pdf'%(N,date))

	start = time.clock()
	for index in range(len(kT_list)):

		print('Generating for T = ',kT_list[index])
		#Start off with a random config
		lattice = rng.choice([1, -1], size=(N, N))
		
		if(kT_list[index]<2.):
			Q = 50000000
		elif(kT_list[index]<2.4):
			Q = 2000000
		else:
			Q = 500000
		
		ising_config_perT, mag_perT = monte_carlo_ising(Q,N,kT_list[index],lattice)

		#sample configs evenly spaced
		idx = np.round(np.linspace(0, len(ising_config_perT) - 1, n_per_T)).astype(int)
		ising_config[(index*n_per_T):(n_per_T*(index+1))] = ising_config_perT[idx]
		mag[(index*n_per_T):(n_per_T*(index+1))] = mag_perT[idx]
		temp[(index*n_per_T):(n_per_T*(index+1))] = kT_list[index]
		if(kT_list[index]<T_c):
			label[(index*n_per_T):(n_per_T*(index+1))] = 1

		plt.hist(mag_perT, bins=np.arange(-1,1,0.05), histtype='step', density=True,linewidth=2)
		plt.title('Probability of magnetization for T = %0.2f'%(kT_list[index]))
		plt.xlabel('magnetization')
		pp.savefig()
		plt.close()

	pp.close()

	end = time.clock()-start
	print('for N = %i, MC generation time taken = %0.2f secs'%(N,end))

	return ising_config,mag,temp,label,end

def create_datasets(f,ising_config,mag,temp,label,dset_type):
	
	ising_dset = f.create_dataset("ising", np.shape(ising_config), data=ising_config)
	mag_dset = f.create_dataset("mag", np.shape(mag), data=mag)
	temp_dset = f.create_dataset("temp", np.shape(temp), data=temp)
	label_dset = f.create_dataset("label", np.shape(label), data=label)

	print("made %s h5 file. no. of events to %s on: %i"%(dset_type,dset_type,len(label)))

N_list = [20]
J = 1
date = 'dec10'
end = 0
n_per_T = 25000
n_temps = 40
T_c = 2.268
n_train = int(n_per_T*n_temps*0.75) #75% of matrices will be for train+val
n_test = n_per_T*n_temps - n_train

for N in N_list:

	ising_config,mag,temp,label,time_perN = generate_data_perN(N,date,n_per_T,n_temps,T_c)
	end+=time_perN
	#shuffle entries
	x = np.arange(0,n_per_T*n_temps,1)
	p = rng.permutation(x)
	ising_config,mag,temp,label = ising_config[p],mag[p],temp[p],label[p]	

	#create h5 files
	#remember configs are being stored as NxN
	#need to flatten for DNN or reshape for CNN
	f = h5py.File("h5_files/train_N%i_%s.hdf5"%(N,date), "w")
	create_datasets(f,ising_config[:n_train],mag[:n_train],temp[:n_train],label[:n_train],'train')
	f = h5py.File("h5_files/N%i_%s.hdf5"%(N,date), "w")
	create_datasets(f,ising_config[n_train:],mag[n_train:],temp[n_train:],label[n_train:],'test')

print('total time taken for MC generation = ',time_perN)


