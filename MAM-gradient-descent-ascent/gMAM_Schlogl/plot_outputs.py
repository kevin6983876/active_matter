from numpy import loadtxt
from matplotlib import cm
import math
import cmath
import time
import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

import time
start_time = time.time()



parameters = np.loadtxt('listOfParameters.txt')
Lx    = 1.
Ncopy = int(parameters[0])
L     = int(parameters[1])
D	  = parameters[2]
dt    = parameters[3]
plotStep   = int(parameters[4])
iterations = int(parameters[5])
dx = 1./L


s = np.arange(0,Ncopy)/(float(Ncopy-1))
ds = s[1]-s[0]
sizeLabel = 17



############ Plot outputs from String Method

Lag = np.loadtxt('stringLagrangian_Ncopy'+str(int(Ncopy))+'_L'+str(int(L))+'.txt')

for k in range(0, iterations+1, plotStep):
	print(k)
	rho   = np.loadtxt('string_rho_'+str(k)+'.txt')
	theta = np.loadtxt('string_theta_'+str(k)+'.txt')
	
	LagString = Lag[k//(plotStep)]
	
	fig = plt.figure(figsize=(15,5))
	ax = fig.add_subplot(131)
	fig.suptitle(r'Parameters: $N_\mathrm{path}=$'+str(Ncopy)+r', $N_\mathrm{site}=$'+str(L)+'\n'+r'S='+str("%.6f"%(ds*np.sum(LagString)))+'\n'+ r'$t_\mathrm{sim}=$ '+str(k*dt), fontsize=14)  
	for i in range(1,Ncopy,5):
		plt.plot(np.arange(0,L), rho[i], lw=0.8)
	plt.plot(np.arange(0,L), rho[0], color='black', ls='-', lw=1.9)
	plt.plot(np.arange(0,L), rho[Ncopy-1], color='black', ls='--', lw=1.9)
	ax.set_xlabel(r'$x$', fontsize=sizeLabel)
	ax.set_ylabel(r'$\rho$', fontsize=sizeLabel)
	ax.set_xlim(0,L)
	ax.set_ylim(0,2)
	
	# Theta
	ax2 = fig.add_subplot(132)
	ax2.plot(s, np.zeros(Ncopy), color='k', linewidth=1 )
	for kk in range(0,L,2):
		ax2.plot(s, theta[:,kk], lw =0.8)
	ax2.set_xlim(0, 1)
	ax2.set_xlabel(r'$s$', fontsize=sizeLabel)
	ax2.set_ylabel(r'$\theta$', fontsize=sizeLabel)
	
	## Lagrangian
	ax3 = fig.add_subplot(133)
	ax3.plot(s, np.zeros(Ncopy), color='k', linewidth=1 )
	ax3.plot(s, LagString, color='k', linewidth=1 )
	ax3.set_xlim(0, 1)
	ax3.set_xlabel(r'$s$', fontsize=sizeLabel)
	ax3.set_ylabel(r'$L$', fontsize=sizeLabel)
	
	plt.tight_layout()
	plt.savefig("string_rho_theta_Lag_N%d_L%d_%09d" % (Ncopy,L,k)+'.png', bbox_inches='tight')
	plt.close()
	plt.clf()



############ Plot outputs from gMAM

Lag = np.loadtxt('gMAMLagrangian_Ncopy'+str(int(Ncopy))+'_L'+str(int(L))+'.txt')
lambdaArray = np.loadtxt('gMAMLambda_Ncopy'+str(int(Ncopy))+'_L'+str(int(L))+'.txt')
Niterations_gMAM = 100 * iterations

for k in range(0, Niterations_gMAM+1, plotStep):
	print(k)
	rho   = np.loadtxt('gMAM_rho_'+str(k)+'.txt')
	theta = np.loadtxt('gMAM_theta_'+str(k)+'.txt')
	
	LagMAM = Lag[k//plotStep]
	lambdaLocal = lambdaArray[k//plotStep]
	
	fig = plt.figure(figsize=(15,5))
	ax = fig.add_subplot(131)
	fig.suptitle(r'Parameters: $N_\mathrm{path}=$'+str(Ncopy)+r', $N_\mathrm{site}=$'+str(L)+'\n'+r'S='+str("%.6f"%(ds*np.sum(LagMAM)))+'\n'+ r'$t_\mathrm{sim}=$ '+str(k*dt), fontsize=14)  
	for i in range(1,Ncopy,5):
		plt.plot(np.arange(0,L), rho[i], lw=0.8)
	plt.plot(np.arange(0,L), rho[0], color='black', ls='-', lw=1.9)
	plt.plot(np.arange(0,L), rho[Ncopy-1], color='black', ls='--', lw=1.9)
	ax.set_xlabel(r'$x$', fontsize=sizeLabel)
	ax.set_ylabel(r'$\rho$', fontsize=sizeLabel)
	ax.set_xlim(0,L)
	ax.set_ylim(0,2)
	
	# Theta
	ax2 = fig.add_subplot(132)
	ax2.plot(s, np.zeros(Ncopy), color='k', linewidth=1 )
	for kk in range(0,L,2):
		ax2.plot(s, theta[:,kk], lw=0.8 )
	ax2.set_xlim(0, 1)
	ax2.set_xlabel(r'$s$', fontsize=sizeLabel)
	ax2.set_ylabel(r'$\theta$', fontsize=sizeLabel)
	
	## Lagrangian
	ax3 = fig.add_subplot(133)
	ax3.plot(s, np.zeros(Ncopy), color='k', linewidth=1 )
	ax3.plot(s, LagMAM, color='k', linewidth=1 )
	ax3.set_xlim(0, 1)
	ax3.set_xlabel(r'$s$', fontsize=sizeLabel)
	ax3.set_ylabel(r'$L$', fontsize=sizeLabel)
	
	plt.tight_layout()
	plt.savefig("gMAM_rho_theta_Lag_N%d_L%d_%09d" % (Ncopy,L,k)+'.png', bbox_inches='tight')
	plt.close()
	plt.clf()




#Plot final result
print("--- %s seconds ---" % (time.time() - start_time))






