"""this script generates data from an AR(1) model. it then prints the least 
squares estimate of the parameter and then uses a particle filter to estimate 
this quantity"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

def f(i,j,rho):
	"""return the i,j th entry of the matrix"""
	if j > i:
		return 0
	else:
		return rho**(i-j)

def A(n,rho):
	"""return an n-by-n matrix (numpy array) with parameter rho"""
	M = [[f(i,j,rho) for j in range(n)] for i in range(n)]
	return np.array(M)

def S(n,rho):
	"""return a vector of n points in the process"""
	M = A(n,rho)
	# Z = np.random.standard_cauchy(size=(n,1))
	Z = np.random.normal(size=(n-1,1))
	Z = np.vstack((0,Z))
	return np.dot(M,Z)

def LSE(X):
	"""returns the least squares estimate of rho"""
	# numerator = np.dot(X[0:-1], X[1:])
	# denominator = np.dot(X[0:-1], X[0:-1])
	numerator = np.sum(X[0:-1]*X[1:])
	denominator = np.sum(X[0:-1]*X[0:-1])
	return numerator/denominator

def PF(X,N):
	"""returns an array where the first row is the mean values at each time and 
	the second row is the variance in the distribution 
	Args:
		X an array where each coloumn is an observation of the process
	Calls:
		PFinitialmeasure
		PFstep
		PFstats
	"""
	H = np.array([[0],[1./12]]) 	# initial condition is fixed at 0
	M = PFinitialmeasure(N)
	for i in range(1,len(X)):
		x = X[i-1]				# previous state
		z = X[i] 				# ith observation
		M = PFstep(M,x,z)		# update the measure
		S = PFstats(M)			# compute statistics
		H = np.hstack((H,S))	# append stats to H
	return H

def PFstep(M,x,z):
	"""returns posterior measure based on the previous measure M and the 
	measurement z
	Args:
		M is an array where first row is the support and the second the mass
		x is the previous state of the system
		z is a measurement of the system (the new state)
	Calls:
		PFimportance
		PFtarweights
	"""
	ImpSample = PFimportance(M)	# sample from importance density and likelihoods
	ImpSupport = ImpSample[0,:] # the new support
	ImpWeights = ImpSample[1,:] # the importance likelihoods
	# compute (proportional) likelihoods of the new support points in target pdf
	TarWeights = PFtarweights(ImpSupport,M,x,z) 
	W = TarWeights/ImpWeights	# compute ratio of pdf values
	W /= np.sum(W)				# normalize
	return np.vstack((ImpSupport,W))

def PFtarweights(ImpSupport,M,x,z):
	"""returns array of weights for the support points ImpM[0,:] with old state 
	x and measure M and new measurement z of the process
	Args:
		ImpM result of PFimportance
		M the old measure
		x the old process state
		z the new measurement of the process
	Calls:
		PFkde
		"""
	# array of P(x_{t+1}|x_t,theta) from the model
	G = np.exp(-0.5*((z - ImpSupport*x)**2)) 
	Result = PFkde(M,ImpSupport)*G
	return Result

def PFimportance(M):
	"""returns array where first row is a sample from the importance density and
	the second row is the likelihoods of these samples
	Args:
		M is a measure
	Calls:
		PFtriangpdf
	"""
	N = len(M[0,:])			# number of particles
	Peak = PFstats(M)		# mean and variance of old measure
	Peak = Peak[0]			# Peak is the old mean
	X = np.random.triangular(-1,Peak,1,size=N) # sample triangular density
	Y = [PFtriangpdf(x,Peak) for x in X]
	Y = np.array(Y)
	return np.vstack((X,Y.T))

def PFtriangpdf(x,Mode):
	"""returns value of pdf for a triangluar density on [-1,1] with 
	mode Mode at x
	Args:
		X a point
		Mode the value of the mode of the triangular density
	"""
	if x <= Mode:
		return (1+x)/(1+Mode)
	else:
		return (x-1)/(Mode-1)

def PFinitialmeasure(N):
	"""returns a measure array where the first row is the support and the second
	 row is the mass
	 Args:
	 	N the number of particles for the particle filter
	 """
	X = np.random.uniform(size=N)	# uniform prior
	M = np.ones(N)/N
	return np.vstack((X,M))

def PFkde(M,X):
	"""returns the evaluation of a kde approximation of M at the points in X 
	using a slightly suspicious resampling step to make the code nicer/easier
	""" 
	N = len(M[0,:])	# number of samples
	V = np.random.choice(M[0,:], size=N, p=M[1,:]) # resample from M
	K = stats.gaussian_kde(V)	# generate a gaussian kde approximation to M
	R = K.evaluate(X) # evaluate kde approximation at the points in X
	R /= K.integrate_box(-1,1)	# condition on being in [-1,1]
	return R 

def PFstats(Measure):
	"""returns the mean and variance of the measure provided
	Args:
		M is an array first row is support second row is mass
	"""
	X = Measure[0,:]
	M = Measure[1,:]
	mean = np.dot(X,M)
	var = np.dot(X**2,M) - mean**2
	return np.array([[mean],[var]])


def main():
	n = 70 # number of time steps
	numParticles = 10**3
	rho = 0.5 # model parameter

	T = np.arange(n)
	X = S(n,rho) # simulate the process
	LSestimates = [LSE(X[0:i]) for i in T]
	# print "The least squares estimate is rho = "+str(LSE(X))

	H = PF(X,numParticles)
	print H

	plt.figure()
	plt.plot(T,X,'b')
	plt.title('AR(1) with $\\theta=$'+str(rho))
	plt.show()

	plt.figure()
	plt.plot(\
		T,H[0,:]+2*H[1,:],'r',\
		T,H[0,:],'r',\
		T,H[0,:]-2*H[1,:],'r',\
		T,rho*np.ones(n),'b',T,LSestimates,'b--')
	plt.show()

if __name__ == '__main__':
	main()
