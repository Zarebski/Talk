"""this script generates data from an AR(1) model. it then prints the least 
squares estimate of the parameter and then uses a particle filter to estimate 
this quantity"""

import numpy as np
import matplotlib.pyplot as plt

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

def main():
	n = 10**3 # number of time steps
	rho = 0.8 # model parameter

	T = np.arange(n)
	X = S(n,rho) # simulate the process
	print "The least squares estimate is rho = "+str(LSE(X))

	plt.figure()
	plt.plot(T,X)
	plt.show()

if __name__ == '__main__':
	main()
