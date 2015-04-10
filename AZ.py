"""
This is a collection of functions which I have found useful on previous 
occasions
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss 

def Resample(Measure,NResample=None):
	"""This is a function which returns a measure resampled NResample times
	Args:
		Measure is an array where each column describes a support point and the 
			final entry is the weight attached to that point
		NResample is the number of points to resample from the measure if this 
			arguement is not supplied it defaults to the number of support 
			points in the measure
	"""
	# define the number of points to resample from the measure
	SupportSize = len(Measure[0,:])	# number of support points
	if NResample == None:
		Size = SupportSize
	else:
		Size = NResample

	# resample Measure to get an iid sample of point indices
	Indices = np.arange(SupportSize)	# array indices
	Resample = np.random.choice(Indices, size=Size, p=Measure[-1,:])
	# construct the new measure based on the indices just sampled
	# Dimension = len(Measure[:,0]) - 1 # dimension of support points
	Result = np.vstack(( Measure[0:-1,Resample], np.ones(Size)/Size ))
	return Result

def KDEplot(Measure,Filename,NResample=None):
	"""This function takes a 2-by-N numpy array which encodes a random measure.
	the first row is the support of of the measure and the second the weights 
	associated with each point. the measure is sampled to obtain an iid sample 
	and then gaussian kde is used to approximate the density. a plot of the 
	density is saved under the given filename
	Args:
		Measure is an array (random measure) as described above
		Filename is the name given to the saved plot
		NResample is the number of points to resample
	"""
	if NResample == None:
		Size = len(Measure[0,:])
	else:
		Size = NResample
	# resample the given measure to get an iid sample
	Resample = np.random.choice(Measure[0,:], size=Size, p=Measure[1,:])
	# generate a gaussian KDE approximation to the density
	GKDE = ss.gaussian_kde(Resample)
	# generate a plot of the figure and save it as Filename
	xMin = Measure[0,:].min()
	xMax = Measure[0,:].max()
	X = np.linspace(xMin, xMax, num=100)	# domain
	Y = GKDE(X)								# density values

	plt.figure()
	plt.plot(X,Y)
	plt.savefig(Filename)

def EstEffNum(Measure):
	"""Takes a Measure in which the bottom row is the weights associated to the 
	support points defined by the upper part of the array (so you can have 
	support points in higher dimensions). it then returns an estimate of the 
	effective number of particles as given in equation 51 of "A Tutorial on 
	Rarticles Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking"
	Args:
		Measure is an array with last row being the weights
	"""
	W = Measure[-1,:]
	return 1./(np.sum(W**2))
