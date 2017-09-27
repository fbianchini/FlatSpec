import numpy as np
import sys
from scipy import ndimage
import matplotlib.pyplot as plt
from Utils import *
from Spec2D import *
from Sims import *
from scipy.linalg import pinv2
import cPickle as pickle
from IPython import embed
from curvspec.master import Binner
from tqdm import tqdm

def J(k1, k2, k3):	
	"""
	Computes the J function as defined in Eq.(A10) of MASTER paper (astro-ph/0105302)
	- k1 and k2 are numbers
	- k3 can be number/array
	"""
	k1 = np.asarray(k1, dtype=np.double)
	k2 = np.asarray(k2, dtype=np.double)
	k3 = np.asarray(k3, dtype=np.double)

	d  = np.broadcast(k1,k2,k3)
	J_ = np.zeros(d.shape)
	
	# Indices where J function is different from zero
	# idx = np.where((k1 < k2 + k3) & (k1 > np.abs(k2 - k3)))[0]
	idx1 = np.where(k1 > np.abs(k2-k3))[0]
	idx2 = np.where(k1 < k2+k3)[0]
	idx  = np.intersect1d(idx1,idx2)

	tmp = 2*k1**2*k2**2 + 2*k1**2*k2**2 + 2*k2**2*k3**2 - k1**4 - k2**4 - k3**4

	# print k1, k2, k3[idx], np.min(tmp[idx])


	if J_.ndim > 0: # J_ is an array
		J_[idx] = 2.0 / np.pi / np.sqrt(tmp[idx])
	else: # J_ is a number
		if idx.size > 0:
			J_ = 2.0 / np.pi / np.sqrt(tmp)
		else:
			J_ = 0.0
	
	J_[np.isnan(J_)] = 0.0

	return J_

def GetMll(mask, reso, lmin=0, lmax=4000, npts=4000):
	"""
	Returns the mode-coupling matrix from MASTER paper (astro-ph/0105302).
	Here k-vectors are already converted into l-vectors, while the paper uses wavenumber, hence the differences.
	The integral over the mask power spectrum is computed by means of Chebyschev-Gauss quadrature.

	TODO: optimize the for loops ! 

	"""
	assert ( len(mask.shape) == 2 )

	ell_bins, lbins, nbins = GetBins(delta_ell=1, lmin=lmin, lmax=lmax) 

	Mask   = FlatMapReal(mask.shape[1], reso, ny=mask.shape[1], dy=reso, map=mask)
	MaskFT = FlatMapFFT(mask.shape[1], reso, ny=mask.shape[1], dy=reso, map=Mask)

	# Calculating mask 1D power spectrum W(l)
	l3, Wl = MaskFT.GetCl(lbins=lbins)

	l1 = np.arange(lbins[-1]+1)
	l2 = np.arange(lbins[-1]+1)

	M_out = np.empty([l1.size,l2.size])

	# CG quadrature points & weights
	wi  = np.pi / npts
	tmp = (2.*(np.arange(1,npts+1)-1)) / (2.*npts) * np.pi
	vi  = np.cos(tmp)

	for i in xrange(l1.size):
		for j in xrange(l2.size):
			v    = (l3**2 - l1[i]**2 - l2[j]**2) / (2.0 * l1[i] * l2[j])
			fv   = Wl
			fvi  = np.interp(vi,v,fv, left=0., right=0.)
			tmp2 = np.sum(wi*fvi)

			M_out[i,j] = l2[j]/(2.*np.pi)  * tmp2

	return np.nan_to_num(M_out)

def GetMllCooray(mask, nx, reso, bins, nsim=100, buff=1):

	Mll = np.zeros((len(bins)-1,len(bins)-1))

	for ib in xrange(len(bins)-1):
		cl = np.zeros(1e4)
		cl[bins[ib]:bins[ib+1]+1] = 1.

		# plt.plot(cl)
		# plt.show()

		CL = np.zeros(len(bins)-1)

		for i in tqdm(xrange(nsim)):
			fake = GenCorrFlatMaps(cl, nx, reso, buff=buff)
			FAKE = FlatMapReal(nx, reso, map=fake, mask=mask)
			FT_FAKE = FlatMapFFT(nx, reso, map=FAKE)
			lb, clb = FT_FAKE.GetCl(lbins=bins)
			clb *= np.mean(mask**2)
			# plt.plot(lb, clb)
			# plt.show()
			CL[:] += clb

		Mll[:,ib] = CL/nsim

	return Mll