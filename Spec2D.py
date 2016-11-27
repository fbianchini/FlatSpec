import numpy as np
import sys
from scipy import ndimage
import matplotlib.pyplot as plt
from Utils import *

from IPython import embed
# Here you can find some classes to deal with flat-sky scalar maps 
# (like temperature, lensing, galaxies,..) and their FFTs.
# The following code is heavily inspired from some of Duncan's quicklens code 
# that can be found at https://github.com/dhanson/quicklens


arcmin2rad = np.pi / 180. / 60. 
rad2arcmin = 1./arcmin2rad

def GaussSmooth(map, fwhm, reso, order=0):
	"""
	Smooth the map with a Gaussian beam specified by its FWHM (in arcmin).
	- fwhm: 
	- reso: pixel resolution (in arcmin)
	"""
	# reso_  = reso * 180.*60./np.pi # in arcmin
	sigma  = fwhm / np.sqrt(8*np.log(2)) / reso
	# print("\t smoothing map with sigma = %4f" %sigma)
	return ndimage.gaussian_filter(map, sigma=sigma, order=order)


class Pix(object):
	""" 
	Class that contains the pixelization scheme (considers rectangular pixels).
	This is at the core of both the Maps and FFT classes.

	Params
	------
	- nx:  # of pixels in x dir
	- dx: pixel resolution in *arcmin*
	"""
	def __init__(self, nx, dx, ny=None, dy=None):
		if ny is None:
			ny = nx
		if dy is None:
			dy = dx

		# Converting pixel resolution from arcmin -> rad
		dx *= arcmin2rad 
		dy *= arcmin2rad

		self.nx    = nx 
		self.dx    = dx 
		self.ny    = ny 
		self.dy    = dy
		self.npix  = self.nx * self.ny
		self.shape = (self.ny, self.nx)

		# Area (in radians) and sky fraction of the patch
		self.area = (self.nx * self.dx) * (self.ny * self.dy)
		self.fsky = self.area / 4. / np.pi

	def GetLxLy(self, shift=True):
		""" 
		Returns two grids with the (lx, ly) pair associated with each Fourier mode in the map. 
		If shift=True (default), \ell = 0 is centered in the grid
		~ Note: already multiplied by 2\pi 
		"""
		if shift:
			return np.meshgrid( np.fft.fftshift(np.fft.fftfreq(self.nx, self.dx))*2.*np.pi, np.fft.fftshift(np.fft.fftfreq(self.ny, self.dy))*2.*np.pi )
		else:
			return np.meshgrid( np.fft.fftfreq(self.nx, self.dx)*2.*np.pi, np.fft.fftfreq(self.ny, self.dy)*2.*np.pi )


	def GetL(self, shift=True):
		""" 
		Returns a grid with the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode in the map. 
		If shift=True (default), \ell = 0 is centered in the grid
		"""
		lx, ly = self.GetLxLy(shift=shift)
		return np.sqrt(lx**2 + ly**2)


	def Compatible(self, other):	
		return ( (self.nx == other.nx) and
			     (self.ny == other.ny) and
				 (self.dx == other.dx) and
				 (self.dy == other.dy) )

class FlatMapReal(Pix):
	"""
	Class to store a scalar (real) 2D flat map. 
	~ Note: If 2d array (map) is given as input, it overwrites nx and ny.
	~ TODO: implement coordinates frame through WCS 

	Params
	------
	- nx:   # of pixels in x dir
	- dx:   pixel resolution in *arcmin* 
	- map:  2d array containing the field
	- mask: 2d array containing the mask 
	"""
	def __init__(self, nx, dx, map=None, mask=None, ny=None, dy=None):
		""" class which contains a real-valued map """
		super(FlatMapReal, self ).__init__(nx, dx, ny=ny, dy=dy)

		if map is None:
			self.map = np.zeros( (self.ny, self.nx) )
		else:
			self.map = map
		assert( (self.ny, self.nx) == self.map.shape )

		if mask is None:
			self.mask = np.ones( (self.ny, self.nx) )
		else:
			self.mask = mask
			assert( (self.ny, self.nx) == self.mask.shape)

	def Copy(self):
		return FlatMapReal( self.nx, self.dx, self.map.copy(), mask=self.mask.copy() , ny = self.ny, dy = self.dy )

	def ApplyGaussBeam(self, fwhm, order=0):
		"""
		Smooth the map with a Gaussian beam specified by its FWHM (in arcmin).
		"""
		self.map = GaussSmooth(self.map, fwhm, self.dx * rad2arcmin , order=order)

	def Pad(self, padX, padY=None, fwhm_mask=None, apo_mask=True):
		""" 
		Zero-pad the present map and creates a new one with dimensions nxp (>nx), nyp (>ny) 
		with this map at its center. 
		- fwhm: 
		"""
		assert( padX > 1 )
		if padY == None:
			padY = padX
		assert( padY > 1 )
		# assert( np.mod( nxp - self.nx, 2 ) == 0 )
		# assert( np.mod( nyp - self.ny, 2 ) == 0 )
		assert( np.mod( padX, 2 ) == 0 )
		assert( np.mod( padY, 2 ) == 0 )

		nxp = int(padX*self.nx)
		nyp = int(padY*self.ny)

		new = FlatMapReal( nx=nxp, dx=self.dx * rad2arcmin, ny=nyp, dy=self.dy * rad2arcmin ) # FIX ME: Multiply by rad2arcmin because of line 16/17 
		new.map[ (nyp-self.ny)/2:(nyp+self.ny)/2, (nxp-self.nx)/2:(nxp+self.nx)/2 ]  = self.map
		new.mask[:] = 0.
		new.mask[ (nyp-self.ny)/2:(nyp+self.ny)/2, (nxp-self.nx)/2:(nxp+self.nx)/2 ] = self.mask
		if fwhm_mask is not None:
			new.mask = GaussSmooth(new.mask, fwhm_mask, self.dx * rad2arcmin)
		if apo_mask:
			new.mask = smooth_window(new.mask)

		return new

	def Extract(self, npix):
		""" 
		Select a subpatch of the map:

		~ TODO: ropagate extaction onto the mask

		Params
		------
		- if len(npix) == 4 => vertices of the patch to extract, i.e. npix = [xmin, xmax, ymin, ymax] 
		- if len(npix) == 2 => size of patch size, i.e. npix = [nxp, nyp]
		"""
		assert( (len(npix) == 4) or (len(npix) == 2) )
		
		if len(npix) == 4:
			nxmin, nxmax, nymin, nymax = npix
			assert( 0 < nxmin < nxmax < self.nx )
			assert( 0 < nymin < nymax < self.ny )

			cut = FlatMapReal( nx=nxmax-nxmin, dx=self.dx*rad2arcmin, ny=nymax-nymin, dy=self.dy*rad2arcmin )
			cut.map  = self.map[nymin:nymax, nxmin:nxmax]
			cut.mask = self.mask[nymin:nymax, nxmin:nxmax]

		if len(npix) == 2:
			nxp = npix[0]
			nyp = npix[1]
			assert( nxp < self.nx )
			if nyp == None:
				nyp = nxp
			assert( nyp < self.ny )
			assert( np.mod( nxp - self.nx, 2 ) == 0 )
			assert( np.mod( nyp - self.ny, 2 ) == 0 )

			cut = FlatMapReal( nx=nxp, dx=self.dx*rad2arcmin, ny=nyp, dy=self.dy*rad2arcmin )
			cut.map  = self.map[ (self.ny)/2.-nyp:(self.ny)/2.+nyp, (self.nx)/2.-nxp:(self.nx)/2.+nxp ]
			cut.mask = self.mask[ (self.ny)/2.-nyp:(self.ny)/2.+nyp, (self.nx)/2.-nxp:(self.nx)/2.+nxp ]
		return cut

	def Display(self, title=None, cbu=None, xu=None, yu=None, cmap='inferno'):
		"""
		Plot the flat-sky map.

		Params
		------
		- Strings containing title and units
		"""
		mask = self.mask.copy()
		mask[mask == 0.] = np.nan

		fig = plt.figure()
		ax  = fig.add_subplot(111)
		cax = plt.imshow(self.map*mask, interpolation='nearest', cmap=cmap)
		cb = fig.colorbar(cax)
		if title is not None: fig.set_title(title)
		if cbu   is not None: cb.set_label(cbu)
		if xu    is not None: plt.xlabel(xu)
		if yu    is not None: plt.ylabel(yu)
		plt.show()

	def FilterMap(self, filt, padX=2, array=True, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None):
		if padX != 1:
			if np.all(self.mask != np.ones(self.shape)):
				newmap = self.Pad(padX, apo_mask=False)
			else:
				newmap = self.Pad(padX, apo_mask=True)
		else:
			newmap = self.Copy()

		if callable(filt):
			L = newmap.GetL(shift=False)
			filt = filt(L) 
		else:
			assert (filt.shape == newmap.map.shape)

		# FT = FlatMapFFT(newmap.map.shape[0], newmap.dx*rad2arcmin, map=newmap)
		FT = FlatMapFFT(newmap.map.shape[0], newmap.dx*rad2arcmin, ft=np.fft.fft2(newmap.map))

		filtmap = FT.FT2Map(filt=filt, array=array, lmin=lmin, lmax=lmax, lxmin=lxmin, lxmax=lxmax, lymin=lymin, lymax=lymax)
		# filtmap.mask = self.mask

		if array:
			if padX != 1:
				filtmap = filtmap[self.nx/2.*(padX-1):self.nx/2.*(padX+1.),self.nx/2.*(padX-1):self.nx/2.*(padX+1.)]
		else:
			if padX	!= 1:
				filtmap = filtmap.Extract([int(self.nx/2.*(padX-1.)),int(self.nx/2.*(padX+1.)),int(self.nx/2.*(padX-1.)),int(self.nx/2.*(padX+1.))])

		return filtmap

class FlatMapFFT(Pix):
	"""
	Class to store a (complex) 2D Fourier Transform (FT) and to perfom 
	calculations on it such as the auto- and cross-power spectrum estimation. 
	It can be initialized by passing a 2dFT *OR* a map.

	~ Note: If 2d array (map) is given as input, it overwrites nx and ny, 
	  and calculates the 2dFT *without shifting it*

	Params
	------
	- nx:   # of pixels in x dir
	- dx:   pixel resolution in *arcmin* 
	- ft:   2d (complex) array containing FT of the map
	- map:  FlatMapReal object containing the field
	"""
	def __init__(self, nx, dx, ft=None, map=None, ny=None, dy=None):
		super(FlatMapFFT, self).__init__(nx, dx, ny=ny, dy=dy)

		# FFT norm factor (no fsky correction included)
		self.tfac = (self.dx * self.dy)/(self.nx * self.ny)

		if ft is None:
			if map is None:
				self.ft  = np.zeros((self.ny, self.nx), dtype=np.complex)
				self.map = None
			else:
				assert( self.Compatible(map) ) 
				self.map = map
				self.ft  = np.fft.fft2(map.map*map.mask)#/np.mean(map.mask**2) 
		else:
			self.ft  = ft
			self.map = self.FT2Map(ft=self.ft)
			# What about the mask?

		assert( (self.ny, self.nx) == self.ft.shape )

	def GetPixTransf(self):
		""" return the FFT describing the map-level transfer function for the pixelization of this object. """
		lx, ly = self.GetLxLy()

		fftp        = np.zeros( (self.ny, self.nx) )
		fftp[0,  0] = 1.0
		fftp[0, 1:] = np.sin(self.dx*lx[ 0,1:]/2.) / (self.dx * lx[0,1:] / 2.)
		fftp[1:, 0] = np.sin(self.dy*ly[1:, 0]/2.) / (self.dy * ly[1:,0] / 2.)
		fftp[1:,1:] = np.sin(self.dx*lx[1:,1:]/2.) * np.sin(self.dy*ly[1:,1:]/2.) / (self.dx * self.dy * lx[1:,1:] * ly[1:,1:] / 4.)

		return fftp

	def GetLMask(self, shift=False, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None):
		""" 
		return a Fourier mask for the pixelization associated with this object which is zero over customizable ranges of L. 
		"""
		mask      = np.ones((self.ny, self.nx), dtype=np.complex)
		lx, ly    = self.GetLxLy(shift=shift)
		L         = self.GetL(shift=shift)
		if lmin  != None: mask[ np.where(L < lmin) ] = 0.0
		if lmax  != None: mask[ np.where(L >=lmax) ] = 0.0
		if lxmin != None: mask[ np.where(np.abs(lx) < lxmin) ] = 0.0
		if lymin != None: mask[ np.where(np.abs(ly) < lymin) ] = 0.0
		if lxmax != None: mask[ np.where(np.abs(lx) >=lxmax) ] = 0.0
		if lymax != None: mask[ np.where(np.abs(ly) >=lymax) ] = 0.0
		return mask

	def FT2Map(self, ft=None, filt=None, array=True, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None):
		""" 
		Returns the FlatMapReal associated to this FT. 
		- array: return just the 2d array containing the map?
		"""
		if ft is None:
			ft = self.ft # FT not shifted

		if filt is None:
			filt = np.ones(ft.shape, dtype=np.complex)
		else:
			filt = filt.astype(np.complex)

		mask = self.GetLMask(lmin=lmin, lmax=lmax, lxmin=lxmin, lxmax=lxmax, lymin=lymin, lymax=lymax)
		filt *= mask.astype(filt.dtype) 

		if array:
			return np.fft.ifft2(ft*filt).real
		else:
			return FlatMapReal(self.nx, self.dx*rad2arcmin, map=np.fft.ifft2(ft*filt).real, ny=self.ny, dy=self.dy*rad2arcmin)

	def FilterFT(self, filt, ft=None, array=True):
		"""
		Filter FT 
		*Filter should not be fftshifted*
		- array: return just the 2d array containing the filtered FT?
		"""
		if ft is None:
			ft = self.ft # FT not shifted

		assert (filt.shape == ft.shape)

		if array:
			return ft * filt
		else:
			return FlatMapFFT(self.nx, self.dx, ft=ft*filt, ny=self.ny, dy=self.dy)

	def Get2DSpectra(self, map1=None, map2=None, plot2D=False, shift=True):
		""" 
		Returns the (centered) 2d FT of a given map.

		Params
		------
		- map1: FlatMapReal object

		~ Note: if no maps is given as input, it computes the 2d FT of stored map.
		
		"""
		if map1 is None:
			map1 = self.map

		assert( self.Compatible(map1) )
	
		if map2 is None:
			map2 = map1
		# else:
			assert( self.Compatible(map2) )

		# Creating common mask 
		mask = map1.mask * map2.mask
		fsky = np.mean(mask**2.) # ! Not to be confused with the sky coverage of the *patch* given by Pix object, i.e. self.fsky

		fft1 = np.fft.fftshift(np.fft.fft2(map1.map*mask))
		fft2 = np.fft.fftshift(np.fft.fft2(map2.map*mask))
		fft_ = (fft1 * np.conj(fft2)).real * self.tfac / fsky

		if plot2D:
			# plt.subplot(121)
			lx, ly = self.GetLxLy()
			plt.imshow(np.log10(np.abs(fft_)), cmap='viridis')
			plt.colorbar()
			plt.title(r'$\log_{10}{|FT|}$', extent=[lx.min(),lx.max(),ly.min(),ly.max()])
			# plt.subplot(122)
			# plt.imshow(np.abs(fft_))
			# plt.colorbar()
			# plt.title(r'$|FFT|$')
			plt.show()

		if shift:
			return fft_
		else:
			return np.fft.fftshift(fft_)

	def Bin2DSpectra(self, ft=None, lbins=None, delta_ell=None, prefact=None, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None):
		""" 
		Bins a 2D power spectrum by azimuthally averaging it and returns 1D power spectrum 
		- lmin, lmax, lxmax, lxmin, ... filtering in lspace
		"""
		if ft is None:
			ft = self.ft

		L = self.GetL()

		# Setup the bins if not provided
		if lbins is None:
			lmin_ = int(np.ceil(2 * np.pi / np.sqrt(self.dx * self.nx * self.ny * self.dy))) # Set by patch size
			#lmax = int(np.ceil(np.pi / self.dx)) # ! Just an approx: assumes same-size pixels 
			if delta_ell is None:
				delta_ell = 2 * lmin # Minimun bins bandwith: suggested in PoKER paper (astro-ph:1111.0766v1)
			nbins    = int(np.ceil(np.max(L / delta_ell))) # number of bins
			ell_bins = np.asarray([ [delta_ell*i+lmin_, delta_ell*(i+1)+lmin_] for i in xrange(nbins)]) 
			lbins    = np.append(ell_bins[:,0], [ell_bins[-1,1]]) # bins edges
		else:
			nbins    = len(lbins) - 1
			ell_bins = np.asarray([[lbins[i], lbins[i+1]] for i in xrange(nbins)])

		# Flattening factors
		if prefact == None:
		    prefact = np.ones(L.shape)
		else:
		    prefact = prefact(L)

		ell_mask = self.GetLMask(lmin=lmin, lmax=lmax, lxmin=lxmin, lxmax=lxmax, lymin=lymin, lymax=lymax)

		av_cls, bins     = np.histogram(L, bins=lbins, weights=ft*prefact*ell_mask)
		av_weights, bins = np.histogram(L, bins=lbins, weights=ell_mask)

		# Azimuthal-average of 2D spectrum
		cl  = np.nan_to_num(av_cls / av_weights)
		lb  = np.mean(ell_bins, axis=1)

		return lb, cl.real

	def GetCl(self, map1=None, map2=None, lbins=None, delta_ell=None, prefact=None, plot2D=False, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None):
		"""
		Returns 1D power spectrum.
		"""
		if map1 is None:
			map1 = self.map

		fft2d  = self.Get2DSpectra(map1=map1, map2=map2, plot2D=plot2D)
		lb, cl = self.Bin2DSpectra(fft2d, lbins=lbins, delta_ell=delta_ell, prefact=prefact, lmin=lmin, lmax=lmax, lxmin=lxmin, lxmax=lxmax, lymin=lymin, lymax=lymax)

		return lb, cl

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

def GetMll(mask, reso, lbins=None, delta_ell=None, npts=5000):
	"""
	Returns the mode-coupling matrix from MASTER paper (astro-ph/0105302).
	Here k-vectors are already converted into l-vectors, while the paper uses wavenumber, hence the differences.
	The integral over the mask power spectrum is computed by means of Chebyschev-Gauss quadrature.
	"""
	assert ( len(mask.shape) == 2 )

	if lbins is None:
		if delta_ell is None:
			delta_ell = 10					
		nbins    = int(np.ceil(np.max(4000/ delta_ell))) # number of bins
		ell_bins = np.asarray([ [delta_ell*i, delta_ell*(i+1)] for i in xrange(nbins)]) 
		lbins    = np.append(ell_bins[:,0], [ell_bins[-1,1]]) # bins edges
	else:
		nbins    = len(lbins) - 1
		ell_bins = np.asarray([[lbins[i], lbins[i+1]] for i in xrange(nbins)])

	m_ = FlatMapReal(mask.shape[1], reso, ny=mask.shape[0], dy=reso, map=mask)
	m  = FlatMapFFT(mask.shape[1], reso, ny=mask.shape[0], dy=reso, map=m_)

	# Calculating mask 1D power spectrum W(l)
	l3, Wl = m.GetCl(lbins=lbins)

	l1 = np.mean(ell_bins,axis=-1)
	l2 = np.mean(ell_bins,axis=-1)

	M_out = np.empty([l1.size,l2.size])

	# CG quadrature points & weights
	wi  = np.pi / npts
	tmp = (2.*(np.arange(1,npts+1)-1)) / (2.*npts) * np.pi
	vi  = np.cos(tmp)

	for i in xrange(l1.size):
		for j in xrange(l2.size):
			v    = (l3**2 - l1[i]**2 - l2[j]**2) / (2.0*l1[i]*l2[j])
			fv   = Wl
			fvi  = np.interp(vi,v,fv, left=0., right=0.)
			tmp2 = np.sum(wi*fvi)

			# TODO: should I multiply for \Delta k_2??
			M_out[i,j] = l2[j]/(2.*np.pi)  * tmp2 

	return np.nan_to_num(M_out)