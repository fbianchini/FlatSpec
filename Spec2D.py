import numpy as np
import sys
from scipy import ndimage
import matplotlib.pyplot as plt
from Utils import *

# Here you can find some classes to deal with flat-sky scalar maps 
# (like temperature, lensing, galaxies,..) and their FFTs.
# The following code is heavily inspired from some of Duncan's quicklens code 
# that can be found at https://github.com/dhanson/quicklens


# arcmin2rad = np.pi / 180. / 60. 

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
		dx *= np.pi / 180. / 60. 
		dy *= np.pi / 180. / 60. 

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

	def ApplyGaussBeam(self, fwhm, order=0):
		"""
		Smooth the map with a Gaussian beam specified by its FWHM (in arcmin).
		"""
		self.map = GaussSmooth(self.map, fwhm, self.dx * 180.*60./np.pi , order=order)


	def Pad(self, nxp, nyp=None, fwhm_mask=None, apo_mask=True):
		""" 
		Zero-pad the present map and creates a new one with dimensions nxp (>nx), nyp (>ny) 
		with this map at its center. 
		- fwhm: 
		"""
		assert( nxp > self.nx )
		if nyp == None:
			nyp = nxp
		assert( nyp > self.ny )
		assert( np.mod( nxp - self.nx, 2 ) == 0 )
		assert( np.mod( nyp - self.ny, 2 ) == 0 )

		new = FlatMapReal( nx=nxp, dx=self.dx * 180 * 60 / np.pi, ny=nyp, dy=self.dy * 180 * 60 /np.pi ) # FIX ME: Multiply by 180*60/pi because of line 16/17 
		new.map[ (nyp-self.ny)/2:(nyp+self.ny)/2, (nxp-self.nx)/2:(nxp+self.nx)/2 ]  = self.map
		new.mask[:] = 0.
		new.mask[ (nyp-self.ny)/2:(nyp+self.ny)/2, (nxp-self.nx)/2:(nxp+self.nx)/2 ] = self.mask
		if fwhm_mask is not None:
			new.mask = GaussSmooth(new.mask, fwhm_mask, self.dx * 180.*60./np.pi)
		if apo_mask:
			new.mask = smooth_window(new.mask)

		return new

	def Extract(self, np):
		""" 
		Select a subpatch of the map:

		~ TODO: ropagate extaction onto the mask

		Params
		------
		- if len(np) == 4 => vertices of the patch to extract, i.e. np = [xmin, xmax, ymin, ymax] 
		- if len(np) == 2 => size of patch size, i.e. np = [nxp, nyp]
		"""
		assert( (len(np) == 4) or (len(np) == 2) )
		
		if len(np) == 4:
			nxmin, nxmax, nymin, nymax = np
			assert( 0 < nxmin < nxmax < self.nx )
			assert( 0 < nymin < nymax < self.ny )

			cut = FlatMapReal( nx=nxmax-nxmin, dx=self.dx, ny=nymax-nymin, dy=self.dy )
			cut.map = self.map[nymin:nymax, nxmin:nxmax]

		if len(np) == 2:
			assert( nxp < self.nx )
			if nyp == None:
				nyp = nxp
			assert( nyp < self.ny )
			assert( np.mod( nxp - self.nx, 2 ) == 0 )
			assert( np.mod( nyp - self.ny, 2 ) == 0 )

			cut = FlatMapReal( nx=nxp, dx=self.dx, ny=nyp, dy=self.dy )
			cut.map[ (nyp-self.ny)/2:(nyp+self.ny)/2, (nxp-self.nx)/2:(nxp+self.nx)/2 ] = self.map
		
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
	- map:  2d array containing the field
	"""
	def __init__(self, nx, dx, ft=None, map=None, ny=None, dy=None):
		super(FlatMapFFT, self).__init__(nx, dx, ny=ny, dy=dy)

		# FFT norm factor (no fsky correction included)
		self.tfac = (self.dx * self.dy)/(self.nx * self.ny)

		if ft is None:
			if map is None:
				ft       = np.zeros((self.ny, self.nx), dtype=np.complex)
				self.map = None
			else:
				assert( self.Compatible(map) ) 
				self.map = map
				self.ft  = np.fft.fft2(map.map)
		else:
			self.ft  = ft
			self.map = self.GetMapReal(shift=True)

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

	def GetLMask(self, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None):
		""" 
		return a Fourier mask for the pixelization associated with this object which is zero over customizable ranges of L. 
		"""
		mask      = np.ones((self.ny, self.nx), dtype=np.complex)
		lx, ly    = self.GetLxLy()
		L         = self.GetL()
		if lmin  != None: mask[ np.where(L < lmin) ] = 0.0
		if lmax  != None: mask[ np.where(L >=lmax) ] = 0.0
		if lxmin != None: mask[ np.where(np.abs(lx) < lxmin) ] = 0.0
		if lymin != None: mask[ np.where(np.abs(ly) < lymin) ] = 0.0
		if lxmax != None: mask[ np.where(np.abs(lx) >=lxmax) ] = 0.0
		if lymax != None: mask[ np.where(np.abs(ly) >=lymax) ] = 0.0
		return mask

	def GetMapReal(self, ft=None, shift=False):
		""" 
		Returns the FlatMapReal associated to this FT. 
		"""
		if ft is None:
			ft = self.ft

		# assert( self.Compatible(ft) )
		if shift:
			return FlatMapReal(self.nx, self.dx, map=np.fft.ifft2(np.fft.fftshift(ft)).real, ny=self.ny, dy=self.dy)
		else:
			return FlatMapReal(self.nx, self.dx, map=np.fft.ifft2(ft).real, ny=self.ny, dy=self.dy)

	def Get2DSpectra(self, map1=None, map2=None, plot2D=False):
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
		else:
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

		return fft_


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
