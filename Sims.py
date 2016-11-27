import numpy as np
from Spec2D import *
from IPython import embed

def GenCorrFlatMaps(cls, nx, dx, ny=None, dy=None, buff=1, seed=None):
	"""
	Routine to generate Gaussian correlated random realizations
	
	Params
	------
	- cls: array with spectra. If cls is a vector the routine ouputs a single map,
		   if cls contains 3 vectors (clxx,clyy,clxy) so that cls.shape = (lmax,3)
		   the routine outputs two correlated realizations
	- nx: # of pixels x-direction
	- dx: pixel resolution in arcmin
	- buffer: 
	- seed:  
	
	Note
	----
	cls must be input from l = 0
 	"""
	if len(cls.shape) == 1: # Just auto-correlation
		clxx = cls
		corr = False
	elif len(cls.shape) == 2: # Or cross-correlation
		assert(cls.shape[1] == 3)
		clxx = cls[:,0]
		clyy = cls[:,1]
		clxy = cls[:,2]
		corr = True
	else:
		print('Input spectra array shape is not valid!')
		sys.exit()

	ls = np.arange(len(cls))

	Nx = int(nx * buff)
	if ny is None:
		ny = nx
	Ny = int(ny * buff)

	# Assuming all pixels equal !
	pix    = Pix(Nx, dx, ny=Ny, dy=dy) 
	npix   = pix.npix
	shape  = pix.shape
	lx, ly = pix.GetLxLy(shift=False)
	L      = pix.GetL(shift=False)
	idx    = np.where(L > ls.max())

	xx = np.interp(L, ls, clxx)
	xx =  xx / pix.area * (pix.nx*pix.ny)**2
	xx[idx] = 0.
	
	if corr:
		xy  = np.interp(L, ls, clxy)
		yy  = np.interp(L, ls, clyy)
		xy  = xy / pix.area * (pix.nx*pix.ny)**2
		yy  = yy / pix.area * (pix.nx*pix.ny)**2
		xy[idx] = 0.
		yy[idx] = 0.

	# Injecting the correlation degree
	A = np.sqrt(xx)
	if corr:
		B  = np.nan_to_num(xy/A)
		C_ = yy - xy*xy/xx
		if np.any(C_ < 0.0):
			print "Could not calculate XX,XY,YY covariance square root - YY > XY^2/XX was not true at all l"
			sys.exit()
		C = np.nan_to_num(np.sqrt(C_))
	
	if seed == None:
		np.random.seed(0)
	else:
		np.random.seed(seed)

	# Generating random Fourier maps 
	u = np.random.normal(size=npix).reshape(shape)+1.0j*np.random.normal(size=npix).reshape(shape)
	if corr:
		v = np.random.normal(size=npix).reshape(shape)+1.0j*np.random.normal(size=npix).reshape(shape)

	kMapX = A*u
	if corr:
		kMapY = B*u + C*v


	if corr:
		mapX = (np.fft.ifft2(kMapX)).real
		mapY = (np.fft.ifft2(kMapY)).real
		mapX = mapX[(buff-1)/2.*ny:(buff+1)/2.*ny,(buff-1)/2.*nx:(buff+1)/2.*nx]
		mapY = mapY[(buff-1)/2.*ny:(buff+1)/2.*ny,(buff-1)/2.*nx:(buff+1)/2.*nx]
		return mapX, mapY
	else:
		mapX = (np.fft.ifft2(kMapX)).real
		# plt.subplot(121)
		# plt.imshow(mapX)
		# plt.colorbar()
		mapX = mapX[(buff-1)/2*ny:(buff+1)/2*ny,(buff-1)/2*nx:(buff+1)/2*nx]
		# plt.subplot(122)
		# plt.imshow(mapX)
		# plt.colorbar()
		# plt.show()
		return mapX

def GenPoissNoise(mean, nx, dx, ny=None, dy=None, dim='pix'):
	"""
	Routine to generate Poisson noise maps given the average rate of observations.
	
	Params
	------
	- mean: parameter \lambda of Poissonian process (i.e. avg galaxies #, photon counts $ ,...)
	- nx: # of pixels x-direction
	- dx: pixel resolution in arcmin
	- dim: units of mean param, i.e. mean/pixel (can be 'pixel', 'steradian' or 'arcmin')
 	"""
 	pix    = Pix(nx, dx, ny=ny, dy=dy) 
 	npix   = pix.npix
 	shape  = pix.shape

 	# Converting to avg per pixel
 	if dim == 'pix':
 		pass
 	if dim == 'ster':
 	    mean = mean * pix.dx * pix.dy
 	elif dim == 'arcmin':
 	    mean = mean * (pix.dx * pix.dy) / (np.pi / 180. / 60.)

 	# Creating the noise map
 	noise_map = np.random.poisson(lam=mean, size=npix).reshape(shape)
 	
 	# Converting it to contrast map
 	# if delta:
 	#     noise_bar = np.mean(noise_map)
 	#     delta_map = (noise_map - noise_bar)/noise_bar
 	#     delta_map[noise_map < -1.] = -1.
 	return noise_map*1.

def GetCountsTot(mapgg, mean, nx, dx, ny=None, dy=None, dim='pix'):
	"""
	Routine to generate total galaxy counts maps given the average # of galaxies and the signal overdensity. 
	
	Params
	------
	- mapgg: map containg clustering signal of galaxy *overdensity*
	- mean: avg galaxies #
	- nx: # of pixels x-direction
	- dx: pixel resolution in arcmin
	- dim: units of mean param, i.e. mean/pixel (can be 'pixel', 'steradian' or 'arcmin')
 	"""
 	pix    = Pix(nx, dx, ny=ny, dy=dy) 
 	npix   = pix.npix
 	shape  = pix.shape

 	# Converting to avg per pixel
 	if dim == 'pix':
 		pass
 	if dim == 'ster':
 	    mean = mean * pix.dx * pix.dy
 	elif dim == 'arcmin':
 	    mean = mean * (pix.dx * pix.dy) / (np.pi / 180. / 60.)

 	return np.random.poisson(lam=mean*(1.+mapgg.flatten())).reshape(shape)

