import numpy as np
import scipy.signal
import scipy.interpolate

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

def GetBins(lbins=None, delta_ell=None, lmin=None, lmax=None):
	"""
	Returns binning scheme by providing either:
	i)  list w/ bins edges or;
	ii) delta_ell and/or {lmin, lmax} 

	Outputs
	-------
	i)   ell_bins : list of bins
	ii)  lbins    : bins_edges
	iii) nbins    : # of bins
	"""
	if lbins is None:
		if lmin is None:
			lmin = 0
		if (delta_ell is None) and (lmin != 0):
			delta_ell = 2 * lmin # Minimun bins bandwith: suggested in PoKER paper (astro-ph:1111.0766v1)
		else:
			delta_ell = 1.
		nbins    = int(np.ceil(np.max(lmax / delta_ell))) # number of bins
		ell_bins = np.asarray([ [delta_ell*i+lmin, delta_ell*(i+1)+lmin] for i in xrange(nbins)]) 
		lbins    = np.append(ell_bins[:,0], [ell_bins[-1,1]]) # bins edges
	else:
		nbins    = len(lbins) - 1
		ell_bins = np.asarray([[lbins[i], lbins[i+1]] for i in xrange(nbins)])

	return ell_bins, lbins, nbins

def GetLxLy(nx, dx, ny=None, dy=None, shift=False):
    """ 
    Returns two grids with the (lx, ly) pair associated with each Fourier mode in the map. 
    If shift=True , \ell = 0 is centered in the grid
    ~ Note: already multiplied by 2\pi 
    """
    if ny is None: ny = nx
    if dy is None: dy = dx
    
    dx *= arcmin2rad
    dy *= arcmin2rad
    
    if shift:
        return np.meshgrid( np.fft.fftshift(np.fft.fftfreq(nx, dx))*2.*np.pi, np.fft.fftshift(np.fft.fftfreq(ny, dy))*2.*np.pi )
    else:
        return np.meshgrid( np.fft.fftfreq(nx, dx)*2.*np.pi, np.fft.fftfreq(ny, dy)*2.*np.pi )

def GetL(nx, dx, ny=None, dy=None, shift=False):
    """ 
    Returns a grid with the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode in the map. 
    If shift=True, \ell = 0 is centered in the grid
    """
    lx, ly = GetLxLy(nx, dx, ny=ny, dy=dy, shift=shift)
    return np.sqrt(lx**2 + ly**2)

def GetLxLyAngle(nx, dx, ny=None, dy=None, shift=False):
	""" 
	Returns a grid with the angle between Fourier mode in the map. 
	If shift=True (default), \ell = 0 is centered in the grid
	~ Note: already multiplied by 2\pi 
	"""
	lx, ly = GetLxLy(nx, dx, ny=ny, dy=dy, shift=shift)
	return 2*np.arctan2(lx, -ly)

def Interpolate2D(nx, dx, l, cl, dy=None, ny=None, shift=False):
    """ 
    Returns a function cl interpolated on the 2D L plane.
    If shift=True (default), \ell = 0 is centered in the grid
    """
    if ny is None: ny = nx
    if dy is None: dy = dx
 
    L = GetL(nx, dx, ny=ny, dy=dy, shift=shift)
    CL = np.interp(L, l, cl)

    return CL

def EB2QU(E, B, dx):
	# E and B are 2D arrays of same shape containing E and B maps
	assert E.shape == B.shape
	nx = E.shape[0]
	ny = E.shape[1]
	angle = GetLxLyAngle(nx, dx, ny=ny, dy=dx, shift=False)

	efft = np.fft.fft2(E)
	bfft = np.fft.fft2(B)

	Q = np.fft.ifft2( np.cos(angle) * efft - np.sin(angle) * bfft ).real
	U = np.fft.ifft2( np.sin(angle) * efft + np.cos(angle) * bfft ).real

	return Q, U

def QU2EB(Q, U, dx):
	# Q and U are 2D arrays of same shape containing Q and U maps
	assert Q.shape == U.shape
	nx = Q.shape[0]
	ny = U.shape[1]
	angle = GetLxLyAngle(nx, dx, ny=ny, dy=dx, shift=False)

	qfft = np.fft.fft2(Q)
	ufft = np.fft.fft2(U)

	E = np.fft.ifft2( np.cos(angle) * qfft + np.sin(angle) * ufft ).real
	B = np.fft.ifft2(-np.sin(angle) * qfft + np.cos(angle) * ufft ).real

	return E, B


def LensMe( mymap, phimap, psi=0.0 ):
    """ perform the remapping operation of lensing in the flat-sky approximation.
         tqumap         = unlensed tqumap object to sample from.
         phifft         = phi field to calculate the deflection d=\grad\phi from.
         (optional) psi = angle to rotate the deflection field by, in radians (e.g. psi=pi/2 results in phi being treated as a curl potential).
    """
    assert( mymap.Compatible(phimap) )

    nx, ny = phimap.nx, phimap.ny
    dx, dy = phimap.dx, phimap.dy

    # lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx )[0:nx/2+1]*2.*np.pi, np.fft.fftfreq( ny, dy )*2.*np.pi )
    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx )*2.*np.pi, np.fft.fftfreq( ny, dy )*2.*np.pi )

    pfft   = np.fft.fft2(phimap.map)

    # deflection field
    x, y   = np.meshgrid( np.arange(0,nx)*dx, np.arange(0,ny)*dy )
    gpx    = np.fft.ifft2( pfft * lx * -1.j)# * np.sqrt( (nx*ny)/(dx*dy) ) )
    gpy    = np.fft.ifft2( pfft * ly * -1.j)# * np.sqrt( (nx*ny)/(dx*dy) ) )

    if psi != 0.0:
        gp = (gpx + 1.j*gpy)*np.exp(1.j*psi)
        gpx = gp.real
        gpy = gp.imag

    lxs    = (x+gpx).flatten(); del x, gpx
    lys    = (y+gpy).flatten(); del y, gpy

    # interpolate
    mymap_lensed = scipy.interpolate.RectBivariateSpline( np.arange(0,ny)*dy, np.arange(0,nx)*dx, mymap.map ).ev(lys, lxs).reshape([ny,nx])

    return mymap_lensed

def fn_apodize(MAP, mapparams, apodmask='circle', edge_apod=2, min_kernel_size=10):
	"""
	!! Srini's routine !!
	
	MAP = input map to be apodized
	mapparams = [nx, ny, dx, dy] (nx, ny is normally MAP.shape)
	mask = circlular / square
	edge_apod = how many arcmins to apod from the edge. You can play with this number
	"""

	import numpy as np, scipy.ndimage as ndimage

	nx, ny, dx, dy = mapparams

        radius = (nx * dx)/20
	if radius < min_kernel_size: 
		radius = min_kernel_size

        ker   = np.hanning(int(radius/dx))
        ker2d = np.asarray( np.sqrt(np.outer(ker,ker)) )

        MASKf=np.zeros((nx,ny))

	#the below grid is in arcmins - note the dx factor
	minval, maxval = -(nx*dx)/2,  (nx*dx)/2
	x = y = np.linspace(minval, maxval, nx)
	X, Y = np.meshgrid(x,y)
	xc, yc = 0., 0.

	if apodmask == 'circle':
		radius = (nx * dx/2) - edge_apod
		inds=np.where((X-xc)**2. + (Y-yc)**2. <= radius**2.) #all in arcmins

	elif apodmask == 'square':
		radius = (nx * dx/2) - edge_apod
	        inds=np.where((abs(X)<=radius) & (abs(Y)<=radius)) #all in arcmins

        MASKf[inds]=1.

	apodMASKf=ndimage.convolve(MASKf, ker2d)#, mode='wrap')
	apodMASKf/=apodMASKf.max()

	#imshow(apodMASKf);colorbar();show();quit()

	return apodMASKf * MAP

def fn_psource_mask(MAP, mapparams, coords, disc_rad = None, min_kernel_size = 10):
	"""
	!! Srini's routine !!

	MAP - input map
	mapparams = [nx, ny, dx, dy] (nx, ny is normally MAP.shape)
	coords - (x,y) pixel coords
	disc_rad - x arcmins to apodize around each point source
		roughly disc_rad/dx will give you the right arcmins. 
	"""

	import numpy as np, scipy.ndimage as ndimage


	x, y = np.arange(np.shape(MAP)[1]), np.arange(np.shape(MAP)[0])
	X, Y = np.meshgrid(x,y)

	#create a mask full of ones and make holes at the location of point sources
	MASK = np.ones(MAP.shape)
	for (i,j) in coords:
		inds=np.where((X-i)**2. + (Y-j)**2. <= disc_rad**2.)
		MASK[inds]=0.

	#imshow(MASK);colorbar();show();quit()

	#create a hanning kernel
	nx, ny, dx, dy = mapparams
        radius = (nx * dx)/20
	if radius < min_kernel_size: 
		radius = min_kernel_size

	ker=np.hanning(int(radius)+1)
	ker2d=np.sqrt(np.outer(ker,ker))

	#imshow(ker2d);colorbar();show();quit()

	#convolve
	MASK=ndimage.convolve(MASK, ker2d)
	MASK/=np.max(MASK)

	#imshow(MASK);colorbar();show();quit()

	return MAP * MASK

def smooth_window(win,ker_size=16):
	'''Smooth input window so that it and its gradients vanish on the boundary region. This 
	is required for B_pure to be identically 0
	
	@type win: numpy.ndarray
	@param win: the window function
	@rtype: numpy.ndarray
	@return: the window function smoothed so that its value and its derivative go to 0 at the boundary'''
	#This is my attempt to do some sort of smoothing. Smoothing makes the gradient zero at the edges (hopefully), while 
	#keeping the mask zero at the edges of our data
	win_temp = 1.0*win

	for i in range(ker_size/2):
		win_temp[get_edge_pixels(win_temp)] = 0

	#ker = np.ones([ker_size,ker_size])

	k1 = np.hamming(ker_size)
	k2x,k2y = np.meshgrid(k1,k1)
	ker = k2x*k2y
	ker /= np.sum(ker)

	win_smooth = scipy.signal.fftconvolve(win_temp,ker,mode='same')
	return win_smooth

def get_edge_pixels(win):
	'''Gets edge pixels of a 2d image. Edges are defined as a non-zero pixel where any of its 8 neighbors is zero
	
	@type win: numpy.ndarray
	@param win: the window function
	@rtype: numpy.ndarray
	@return: array with a value of 1 at the edges of the window function
	'''
	#create mask (1 to where it was non-zero)
	mask = np.ones(win.shape)
	mask[win == 0] = 0

	#8 pixels that are next to each pixel
	offp1x = np.roll(mask,1,axis=0)
	offm1x = np.roll(mask,-1,axis=0)
	offp1y = np.roll(mask,1,axis=1)
	offm1y = np.roll(mask,-1,axis=1)
	offp1xp1y = np.roll(offp1x,1,axis=1)
	offp1xm1y = np.roll(offp1x,-1,axis=1)
	offm1xp1y = np.roll(offm1x,1,axis=1)
	offm1xm1y = np.roll(offm1x,-1,axis=1)

	#Pixels that are not next to the edges are 1 in all shifts so anding everything will return True for anything that
	#is not next to an edge and in the mask
	pix_non_edge = (offp1x>0) & (offm1x>0) & (offp1y>0) & (offm1y>0) & (offp1xp1y>0) & (offp1xm1y>0) & (offm1xp1y>0) & (offm1xm1y>0) & (mask>0)
	edge_pix = pix_non_edge != mask

	return edge_pix
