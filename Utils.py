import numpy as np
import scipy.signal

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
