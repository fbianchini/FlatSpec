import numpy as np
import scipy.signal

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
