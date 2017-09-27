import numpy as np
import os, sys
import matplotlib.pyplot as plt
from Utils import *
from Sims import *
from Spec2D import *
from IPython import embed
from cosmojo.universe import Cosmo
from tqdm import tqdm
sys.path.append('../PBxHer/')
from XCroutines import *

import warnings
warnings.filterwarnings('ignore')

class Avg():
	def __init__(self):
		self.cls = [] 

	def add(self, cl):
		if len(self.cls) == 0:
			self.cls.append(cl)
		else: 
			assert( cl.shape == self.cls[0].shape)
			self.cls.append(cl)

	def mean(self, axis=0):
		return np.asarray(self.cls).mean(axis=axis)

	def errs(self):
		return np.sqrt(np.diag(np.cov(np.asarray(self.cls).T)/len(self.cls)))

# set simulation parameters.
nsims      = 20
lmax       = 3500
nx         = 271
reso       = 2. # arcmin
buff       = 1
one_lens   = False
seed       = 62347 # Change seed for every patch !!!!! 10029 --> RA12 !!!! 30419 --> RA23 
patch      = 'LST4P5'
calc_specs = True
write_maps = False 
rootname   = 'SimsForDavid_20170927/'

# PB stuff 
datapath = '/Users/fbianchini/Research/PBxHer/secondseason_gain1122/'

# Beam n transfer function
blsqrtfl_l, blsqrtfl_T, blsqrtfl_E, blsqrtfl_B = np.loadtxt(os.path.join(datapath,'blsqrtfl_'+patch.lower()+'.txt'), unpack=True)

T_mask = {} # Temperature mask
T_mask['RA12']   = file2array(os.path.join(datapath,'pb_I_mask_pb1ra12hab_3x3.txt')).real
T_mask['RA23']   = file2array(os.path.join(datapath,'pb_I_mask_pb1ra23hab_3x3.txt')).real
T_mask['LST4P5'] = file2array(os.path.join(datapath,'pb_I_mask_pb1lst4p5_3x3.txt')).real

P_mask = {} # Polarization mask
P_mask['RA12']   = file2array(os.path.join(datapath,'pb_P_mask_pb1ra12hab_3x3.txt')).real
P_mask['RA23']   = file2array(os.path.join(datapath,'pb_P_mask_pb1ra23hab_3x3.txt')).real
P_mask['LST4P5'] = file2array(os.path.join(datapath,'pb_P_mask_pb1lst4p5_3x3.txt')).real

ps_mask = {} # Point-source mask
ps_mask['RA12']   = file2array(os.path.join(datapath,'psRA12.txt')).real
ps_mask['RA23']   = file2array(os.path.join(datapath,'psRA23.txt')).real
ps_mask['LST4P5'] = file2array(os.path.join(datapath,'psLST4.5.txt')).real

T_depth = {} # Temperature depth
T_depth['RA12']   = file2array(os.path.join(datapath,'pb_t_depth_pb1ra12hab_3x3.txt')).real
T_depth['RA23']   = file2array(os.path.join(datapath,'pb_t_depth_pb1ra23hab_3x3.txt')).real
T_depth['LST4P5'] = file2array(os.path.join(datapath,'pb_t_depth_pb1lst4p5_3x3.txt')).real

P_depth = {} # Polarization depth
P_depth['RA12']   = file2array(os.path.join(datapath,'pb_p_depth_pb1ra12hab_3x3.txt')).real
P_depth['RA23']   = file2array(os.path.join(datapath,'pb_p_depth_pb1ra23hab_3x3.txt')).real
P_depth['LST4P5'] = file2array(os.path.join(datapath,'pb_p_depth_pb1lst4p5_3x3.txt')).real

if write_maps:
	if not os.path.exists(rootname):
		os.mkdir(rootname)
	if not os.path.exists(rootname+patch):
		os.mkdir(rootname+patch)
	if not os.path.exists(rootname+patch+'/PHI/'):
		os.mkdir(rootname+patch+'/PHI/')
	if not os.path.exists(rootname+patch+'/T/'):
		os.mkdir(rootname+patch+'/T/')
	if not os.path.exists(rootname+patch+'/Q/'):
		os.mkdir(rootname+patch+'/Q/')
	if not os.path.exists(rootname+patch+'/U/'):
		os.mkdir(rootname+patch+'/U/')

if calc_specs:
	cltt_len_avg = Avg()
	cltt_unl_avg = Avg()
	clee_len_avg = Avg()
	clee_unl_avg = Avg()
	clbb_len_avg = Avg()
	# cltt_unl_avg = Avg()

# Load power spectra (unlensed)
print("...loading theoretical un/lensed power spectra...")
cls_unl_camb = Cosmo().cmb_spectra(lmax, dl=False, spec='unlensed_scalar')
cls_len_camb = Cosmo().cmb_spectra(lmax, dl=False)#, spec='unlensed_scalar')
l = np.arange(lmax+1)
cls_unl = np.zeros((lmax+1,3))
cls_unl[:,0] = cls_unl_camb[:,0]
cls_unl[:,1] = cls_unl_camb[:,1]
cls_unl[:,2] = cls_unl_camb[:,3]

clkk = cls_unl_camb[:,4]
clpp = np.nan_to_num(4./(l*(l+1))**2 * clkk)
fact = l*(l+1)/2./np.pi
print("...done...")

# Initialize seed
np.random.seed(seed)

# How many kappa sims do we want?
if one_lens:
	print("...Generating *one* PHI realization to lens the CMB maps...")
	phi = FlatMapReal(nx, reso, map=GenCorrFlatMaps(clpp, nx, reso, buff=buff))
	print("...done...")
else:
	print("...%d PHI realizations will be generated together with CMB maps..." %nsims)

# Need this to calculate PS
FFT = FlatMapFFT(nx, reso)

# Loop over sims
print("...here we go!!!")
for i in tqdm(range(nsims)):

	# i) Generate T/E/B maps
	t_unl, e_unl = GenCorrFlatMaps(cls_unl, nx, reso, buff=buff) # T/E are correlated
	b_unl = np.zeros_like(t_unl) # Assume zero primordial power
	T_unl = FlatMapReal(nx, reso, map=t_unl)
	E_unl = FlatMapReal(nx, reso, map=e_unl)
	B_unl = FlatMapReal(nx, reso, map=b_unl)

	# ii) Convert E/B -> Q/U
	q_unl, u_unl = EB2QU(E_unl.map, B_unl.map, reso)
	Q_unl = FlatMapReal(nx, reso, map=q_unl)
	U_unl = FlatMapReal(nx, reso, map=u_unl)

	# iii) Lens T/Q/U maps
	if one_lens:
		T_len = FlatMapReal(nx, reso, map=LensMe(T_unl, phi)) 
		Q_len = FlatMapReal(nx, reso, map=LensMe(Q_unl, phi)) 
		U_len = FlatMapReal(nx, reso, map=LensMe(U_unl, phi)) 
	else:
		phi   = FlatMapReal(nx, reso, map=GenCorrFlatMaps(clpp, nx, reso, buff=buff))
		T_len = FlatMapReal(nx, reso, map=LensMe(T_unl, phi)) 
		Q_len = FlatMapReal(nx, reso, map=LensMe(Q_unl, phi)) 
		U_len = FlatMapReal(nx, reso, map=LensMe(U_unl, phi)) 
		# TODO: write phimap to file 
		if write_maps:
			array2file(phi.map, rootname+'/'+patch+'/PHI/pb_phi_%03d.txt'%i, SIZE=271)

	# iv) Convert lensed Q/U -> E/B
	e_len, b_len = QU2EB(Q_len.map, U_len.map, 2)
	E_len = FlatMapReal(nx, reso, e_len) 
	B_len = FlatMapReal(nx, reso, b_len) 

	# v) Filter for transfer function/beam
	T_len = T_len.FilterMap(np.vstack([blsqrtfl_l, blsqrtfl_T]), padX=2, array=False)
	E_len = E_len.FilterMap(np.vstack([blsqrtfl_l, blsqrtfl_E]), padX=2, array=False)
	B_len = B_len.FilterMap(np.vstack([blsqrtfl_l, blsqrtfl_B]), padX=2, array=False)

	# vii) Convert E/B -> Q/U
	q_len, u_len = EB2QU(E_len.map, B_len.map, reso)
	Q_len = FlatMapReal(nx, reso, map=q_len)
	U_len = FlatMapReal(nx, reso, map=u_len)

	# # vi) Add instrumental noise
	# noise_T = GenNoiseFromDepth(T_depth[patch])
	# noise_Q = GenNoiseFromDepth(P_depth[patch])
	# noise_U = GenNoiseFromDepth(P_depth[patch])

	"""
	noise_T[np.abs(noise_T) > 1e10] = 0.
	noise_Q[np.abs(noise_Q) > 1e10] = 0.
	noise_U[np.abs(noise_U) > 1e10] = 0.
	"""

	# T_len.map += noise_T 
	# Q_len.map += noise_Q
	# U_len.map += noise_U 

	# # viii) Mask the maps
	# T_len.mask = T_mask[patch]*ps_mask[patch]
	# Q_len.mask = P_mask[patch]*ps_mask[patch]
	# U_len.mask = P_mask[patch]*ps_mask[patch]

	# ix) Dump maps to file
	if write_maps:
		array2file(T_len.map, rootname+'/'+patch+'/T/pb_T_%03d.txt'%i, SIZE=271)
		array2file(Q_len.map, rootname+'/'+patch+'/Q/pb_Q_%03d.txt'%i, SIZE=271)
		array2file(U_len.map, rootname+'/'+patch+'/U/pb_U_%03d.txt'%i, SIZE=271)

	# E_len.map = QU2EB(Q_len.map*Q_len.mask, U_len.map*U_len.mask, 2)[0].copy()
	# E_len.mask = P_mask[patch]*ps_mask[patch]


	# plot the comparison between lensed/unlensed maps
	if 1 == 0:
		plt.subplot(3,3,1)
		plt.imshow(T_unl.map, vmin=-300, vmax=300)#; plt.colorbar()
		plt.subplot(3,3,2)
		plt.imshow(T_len.map, vmin=-300, vmax=300)#; plt.colorbar()
		plt.subplot(3,3,3)
		plt.imshow(T_len.map-T_unl.map, vmin=-100, vmax=100)#; plt.colorbar()
		plt.subplot(3,3,4)
		plt.imshow(Q_unl.map, vmin=-20, vmax=20)#; plt.colorbar()
		plt.subplot(3,3,5)
		plt.imshow(Q_len.map, vmin=-20, vmax=20)#; plt.colorbar()
		plt.subplot(3,3,6)
		plt.imshow(Q_len.map-Q_unl.map, vmin=-10, vmax=10)#; plt.colorbar()
		plt.subplot(3,3,7)
		plt.imshow(U_unl.map, vmin=-20, vmax=20)#; plt.colorbar()
		plt.subplot(3,3,8)
		plt.imshow(U_len.map, vmin=-20, vmax=20)#; plt.colorbar()
		plt.subplot(3,3,9)
		plt.imshow(U_len.map-U_unl.map, vmin=-10, vmax=10)#; plt.colorbar()
		plt.show()

	if 1 == 0:
		plt.subplot(3,2,1)
		plt.imshow(T_unl.map*Q_len.mask,); plt.colorbar()
		plt.subplot(3,2,2)
		plt.imshow(T_len.map*Q_len.mask,); plt.colorbar()
		plt.subplot(3,2,3)
		plt.imshow(Q_unl.map*Q_len.mask,); plt.colorbar()
		plt.subplot(3,2,4)
		plt.imshow(Q_len.map*Q_len.mask,); plt.colorbar()
		plt.subplot(3,2,5)
		plt.imshow(U_unl.map*U_len.mask,); plt.colorbar()
		plt.subplot(3,2,6)
		plt.imshow(U_len.map*U_len.mask,); plt.colorbar()
		plt.show()

	# calculate PS of lensed/unlensed maps
	if calc_specs:
		pref = lambda l: l*(l+1)/2/np.pi
		lb, cltt_unl = FFT.GetCl(T_unl, prefact=pref)
		lb, cltt_len = FFT.GetCl(T_len, prefact=pref)

		cltt_len_avg.add(cltt_len)
		cltt_unl_avg.add(cltt_unl)

		lb, clee_unl = FFT.GetCl(E_unl, prefact=pref)
		lb, clee_len = FFT.GetCl(E_len, prefact=pref)

		clee_len_avg.add(clee_len)
		clee_unl_avg.add(clee_unl)

		# lb, clbb_unl = FFT.GetCl(B_unl, prefact=pref)
		lb, clbb_len = FFT.GetCl(B_len, prefact=pref)

		clbb_len_avg.add(clbb_len)
		# clbb_unl_avg.add(cltt_unl)

if calc_specs:
	# TT
	f = np.ones_like(fact) 
	f[0] = f[1] = 1. 
	f[2:] = blsqrtfl_T[:3499]
	plt.plot(l, fact*cls_unl[:,0],'k--')
	# plt.plot(l, fact*cls_len_camb[:,0],'r--')
	plt.plot(l, fact*cls_len_camb[:,0]*f**2,'r--')
	plt.errorbar(lb, cltt_unl_avg.mean(), yerr=cltt_unl_avg.errs(), color='k', fmt='x')
	plt.errorbar(lb+5, cltt_len_avg.mean(), yerr=cltt_len_avg.errs(), color='r', fmt='o')
	plt.xlabel(r'$\ell$', size=15)
	plt.ylabel(r'$\ell(\ell+1)C_{\ell}^{TT}/2\pi \,[\mu K^2]$', size=15)

	plt.show()

	# EE
	f = np.ones_like(fact) 
	f[0] = f[1] = 1. 
	f[2:] = blsqrtfl_E[:3499]
	plt.plot(l, fact*cls_unl[:,1],'k--')
	# plt.plot(l, fact*cls_len_camb[:,1],'r--')
	plt.plot(l, fact*cls_len_camb[:,1]*f**2,'r--')
	plt.errorbar(lb, clee_unl_avg.mean(), yerr=clee_unl_avg.errs(), color='k', fmt='x')
	plt.errorbar(lb+5, clee_len_avg.mean(), yerr=clee_len_avg.errs(), color='r', fmt='o')
	plt.xlabel(r'$\ell$', size=15)
	plt.ylabel(r'$\ell(\ell+1)C_{\ell}^{EE}/2\pi \,[\mu K^2]$', size=15)

	plt.show()

	# BB
	f = np.ones_like(fact) 
	f[0] = f[1] = 1. 
	f[2:] = blsqrtfl_B[:3499]
	# plt.plot(l, l**2*cls_unl[:,1],'k--')
	# plt.plot(l, fact*cls_len_camb[:,2],'r--')
	plt.plot(l, fact*cls_len_camb[:,2]*f**2,'r--')
	# plt.errorbar(lb, clbb_unl_avg.mean(), yerr=clbb_unl_avg.errs(), color='k', fmt='x')
	plt.errorbar(lb+5, clbb_len_avg.mean(), yerr=clbb_len_avg.errs(), color='r', fmt='o')
	plt.xlabel(r'$\ell$', size=15)
	plt.ylabel(r'$\ell(\ell+1)C_{\ell}^{BB}/2\pi \,[\mu K^2]$', size=15)

	plt.show()

embed()

