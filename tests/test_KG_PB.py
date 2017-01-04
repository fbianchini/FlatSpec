import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import argparse, sys
sys.path.append('../')
sys.path.append('../../CurvSpec/')
from Spec2D import *
from Sims import *
import healpy as hp
import scipy.signal
from tqdm import tqdm
import cPickle as pickle
import gzip

arcmin2rad = np.pi / 180. / 60. 
rad2arcmin = 1./arcmin2rad

def Cnt2DltApo(cnt, mask, reso=2., thr=1e-14, ngal=False):
	dlt = np.zeros(cnt.shape)
	ngal_pix = np.sum(cnt*mask)/np.sum(mask)
	dlt = cnt / ngal_pix - 1.
	ngal = ngal_pix * (reso*arcmin2rad)**2
	# if ngal == True:
	# 	return dlt, ngal, ngal_pix
	# else: 
	return dlt

def main(args):

	# Loading CMB TT spectra
	l, clkg, clgg, clkk, nlkk = np.loadtxt('spectra/XCSpectra.dat', unpack=True, usecols=[0,1,3,6,7])

	for arg in sorted(vars(args)):
		print arg, getattr(args, arg)

	print("...Theory spectra loaded...")

	cls = np.zeros((len(l),3))
	cls[:,0] = clkk
	cls[:,1] = 9.*clgg
	cls[:,2] = 3.*clkg
	# nldd = np.nan_to_num(nlkk * 4 / (l*(l+1)))

	cl_sims = []

	# Bins
	if args.delta_ell == 100:
		bins = np.arange(0,21)*args.delta_ell
	elif args.delta_ell == 200:
		bins = np.arange(0,11)*args.delta_ell

	# Reading mask
	if args.mask is not None:
		print("mask -->", args.mask)
		mask = np.load(str(args.mask))
		# plt.imshow(mask)
		# plt.show()
		assert (mask.shape == (args.nx, args.nx))
		if args.mll is not None:
			MASTER = True
			print("...loading unbinned mode-coupling matrix --> ", args.mll)
			try:
				Mll = pickle.load(gzip.open(args.mll,'rb') )
			except:
				Mll = pickle.load(open(args.mll,'rb') )
			Mll = Mll[:bins[-1]+1,:bins[-1]+1]
		else:
			MASTER = False
			Mll    = None
		print("...done...")
	else:
		mask   = np.ones((args.nx, args.nx))
		MASTER = False
		Mll    = None


	if args.noise:
		print("...Start running %d sims w/ noise" %args.nsim)
	else:
		print("...Start running %d sims w/o noise" %args.nsim)		

	# Running sims
	for i in tqdm(xrange(args.nsim)):	
		
		# Genereting mock data maps
		data_kk, data_gg = GenCorrFlatMaps(cls, args.nx, args.reso, buff=args.buff, seed=i)
		if args.noise:
			data_nkk   = GenCorrFlatMaps(nlkk/15., args.nx, args.reso, buff=args.buff, seed=args.nsim*i)
			data_kktot = data_kk + data_nkk
			data_ggtot = GetCountsTot(data_gg, 5e6, args.nx, args.reso, dim='sterad') # Hard-coded H-ATLAS galaxy density in gal/steradian
			data_ggtot = Cnt2DltApo(data_ggtot, mask)
			# data_ggtot = data_ggtot/data_ggtot.mean() - 1. # To delta_g
		else:
			data_kktot = data_kk
			data_ggtot = data_gg

		KK = FlatMapReal(args.nx, args.reso, map=data_kktot, mask=mask)
		GG = FlatMapReal(args.nx, args.reso, map=data_ggtot, mask=mask)

		# Want some smoothies?		
		if args.smooth != 0: 
			KK.ApplyGaussBeam(args.smooth)
			GG.ApplyGaussBeam(args.smooth)

		# Want some paddies?
		if args.pad != 0: 
			KK = KK.Pad(args.pad, apo_mask=False)#, fwhm=args.smooth_pad)
			GG = GG.Pad(args.pad, apo_mask=False)#, fwhm=args.smooth_pad)
			nx = args.pad * args.nx
		else:
			nx = args.nx

		FT_KK = FlatMapFFT(nx, args.reso, map=KK)
		FT_GG = FlatMapFFT(nx, args.reso, map=GG)
		
		lb, cl = FT_KK.GetCl(lbins=bins, map2=GG, MASTER=MASTER, mll=Mll)

		cl_sims.append(cl)
		# embed() 
		# sys.exit()
	print("...sims done...")

	# Computing statistics
	print("...computing stats...")
	cl_sims = np.asarray(cl_sims)
	cl_mean = np.mean(cl_sims, axis=0)
	cl_cov  = np.cov(cl_sims.T)
	cl_corr = np.corrcoef(cl_sims.T)
	cl_err  = np.sqrt(np.diag(cl_cov))

	# Dumping stats
	results = {}
	results['cl_sims'] = cl_sims
	results['cl_mean'] = cl_mean
	results['cl_cov']  = cl_cov
	results['cl_corr'] = cl_corr
	results['cl_err']  = cl_err
	results['lb']      = lb
	results['smooth']  = args.smooth
	results['reso']    = args.reso
	results['nx']      = args.nx
	results['clkk']    = cls[:,0]
	results['clgg']    = cls[:,1]
	results['clkg']    = cls[:,2]
	results['bins']    = bins

	pickle.dump(results, open(args.fout,'wb'))
	print("...stats dumped...")
	print("...GAME OVER...")

if __name__=='__main__':
		parser = argparse.ArgumentParser(description='')
		parser.add_argument('-nx', dest='nx', action='store', help='# of pixels x-direction', type=int, required=True)
		parser.add_argument('-reso', dest='reso', action='store', help='pixel resolution in arcmin', type=float, required=True)
		parser.add_argument('-nsim', dest='nsim', action='store', help='# of simulations', type=int, default=100)
		parser.add_argument('-pad', dest='pad', action='store', help='# of pixels in the padded map', type=int, default=0)
		parser.add_argument('-smooth', dest='smooth', action='store', help='fwhm [arcmin] Gaussian beam smoothing', type=float, default=0)
		parser.add_argument('-buff', dest='buff', action='store', help='create map from nx*buffer to avoid periodic boundaries', type=float, default=1)
		parser.add_argument('-mask', dest='mask', action='store', help='pkl file containing the mask', default=None)
		parser.add_argument('-smoothpad', dest='smooth_pad', action='store', help='fwhm [arcmin] smoothing of borders when padding', type=float, default=0.)
		parser.add_argument('-deltaell', dest='delta_ell', action='store', help='binsize', type=int, default=100)
		parser.add_argument('-noise', dest='noise', action='store', help='Add some lensing and galaxy noise?', type=bool, default=False)
		parser.add_argument('-mll', dest='mll', action='store', help='pkl file w/ mode-coupling matrix', default=None)
		parser.add_argument('-fout', dest='fout', action='store', help='pkl file where to store sims', default='results.pkl')
		args = parser.parse_args()
		main(args)

