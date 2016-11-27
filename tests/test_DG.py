import numpy as np
from scipy.linalg import inv
from scipy import stats
from IPython import embed
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import argparse, sys
sys.path.append('../')
from Spec2D import *
from Sims import *
import seaborn as sns
import healpy as hp
import scipy.signal
from tqdm import tqdm

arcmin2rad = np.pi / 180. / 60. 
rad2arcmin = 1./arcmin2rad


def SetPlotStyle():
   rc('text',usetex=True)
   rc('font',**{'family':'serif','serif':['Computer Modern']})
   plt.rcParams['axes.linewidth']  = 3.
   plt.rcParams['axes.labelsize']  = 28
   plt.rcParams['axes.titlesize']  = 22
   plt.rcParams['xtick.labelsize'] = 20
   plt.rcParams['ytick.labelsize'] = 18
   plt.rcParams['xtick.major.size'] = 7
   plt.rcParams['ytick.major.size'] = 7
   plt.rcParams['xtick.minor.size'] = 3
   plt.rcParams['ytick.minor.size'] = 3
   plt.rcParams['legend.fontsize']  = 22
   plt.rcParams['legend.frameon']  = False

   plt.rcParams['xtick.major.width'] = 1
   plt.rcParams['ytick.major.width'] = 1
   plt.rcParams['xtick.minor.width'] = 1
   plt.rcParams['ytick.minor.width'] = 1
   # plt.clf()
   sns.set(rc('font',**{'family':'serif','serif':['Computer Modern']}))
   sns.set_style("ticks", {'figure.facecolor': 'grey'})

SetPlotStyle() 

class Binner(object):
	"""
	Class for computing binning scheme.
	"""
	def __init__(self, bin_edges=None, lmin=2, lmax=500, delta_ell=50, flat=None):
		"""
		Parameters
		----------
		bin_edges: array
			Edges of bandpowers
		lmin : int
		    Lower bound of the first l bin.
		lmax : int
		    Highest l value to be considered. The inclusive upper bound of
		    the last l bin is lesser or equal to this value.
		delta_ell :
		    The l bin width.
		flat : function flat(l)
			Power spectrum flattening type. Default = None
		"""
		
		if bin_edges is None:
			if lmin < 1:
			    raise ValueError('Input lmin is less than 1.')
			if lmax < lmin:
			    raise ValueError('Input lmax is less than lmin.')

			self.lmin      = int(lmin)
			self.lmax      = int(lmax)
			self.delta_ell = int(delta_ell)

			nbins = (self.lmax - self.lmin + 1) // self.delta_ell
			start = self.lmin + np.arange(nbins) * self.delta_ell
			stop  = start + self.delta_ell
			self.lmax = stop[-1]

		else:
			self.bin_edges = bin_edges
			self.lmin      = int(bin_edges[0])
			self.lmax      = int(bin_edges[-1])
			self.delta_ell = bin_edges[1:] - bin_edges[:-1]
			
			nbins = len(self.delta_ell)
			start = np.floor(self.bin_edges[:-1])
			stop  = np.ceil(self.bin_edges[1:])

		# Centers of bandpowers
		self.lb = (start + stop - 1) / 2

		# Apply prewhitening to bandpowers
		self.flat = flat
		if self.flat == None:
		    flat_ = np.ones(self.lmax + 1)
		else:
		    flat_ = self.flat(np.arange(self.lmax + 1))

		# Creating binning operators 		 
		self.P_bl = np.zeros((nbins, self.lmax + 1))
		self.Q_lb = np.zeros((self.lmax + 1, nbins))

		for b, (a, z) in enumerate(zip(start, stop)):
			a = int(a); z = int(z)
			self.P_bl[b, a:z] = 1. * flat_[a:z] / (z - a)
			self.Q_lb[a:z, b] = 1. / flat_[a:z]

	def bin_spectra(self, spectra):
		"""
		Average spectra in bins.
		"""
		spectra = np.asarray(spectra)
		lmax    = spectra.shape[-1] - 1
		if lmax < self.lmax:
			raise ValueError('The input spectra do not have enough l.')

		return np.dot(self.P_bl, spectra[..., :self.lmax+1])

def GetPValue(data, theory, cov):

	delta = data - theory 
	chi2  = np.dot(delta, np.dot(inv(cov), delta))
	print("~> delta = ", delta)
	print("~> chi2 = %f" %chi2)
	return 1. - stats.chi2.cdf(chi2, delta.size)

def do_plot(name, lb, cl, cl_mean, cl_err, theory, theorybin):

	print args.nsim

	f, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,8))

	plt.suptitle(r'$N_{\rm sim} = %d$'%args.nsim + r' - $N_x =$ %d ' %args.nx +' (%.2f arcmin) '%args.reso +' - Signal only')

	for i in xrange(args.nsim):
		if i == 0:
			ax1.plot(lb, cl[i,:], 'lightgrey', lw=0.4, label='Sims')
		else:
			ax1.plot(lb, cl[i,:], 'lightgrey', lw=0.4)
	
	ax1.plot(theory, 'k-', lw=2, label='Theory')
	ax1.errorbar(lb, cl_mean, yerr=cl_err/np.sqrt(args.nsim), label=r'Mean', color='royalblue', fmt='o', capsize=0)
	ax1.legend(loc='best')
	ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax1.set_ylabel(r'$C_{\ell}^{'+name+'}$')
	ax1.set_xlim([2, 2010])
	# ax1.set_ylim([0,7e-7])

	ax2.errorbar(lb, cl_mean/theorybin - 1., yerr=cl_err/np.sqrt(args.nsim)/theorybin, color='royalblue', fmt='o', capsize=0)
	ax2.axhline(ls='--', color='k')
	ax2.set_xlabel(r'Multipole $\ell$')
	ax2.set_ylabel(r'$\frac{\langle \hat{C}_{\ell}^{'+name+r'} \rangle}{C_{\ell}^{'+name+r',th}}-1$')
	ax2.set_ylim([-0.1,0.15])
	ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))

	print("...here comes the plot...")
	# plt.show()
	plt.tight_layout()
	plt.savefig('plots/'+name+'/cl'+name+'_spectra_nsim'+str(args.nsim)+'_reso'+str(args.reso)+'_nx'+str(args.nx)+'_pad'+str(args.pad)+'_smooth'+str(args.smooth)+'_buff'+str(args.buff)+'_smoothpad'+str(args.smooth_pad)+'_deltaell'+str(args.delta_ell)+'_noise'+str(args.noise)+'_Deflection.pdf', bboxes_inches='tight')
	plt.close()

def main(args):

	# Loading CMB TT spectra
	l, clkg, clgg, clkk, nlkk = np.loadtxt('spectra/XCSpectra.dat', unpack=True, usecols=[0,1,3,6,7])

	print args.noise

	print("...Theory spectra loaded...")

	cls = np.zeros((len(l),3))
	cls[:,0] = np.nan_to_num(clkk * 4 / (l*(l+1)))
	cls[:,1] = clgg
	cls[:,2] = np.nan_to_num(clkg * 2 / np.sqrt(l*(l+1)))

	nldd = np.nan_to_num(nlkk * 4 / (l*(l+1)))

	CLKK = []
	CLGG = []
	CLKG = []

	# Reading mask
	if args.mask is not None:
		print args.mask
		mask = np.load(str(args.mask))
		assert (mask.shape == (args.nx, args.nx))
	else:
		mask = np.ones((args.nx, args.nx))

	if args.delta_ell == 100:
		bins = np.arange(0,21)*args.delta_ell
	elif args.delta_ell == 200:
		bins = np.arange(0,11)*args.delta_ell


	Bin = Binner(bin_edges=bins)

	# white = lambda l:l*(l+1)/2/np.pi

	# p = Pix(args.nx*2, args.reso)
	# L = p.GetL(shift=False)
	# f = np.sqrt(L*(L+1))/2.

	# Running sims
	if args.noise:
		print("...Start running %d sims w/ noise" %args.nsim)
	else:
		print("...Start running %d sims w/o noise" %args.nsim)		
	for i in tqdm(xrange(args.nsim)):
		
		# Genereting signal maps
		data_dd, data_gg = GenCorrFlatMaps(cls, args.nx, args.reso, buff=args.buff, seed=i)
		# data_kk = GenCorrFlatMaps(clkk, args.nx, args.reso, buff=args.buff, seed=i)
		
		# Generate noise maps if requested
		if args.noise:
			data_ndd   = GenCorrFlatMaps(nldd/10., args.nx, args.reso, buff=args.buff, seed=args.nsim*i)
			data_ddtot = data_dd + data_ndd
			data_ggtot = GetCountsTot(data_gg, 5.76e6, args.nx, args.reso, dim='sterad') # Hard-coded H-ATLAS galaxy density in gal/steradian
			data_ggtot = data_ggtot/data_ggtot.mean() - 1. # To delta_g
		else:
			data_ddtot = data_dd
			data_ggtot = data_gg

		# Deflection => convergence
		DD = FlatMapReal(args.nx, args.reso, map=data_ddtot, mask=mask)
		KK = DD.FilterMap(lambda l: np.sqrt(l*(l+1))/2., padX=2, array=False)
		GG = FlatMapReal(args.nx, args.reso, map=data_ggtot, mask=mask)
		KK.mask = mask
		# plt.subplot(121);plt.imshow(data_kk*KK.mask, vmin=-0.2, vmax=0.2);plt.colorbar()
		# plt.subplot(122);plt.imshow(KK.map*KK.mask, vmin=-0.2, vmax=0.2);plt.colorbar()
		# plt.show()

		# Want some smoothies?
		if args.smooth != 0: 
			KK.ApplyGaussBeam(args.smooth)
			GG.ApplyGaussBeam(args.smooth)
			DD.ApplyGaussBeam(args.smooth)
		
		if args.pad != 0: 
			DD = DD.Pad(args.pad, apo_mask=False)
			KK = KK.Pad(args.pad, apo_mask=False)#, fwhm=args.smooth_pad)
			GG = GG.Pad(args.pad, apo_mask=False)#, fwhm=args.smooth_pad)
			# X = GG.map*GG.mask
			# plt.subplot(121)
			# plt.imshow(GG.mask)
			# plt.subplot(122)
			# plt.imshow(X)
			# plt.show()
			# plt.plot(X[:,280])#np.ceil(args.nx/2)])
			# plt.show()
			nx = args.pad * args.nx
		else:
			nx = args.nx

		# embed()

		# FT_DD     = FlatMapFFT(nx, args.reso, map=DD)
		FT_KK     = FlatMapFFT(nx, args.reso, map=KK)
		# FT_GG     = FlatMapFFT(nx, args.reso, map=GG)
		
		# lb, clkk_ = FT_KK.GetCl(lbins=bins)
		# lb, cldd_ = FT_DD.GetCl(lbins=bins, prefact=lambda l:l*(l+1)/4.)
		# lb, clgg_ = FT_GG.GetCl(lbins=bins)
		lb, clkg_ = FT_KK.GetCl(lbins=bins, map2=GG)
		# lb, clkg_ = FT_DD.GetCl(lbins=bins, map2=GG, prefact=lambda l: np.sqrt(l*(l+1))/2.)
		
		# plt.plot(lb, cldd_, '--')
		# plt.plot(lb, clkk_, '-')#, lw=0.3)
		# plt.plot(l, clkk, ':')
		# plt.show()
		# if i >20:
		# 	plt.show()
		# 	sys.exit()
		# CLKK.append(clkk_)
		# CLGG.append(clgg_)
		CLKG.append(clkg_)
		# embed() 
		# sys.exit()
	print("...sims done...")

	# Computing statistics
	# CLKK = np.asarray(CLKK)
	# CLGG = np.asarray(CLGG)
	CLKG = np.asarray(CLKG)

	# clkk_mean = np.mean(CLKK, axis=0)
	# clgg_mean = np.mean(CLGG, axis=0)
	clkg_mean = np.mean(CLKG, axis=0)
	
	# clkk_cov  = np.cov(CLKK.T)
	# clgg_cov  = np.cov(CLGG.T)
	clkg_cov  = np.cov(CLKG.T)

	# clkk_err  = np.sqrt(np.diag(clkk_cov))
	# clgg_err  = np.sqrt(np.diag(clgg_cov))
	clkg_err  = np.sqrt(np.diag(clkg_cov))

	if args.smooth != 0:
		bl = hp.gauss_beam(args.smooth*np.pi / 180. / 60.,l.size-1)
	else:
		bl = 1.0

	# print ('=> KK p-value = %f' %(GetPValue(clkk_mean, Bin.bin_spectra(clkk*bl**2), clkk_cov/args.nsim)))
	# print ('=> GG p-value = %f' %(GetPValue(clgg_mean, Bin.bin_spectra(clgg*bl**2), clgg_cov/args.nsim)))
	print ('=> KG p-value = %f' %(GetPValue(clkg_mean[1:], Bin.bin_spectra(clkg*bl**2)[1:], clkg_cov[1:,1:]/args.nsim)))

	# Plots ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# do_plot('kk', lb, CLKK, clkk_mean, clkk_err, clkk, Bin.bin_spectra(clkk*bl**2))
	# do_plot('gg', lb, CLGG, clgg_mean, clgg_err, clgg, Bin.bin_spectra(clgg*bl**2))
	do_plot('kg', lb, CLKG, clkg_mean, clkg_err, clkg*bl**2, Bin.bin_spectra(clkg*bl**2))

	plt.imshow(np.corrcoef(CLKG.T), interpolation='nearest', cmap='RdBu', vmin=-1., vmax=1.)
	plt.colorbar()
	plt.show()

	embed()

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
		args = parser.parse_args()
		main(args)

