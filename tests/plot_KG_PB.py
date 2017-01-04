import numpy as np
from scipy.linalg import pinv2
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
from curvspec.master import Binner
from Spec2D import *
from Sims import *
import seaborn as sns
import healpy as hp
from tqdm import tqdm
import cPickle as pickle

# TODO: check compatibility of results dict

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

def GetPValue(data, cov, theory=0., bmin=0, bmax=None):

	if theory == 0:
		theory = np.zeros_like(data)

	delta = data[bmin:bmax] - theory[bmin:bmax]
	chi2  = np.dot(delta, np.dot(pinv2(cov[bmin:bmax,bmin:bmax]), delta))
	print("~> delta = ", delta)
	print("~> chi2 = %f" %chi2)
	return 1. - stats.chi2.cdf(chi2, delta.size)

def do_plot(results):

	nres = len(results)
	nbin = len(results[0]['lb'])

	dl = results[0]['lb'][1]-results[0]['lb'][0]

	L = np.zeros((nres,nbin))

	for i in xrange(nbin):
		_bins_ = np.linspace(results[0]['lb'][i]-dl/4., results[0]['lb'][i]+dl/4., nres, endpoint=True)
		for j in xrange(nres):
			L[j,i] = _bins_[j]

	Bin = Binner(bin_edges=results[0]['bins'])

	if results[0]['smooth'] != 0:
		bl = hp.gauss_beam(results[0]['smooth'] * np.pi / 180. / 60., results[0]['clkg']-1)
	else:
		bl = 1.0

	theorybin = Bin.bin_spectra(results[0]['clkg']*bl**2)

	f, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,8))

	plt.suptitle(r'$N_{\rm sim} = %d$'%results[0]['cl_sims'].shape[0] + r' - $N_x =$ %d ' %results[0]['nx'] +' (%.2f arcmin) '%results[0]['reso'])

	for i in xrange(results[0]['cl_sims'].shape[0]):
		if i == 0:
			ax1.plot(results[0]['lb'], results[0]['cl_sims'][i,:], 'lightgrey', lw=0.4, label='Sims')
		else:
			ax1.plot(results[0]['lb'], results[0]['cl_sims'][i,:], 'lightgrey', lw=0.4)
	
	ax1.plot(results[0]['clkg'], 'k-', lw=2, label='Theory')
	for i in xrange(nres):
		ax1.errorbar(L[i,:], results[i]['cl_mean'], yerr=results[i]['cl_err']/np.sqrt(results[i]['cl_sims'].shape[0]), label=r'Mean', fmt='o', capsize=0)
	ax1.legend(loc='best')
	ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax1.set_ylabel(r'$C_{\ell}^{\kappa g}$')
	ax1.set_xlim([2, 2010])
	# ax1.set_ylim([0,7e-7])

	for i in xrange(nres):
		ax2.errorbar(L[i,:], results[i]['cl_mean']/theorybin - 1., yerr=results[i]['cl_err']/np.sqrt(results[i]['cl_sims'].shape[0])/theorybin, fmt='o', capsize=0)
	ax2.axhline(ls='--', color='k')
	ax2.set_xlabel(r'Multipole $\ell$')
	ax2.set_ylabel(r'$\frac{\langle \hat{C}_{\ell}^{\kappa g} \rangle}{C_{\ell}^{\kappa g,th}}-1$')
	# ax2.set_ylim([-0.1,0.15])
	ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))

	print("...here comes the plot...")
	plt.tight_layout()
	plt.show()
	# plt.savefig('plots/'+name+'/cl'+name+'_spectra_nsim'+str(args.nsim)+'_reso'+str(args.reso)+'_nx'+str(args.nx)+'_pad'+str(args.pad)+'_smooth'+str(args.smooth)+'_buff'+str(args.buff)+'_smoothpad'+str(args.smooth_pad)+'_deltaell'+str(args.delta_ell)+'_noise'+str(args.noise)+'_Deflection.pdf', bboxes_inches='tight')
	# plt.close()

def main(args):

	print args.fres

	# Loading results
	results = []
	for i in xrange(len(args.fres)):
		results.append(pickle.load(open(args.fres[i],'rb')))

	do_plot(results)
	
	# print ('=> KG p-value w/o Mll = %f' %(GetPValue(clnomll_mean[1:], Bin.bin_spectra(clkg*bl**2)[1:], clnomll_cov[1:,1:]/args.nsim)))
	# print ('=> KG p-value w/  Mll = %f' %(GetPValue(clmlldl1_mean[1:], Bin.bin_spectra(clkg*bl**2)[1:], clmlldl1_cov[1:,1:]/args.nsim)))

	embed()

if __name__=='__main__':
		parser = argparse.ArgumentParser(description='')
		parser.add_argument('-fres', dest='fres', action='store', help='pkl file where sims are stored', required=True, nargs='+')
		args = parser.parse_args()
		main(args)

