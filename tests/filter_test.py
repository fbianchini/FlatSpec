import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import sys
sys.path.append('../')
from Spec2D import *
from Sims import *
from Utils import *
import seaborn as sns

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

# params
reso = 2.
nx   = 200
buff = 4

# Loading CMB TT spectra
l, clkg, clgg, clkk, nlkk = np.loadtxt('spectra/XCSpectra.dat', unpack=True, usecols=[0,1,3,6,7])

print("...Theory spectra loaded...")

cldd = np.nan_to_num(clkk * 4 / (l*(l+1)))

if False:
   plt.loglog(clkk, '-',  label=r'$C_{\ell}^{\kappa\kappa}$')
   plt.loglog(cldd, '--', label=r'$C_{\ell}^{dd}$')
   plt.xlabel(r'$\ell$')
   plt.legend(loc='best')
   plt.show()

def GetMaps(nx, reso, buff):
   p  = Pix(nx, reso)
   L  = p.GetL(shift=False)
   # L  = np.fft.fftshift(p.GetL())
   f  = np.sqrt(L*(L+1))/2.
   FT = FlatMapFFT(nx, reso)

   if False:
      plt.subplot(121); plt.imshow(L); plt.title(r'$L$');plt.colorbar()
      plt.subplot(122); plt.imshow(f); plt.title(r'$\sqrt{L(L+1)}/2$');plt.colorbar();plt.show()

   data_dd = GenCorrFlatMaps(cldd, nx, reso, seed=0, buff=buff)
   data_kk = GenCorrFlatMaps(clkk, nx, reso, seed=0, buff=buff)


   # data_kk_reco = (np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(data_dd))*f))).real
   data_kk_reco = (np.fft.ifft2(np.fft.fft2(data_dd)*f)).real
   # data_kk_reco2 = FT.FilterMap(f, map=data_dd, array=True)

   # embed()

   plt.suptitle('Buff = '+str(buff))

   plt.subplot(231); plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(data_dd))))); plt.title(r'$\log_{10}{|FT[d]|}$');plt.colorbar()
   plt.subplot(232); plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(data_kk))))); plt.title(r'$\log_{10}|FT[\kappa]|$');plt.colorbar()
   plt.subplot(233); plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(data_kk_reco))))); plt.title(r'$\log_{10}|FT[\kappa]|$ reco');plt.colorbar()
   # plt.show()

   plt.subplot(234); plt.imshow(data_dd); plt.title(r'$d$');plt.colorbar()
   plt.subplot(235); plt.imshow(data_kk); plt.title(r'$\kappa$');plt.colorbar()
   plt.subplot(236); plt.imshow(data_kk_reco); plt.title(r'$\kappa$ reco');plt.colorbar();plt.show()

   return data_dd, data_kk, data_kk_reco

def GetMapsPad(nx, reso, buff, pad, smooth):

   if False:
      plt.subplot(121); plt.imshow(L); plt.title(r'$L$');plt.colorbar()
      plt.subplot(122); plt.imshow(f); plt.title(r'$\sqrt{L(L+1)}/2$');plt.colorbar();plt.show()

   DD = FlatMapReal(nx, reso, map=GenCorrFlatMaps(cldd, nx, reso, seed=0, buff=1))
   KK = FlatMapReal(nx, reso, map=GenCorrFlatMaps(clkk, nx, reso, seed=0, buff=1))
   DD2 = FlatMapReal(nx, reso, map=GenCorrFlatMaps(cldd, nx, reso, seed=0, buff=2))
   KK2 = FlatMapReal(nx, reso, map=GenCorrFlatMaps(clkk, nx, reso, seed=0, buff=2))

   np.savetxt('deflection_field_nobuff_200x200_2arcminpix.txt', DD.map)
   np.savetxt('convergence_field_nobuff_200x200_2arcminpix.txt', KK.map)
   np.savetxt('deflection_field_buff2_200x200_2arcminpix.txt', DD2.map)
   np.savetxt('convergence_field_buff2_200x200_2arcminpix.txt', KK2.map)

   ds
   KK_reco2 = DD.FilterMap(lambda l: np.sqrt(l*(l+1))/2., padX=2, array=False)

   embed()
   DD = DD.Pad(pad, apo_mask=smooth)
   KK = KK.Pad(pad, apo_mask=smooth)

   FTDD = FlatMapFFT(DD.map.shape[0], reso, map=DD)
   FTKK = FlatMapFFT(KK.map.shape[0], reso, map=KK)

   # Filter in multipole space
   L  = FTDD.GetL(shift=False)
   f  = np.sqrt(L*(L+1))/2. # deflection to convergence

   # data_kk_reco = (np.fft.ifft2(FTDD.ft*f)).real
   # KK_reco       = FlatMapReal(data_kk_reco.shape[0], reso, map=data_kk_reco, mask=KK.mask)

   # From deflection to convergence 
   KK_reco      = FTDD.FT2Map(filt=f, array=False, lmin=0)
   KK_reco.mask = KK.mask
   FTKKreco     = FlatMapFFT(KK_reco.map.shape[0], reso, map=KK_reco)
   KK_reco2.mask = KK.mask[100:300,100:300]
   FTKKreco2      = FlatMapFFT(KK_reco2.map.shape[0], reso, map=KK_reco2)


   # Extracting spectra
   l_, cldd_     = FTDD.GetCl(lbins=np.arange(0,21)*100)
   l_, clkk_     = FTKK.GetCl(lbins=np.arange(0,21)*100)
   l_, clkk_reco = FTKKreco.GetCl(lbins=np.arange(0,21)*100)
   l_, clkk_reco_ =  FTKKreco2.GetCl(lbins=np.arange(0,21)*100)

   # plt.loglog(l, cldd, label='cldd Theory')
   # plt.loglog(l_, cldd_, label='FFT')
   # plt.legend()
   # plt.show()

   plt.plot(clkk_reco/clkk_reco_)
   plt.show()

   plt.plot(l, clkk, label='clkk Theory')
   plt.plot(l_, clkk_, label='FFT')
   plt.plot(l_, clkk_reco, label='FFT reco')
   plt.plot(l_, clkk_reco_, label='FFT reco 2')
   plt.legend()
   plt.show()

   embed()

   # plt.subplot(131);plt.imshow(data_kk*FTKKreco.map.Extract([100,300,100,300]).mask); 
   # plt.subplot(132);plt.imshow(FTKKreco.map.Extract([100,300,100,300]).map);
   # plt.subplot(133);plt.imshow(FTKKreco.map.Extract([100,300,100,300]).map/(data_kk*FTKKreco.map.Extract([100,300,100,300]).mask));
   # plt.show()

   KK_reco = KK_reco.Extract([100,300,100,300])
   KK      = KK.Extract([100,300,100,300])
   DD      = DD.Extract([100,300,100,300])

   plt.suptitle('Buff = '+str(buff))

   plt.subplot(231); plt.imshow(np.log10(np.abs(FTDD.Get2DSpectra(shift=True)))); plt.title(r'$\log_{10}{|FT[d]|}$');plt.colorbar()
   plt.subplot(232); plt.imshow(np.log10(np.abs(FTKK.Get2DSpectra(shift=True))),vmin=-24, vmax=-6); plt.title(r'$\log_{10}|FT[\kappa]|$');plt.colorbar()
   plt.subplot(233); plt.imshow(np.log10(np.abs(FTKKreco.Get2DSpectra(shift=True))), vmin=-24, vmax=-6); plt.title(r'$\log_{10}|FT[\kappa]|$ reco');plt.colorbar()

   plt.subplot(234); plt.imshow(DD.map*DD.mask, vmin=-0.002, vmax=0.002); plt.title(r'$d$');plt.colorbar()
   plt.subplot(235); plt.imshow(KK.map*KK.mask, vmin=-0.2, vmax=0.2); plt.title(r'$\kappa$');plt.colorbar()
   plt.subplot(236); plt.imshow(KK_reco.map*KK_reco.mask, vmin=-0.2, vmax=0.2); plt.title(r'$\kappa$ reco');plt.colorbar();plt.show()

   return DD.map, KK.map, KK_reco.map

# data_dd2, data_kk2, data_kk_reco2 = GetMaps(200, 2., 2)
# data_dd3, data_kk3, data_kk_reco3 = GetMaps(200, 2., 3)

data_dd3, data_kk3, data_kk_reco3 = GetMapsPad(200, 2., 2, 2, True)
data_dd3, data_kk3, data_kk_reco3 = GetMapsPad(200, 2., 3, 2, True)
data_dd3, data_kk3, data_kk_reco3 = GetMapsPad(200, 2., 5, 2, True)

# embed()





