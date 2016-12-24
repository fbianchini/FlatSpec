import numpy as np
import argparse
import sys
from ModeCoupl import GetMll
import cPickle as pickle
import pickle as pk, gzip

"""
Code for generating mode-coupling matrix from command line
"""

if __name__=='__main__':
		parser = argparse.ArgumentParser(description='')
		parser.add_argument('-mask', dest='mask', action='store', help='pkl file containing the mask', default=None)
		parser.add_argument('-reso', dest='reso', action='store', help='pixel resolution in arcmin', type=float, required=True)
		parser.add_argument('-lmin', dest='lmin', action='store', help='Minimum ell', type=int, default=0)
		parser.add_argument('-lmax', dest='lmax', action='store', help='Maximum ell', type=int, default=4001)
		parser.add_argument('-npts', dest='npts', action='store', help='# of points for matrix integration', type=int, default=4000)
		parser.add_argument('-fout', dest='fout', action='store', help='pkl filename to store the mode-coupling matrix', default='mll.pkl')

		args = parser.parse_args()

		for arg in sorted(vars(args)):
			print arg, getattr(args, arg)

		mask = np.load(args.mask)

		print("...Starting mode-coupling matrix calculation...")
		Mll = GetMll(mask, args.reso, lmin=args.lmin, lmax=args.lmax, npts=args.npts)
		print("...done...")
		print("...dumping matrix to pkl file...")
		#pickle.dump(Mll, open(args.fout, 'w'))
		pk.dump(Mll,gzip.open(args.fout,'wb'), protocol=2)
		print("...done...")

