import numpy as np
from mfccspec import mfccspec

def runSpec(speech, fs):

	Tw = 25
	Ts = 10
	alpha = 0.97
	R = np.array([300, 3700])
	M = 40
	N = 13
	L = 22

	hamming = lambda T: (0.54-0.46*np.cos(2*np.pi*np.arange(0,T).reshape((-1, 1))/(T-1)))

	SPEC = mfccspec(speech, fs, Tw, Ts, alpha, hamming, R, M, N, L)

	return SPEC






