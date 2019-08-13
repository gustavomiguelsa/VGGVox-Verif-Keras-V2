from rm_dc_n_dither import rm_dc_n_dither
from vec2frames import vec2frames
from scipy.signal import lfilter
import numpy as np
import math as mt

def mfccspec(speech, fs, Tw, Ts, alpha, window, R, M, N, L):

	
	if( max(abs(speech)) <= 1 ):
		speech = speech * 32768 # 2^15 = 32768


	Nw = round(0.001*Tw*fs)
	Ns = round(0.001*Ts*fs)
	
	nfft = 2**(mt.ceil(mt.log2(abs(Nw))))
	K = (nfft/2) + 1


	# Remove dc and add dither
	speech = rm_dc_n_dither(speech, fs)


	# Preemphasis filtering
	a = np.array([1, -alpha])
	speech = lfilter(a, 1, speech, axis=0)

	
	# Framing and windowing (frames as columns)
	frames = vec2frames(speech, Nw, Ns, 'cols', window, 0);

	# Magnitude spectrum computation (as column vectors)
	MAG =  abs(np.fft.fft(frames,nfft,axis=0))


	# 512 spectrum without phase information
	SPEC = MAG
	
	return SPEC



