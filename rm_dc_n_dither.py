#Remove DC component and add dither (rm_dc_n_dither)
import numpy as np
from scipy.signal import lfilter

def rm_dc_n_dither(data, fs):
    
	if fs == 16000:
		alpha = 0.99
	elif fs == 8000:
		alpha = 0.999
	else:
		print('Only 8 and 16 kHz are supported!')
		return -1

	a = np.array([1, -1])
	b = np.array([1, -alpha])
	data = lfilter(a, b, data, axis=0)

	length = len(data)
	r1 = np.random.rand(length,1)
	r2 = np.random.rand(length,1)
	dither = r1 + r2 - 1

	spow = np.std(data)
	aux = 0.000001*spow*dither
	sout = data + aux
	return sout

