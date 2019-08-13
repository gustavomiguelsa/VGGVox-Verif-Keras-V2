import numpy as np
import soundfile as sf

from runSpec import runSpec
from scipy.signal import lfilter

def test_getinput(filename):

	bkt_pool = np.array([2,5,8,11,14,17,20,23,27,30]);
	bkt_width = np.array([100,200,300,400,500,600,700,800,900,1000]);


	# Read data from WAV file
	[data, fs] = sf.read(filename)
	data = data.reshape((-1, 1))


	SPEC = runSpec(data,fs)


	mu = np.mean(SPEC,axis=1).reshape((-1,1))
	stdev = np.std(SPEC,axis=1,ddof=1).reshape((-1,1))
	nSPEC = SPEC - mu
	nSPEC = nSPEC / stdev


	idx = np.max(np.where(bkt_width <= np.shape(nSPEC)[1]))
	rsize = bkt_width[idx]
	rstart = int( round( (  np.shape(nSPEC)[1] - rsize ) / 2 ) )
	rend = int( (rstart+rsize-1) )

	if(rstart == 0):
		rstart = 1

	inp = nSPEC[:, rstart-1 : rend]

	return inp






