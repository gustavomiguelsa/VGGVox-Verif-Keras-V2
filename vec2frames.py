import numpy as np
import math as mt

def vec2frames(vec, Nw, Ns, direction, window, padding):

	if Nw == 0 or Ns == 0:
		print('Wrong usage.')	
		return

	L = len(vec)			#Length of input vector
	vec.shape = (L,1)		#Ensure column vector
	M = mt.floor(((L-Nw)/Ns)+1)	#Number of frames


	E = (L - ((M-1)*Ns + Nw))

	if( E>0 ):
		
		P = Nw - E;

		# pad with zeros
		if(padding == 1):
			zarr = np.zeros((P,1))
			vec = np.concatenate((vec,zarr))
		
		# pad with a specific numeric constant	
		elif(padding > 1):
			oarr = padding * np.ones((P,1))
			vec = np.concatenate((vec,oarr))

		# pad with a low variance white Gaussian noise
		elif(padding == 'noise'):
			rarr = 0.000001 * np.random.randn(P,1)
			vec = np.concatenate((vec,rarr))

		# if not padding required, decrement frame count
		# (not a very elegant solution)
		else:
			M = M - 1

		# increment the frame count
		M = M + 1

	
	if(direction == 'rows'):
		indf = Ns * np.arange(0, M).reshape((-1,1))
		inds = np.arange(1, Nw+1)
		indexes = np.tile(indf,(1,Nw)) + np.tile(inds,(M,1))

	elif(direction == 'cols'):
		indf = Ns * np.arange(0, M)
		inds = np.arange(1, Nw+1).reshape((-1,1))
		indexes = np.tile(indf,(Nw,1)) + np.tile(inds,(1,M))

	else:
		print('Direction is not supported!')
		return -1

	frames = np.take(vec,indexes-1)



	if(window == 0):
		return

	if(isinstance(window, type(lambda:0))):
		window = window( Nw )

	if(~isinstance(window, type(lambda:0)) and len(window) == Nw):
		
		if(direction == 'rows'):
			aux = np.diag(window.reshape((1,-1))[0])
			frames = np.matmul(frames, aux)
		elif(direction == 'cols'):
			aux = np.diag(window.reshape((1,-1))[0])
			frames = np.matmul(aux, frames)

	return frames





































