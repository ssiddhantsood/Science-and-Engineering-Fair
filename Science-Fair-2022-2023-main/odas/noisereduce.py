import odas
import numpy as np

'''
Given a noise signal x, noise_reduce(x, freq) returns:
The denoised noise using a fourier transform.
y is a noise signal as a numpy array.
'''

def noise_reduce(signal, freq=44100):
	y = odas.odas_denoise(signal, freq)
	return y
