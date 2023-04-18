import odas

'''
Given the audio data as one signal and frequency as freq, 
sound_separation(audio_data, freq) returns:
The loudest four signals as a NumPy list.
freq is usually 44100
'''
def sound_separation(audio_data, freq=44100):
	sources = odas.doa(audio_data, freq)
	# sort by the sources by loudness
	loudest_sources = sorted(sources, key=lambda x: x.energy, reverse=True)
	top_4 = loudest_sources[:4]
	return top_4
