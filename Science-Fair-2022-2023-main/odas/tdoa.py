import odas
import numpy as np

'''
Given a list of noise signals, closest_distance(signals, freq, d) returns:
The distance of the microphone with the closest distance.
It returns a negative distance if the noise is behind the microphone.
'''

# freq = 44100 b/c frequency of sound is 44.1 kHz
# d = 0.035 b/c distance between mics are 0.035 meters

def closest_distance(signals, freq=44100, d=0.035):
	z = odas.gcc(signals, freq)
	dist = [(z[i] * freq * 0.5 * d) for i in range(4)]
	return min(dist)

'''
Given a list of noise signals x, direction_of_arrival(signals, freq, d) returns:
The direction of arrival of the noise signal as an angle
'''

def direction_of_arrival(signals, freq=44100, d=0.035):
	z = odas.gcc(signals, freq)
	dist = [(z[i] * freq * 0.5 * d) for i in range(4)]
	angle = math.atan2(dist[2] - dist[0], dist[1] - dist[3]) * 180 / math.pi
	return angle
