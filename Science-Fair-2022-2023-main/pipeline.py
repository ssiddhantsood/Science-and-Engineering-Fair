import odas
import numpy as np
import scipy.io.wavefile
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch
import sys
import librosa
import RPi.GPIO as GPIO
import time
import pyaudio

audio = pyaudio.PyAudio() # start pyaudio device
for indx in range(audio.get_device_count()):
    dev = audio.get_device_info_by_index(indx) # get device
    if dev['maxInputChannels']==4 and \
       len([ii for ii in dev['name'].split('-') if ii=='4mic'])>=1:
        print('-'*30)
        print('Found QuadMic!')
        print('Device Index: {}'.format(indx)) # device index
        print('Device Name: {}'.format(dev['name'])) # device name
        print('Device Input Channels: {}'.format(dev['maxInputChannels'])) # channels

'''
Given a noise signal x, noise_reduce(x, freq) returns:
The denoised noise using a fourier transform.
y is a noise signal as a numpy array.
'''

def noise_reduce(signal, freq=44100):
	y = odas.odas_denoise(signal, freq)
	return y

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

# model

def predict(filename): # y, s = librosa.load('test.wav', sr=16000 ) 
    y, sr = librosa.load(filename, sr = 16000)
    sampling_rate = sr

    sampling_rate = dataset.features["audio"].sampling_rate

    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    inputs = feature_extractor(y, sampling_rate=sampling_rate, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_ids = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_ids]
    return predicted_label


sound_files = np.array([load_audio_file("mic1.wav"),
              load_audio_file("mic2.wav"),
              load_audio_file("mic3.wav"),
              load_audio_file("mic4.wav")])

for i in range(len(sound_files)):
    sound_files[i] = reduceNoise(sound_files[i])

sounds = sound_separation(sound_files[0])  # change to have all dominating with max of four 
soundInfo = {}
distancesID = {}
distances = []
for i in sounds:
    d = closest_distance(i)
    distancesID[d] = i
    soundInfo[i] = (d, direction_of_arrival(i), predict(i))
    distances.append(abs(d))

distances.sort()  # sort based on distance from closest to farthest

# RETURNS POSITIVE OR NEGATIVE VALUE BASED ON DISTANCE
def changeIntensity(bright, led):
	led.ChangeDutyCycle(bright)

# if positive, sound is coming from back, else front
# if DOA return 

for i in distances:   
    (d, doa, pred) =  soundInfo[distancesID[i]]
    if d > 0:
        brightness = 50
        if doa > 0 and doa < 90:   
            changeIntensity(brightness, led1)
        elif doa > 90 and doa < 180:
            changeIntensity(brightness, led2)
        elif doa > 180 and doa < 270:
            changeIntensity(brightness, led3)
        elif doa > 270 and doa < 360:
            changeIntensity(brightness, led4)
    if d < 0:
        if doa > 0 and doa < 90:
            changeIntensity(brightness, led1)
        elif doa > 90 and doa] < 180:
            changeIntensity(brightness, led2)
        elif doa > 180 and doa < 270:
            changeIntensity(brightness, led3)
        elif doa > 270 and doa < 360:
            changeIntensity(brightness, led4)