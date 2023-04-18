import numpy as np
import noisereduce as nr
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import librosa
import librosa.display

fs, audio_data = wf.read('/content/original.wav')
audio_data = np.array(audio_data)

def denoise(audio_data, fs):
	denoised_data = nr.reduce_noise(audio_data, fs)
	return denoised_data

def plot_wave(file, title):
  y, sr = librosa.load(file)
  plt.figure()
  librosa.display.waveplot(y, sr=sr)
  plt.title(title)
  plt.show()

plot_wave('/content/original.wav', 'Original Sound')

denoised_data = denoise(audio_data, fs)

wf.write('/content/output.wav', fs, denoised_data)

plot_wave('/content/output.wav', 'Denoised Sound (Fast Fourier Transform)')
