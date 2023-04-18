import numpy as np
import scipy.io.wavfile as wf
from skimage.filters import threshold_otsu
from scipy.fftpack import fft, ifft

# Read audio file
fs, audio_data = wf.read('original.wav')

# Convert audio data to complex number format
audio_data = audio_data.astype(np.complex64)

# Perform FFT on audio data
fft_data = fft(audio_data)

# Calculate magnitude of FFT data
magnitude = np.abs(fft_data)

# Calculate Otsu threshold
threshold = threshold_otsu(magnitude)

# Apply threshold to magnitude data
denoised_magnitude = np.where(magnitude > threshold, magnitude, 0)

# Perform inverse FFT on denoised data
denoised_data = ifft(denoised_magnitude).astype(np.int16)

# Write denoised audio data to output file
wf.write('output.wav', fs, denoised_data)

def plot_wave(file, title):
  y, sr = librosa.load(file)
  plt.figure()
  librosa.display.waveplot(y, sr=sr)
  plt.title(title)
  plt.show()

def plot_spectogram(audio_data, fs, title):
  plt.specgram(audio_data, Fs=fs)
  plt.xlabel('Time')
  plt.title(title)
  plt.show()


plot_wave('/content/original.wav', 'Original Sound')

plot_spectogram(audio_data, fs, 'Original Sound')

wf.write('/content/output.wav', fs, denoised_data)

plot_wave('/content/output.wav', 'Denoised Sound (Fast Fourier Transform)')

plot_spectogram(denoised_data, fs, 'Denoised Sound (Fast Fourier Transform)')