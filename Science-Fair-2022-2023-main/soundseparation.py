import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

sample_rate, audio_data = wavfile.read('/kaggle/input/urbansound8k/fold1/102106-3-0-0.wav')
ica = FastICA(n_components=2)
sources = ica.fit_transform(audio_data)

plt.plot(audio_data)
plt.xlabel("Time")
plt.title("Sound Before Separation")
plt.show()

plt.plot(sources[:, 0])
plt.xlabel("Time")
plt.title("Sound 1")
plt.show()

plt.plot(sources[:, 1], color = '#F08000')
plt.xlabel("Time")
plt.title("Sound 2")
plt.show()
