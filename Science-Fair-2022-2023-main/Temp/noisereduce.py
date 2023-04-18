from scipy.io import wavfile
import noisereduce as nr
import os

files = os.listdir('/kaggle/input/environmental-sound-classification-50/audio/audio/')

for i in range(len(files)):
    rate, data = wavfile.read('/kaggle/input/environmental-sound-classification-50/audio/audio/' + files[i])
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(str(i) + '.wav', rate, reduced_noise)
