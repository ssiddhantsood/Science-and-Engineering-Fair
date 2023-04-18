import numpy as np
import soundfile as sf

# Load the sound recording into a NumPy array
data, sample_rate = sf.read("recording.wav")

def calcTDOA(data, sample_rate):
    # Calculate the TDOA values for each pair of microphones
    num_channels = data.shape[1]
    tdoa_values = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            if i != j:
                tdoa_values[i, j] = np.argmax(np.abs(np.correlate(data[:, i], data[:, j]))) / sample_rate
    return tdoa_values

def calc_distances(tdoa, positions):
    c = 343 # speed of sound in m/s
    distances = []
    for i in range(4):
        distance = 0
        for j in range(4):
            if i != j:
                distance += c * tdoa[i][j] / 2
        distances.append(distance / 3)
    return distances
