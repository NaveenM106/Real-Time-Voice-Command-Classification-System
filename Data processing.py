#------------------------------------------------------------------------------------------------------------------
#   Speech data processing
#------------------------------------------------------------------------------------------------------------------

import pickle
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from python_speech_features import mfcc

# Load audio records
fs = 44100

file_name = '/Users/claudiogonzalezarriaga/Documents/Progra_Tec/CuartoSemestre/Modelacion del aprendizaje con IA/Voice Classification/06_10_2024_00_39_59.obj'
inputFile = open(file_name, 'rb')
data = pickle.load(inputFile)
n_trials = len(data)

# Plot one record
t = 5
dt = 1/fs
t_size = data[t][2].shape[0]

x = np.arange(0, t_size*dt, dt)

figure, axis = plt.subplots(2, 1)

axis[0].plot(x, data[t][2][:,0])
axis[0].set_title("Channel 1")

axis[1].plot(x, data[t][2][:,1])
axis[1].set_title("Channel 2")

plt.show()

# Filter audio signals
filt = signal.iirfilter(4, [10, 15000], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='ba')

filtered = []
for tr in data:
    ff1 = signal.filtfilt(filt[0], filt[1], tr[2][:,0], method='gust')
    ff2 = signal.filtfilt(filt[0], filt[1], tr[2][:,1], method='gust')
    filtered.append(np.column_stack((ff1, ff2)))

figure, axis = plt.subplots(2, 1)

axis[0].plot(x, filtered[t][:,0])
axis[0].set_title("Channel 1")

axis[1].plot(x, filtered[t][:,1])
axis[1].set_title("Channel 2")

plt.show()

# Calculate MFCC features
mfcc_feat = mfcc(filtered[t], fs, nfft = 2048)
plt.matshow(mfcc_feat.T)
plt.show()

features = []
for tr in filtered:
    mfcc_feat = mfcc(tr, fs, nfft = 2048)
    features.append(mfcc_feat.flatten())

# Build x and y arrays
x = np.array(features)
y = np.array([row[1] for row in data])

# Save processed data
processed_data = np.column_stack((y, x))
np.savetxt("speech_data.txt", processed_data)


#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------