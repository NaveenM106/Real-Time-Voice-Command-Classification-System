import time
import queue
import random
import threading

import numpy as np
import sounddevice as sd

import pickle
from datetime import datetime

print(sd.query_devices())  

# Experiment configuration
conditions = [('Perro', 1), ('Gato', 2), ('Jirafa', 3), ('Tortuga', 4), ('Avestruz', 5), ('Elefante', 6)]
n_trials = 30

fixation_cross_time = 0.5
preparation_time = 0.3
trial_time = 2
rest_time = 0.5

trials = n_trials * conditions
random.shuffle(trials)

# Device configuration
fs = 44100    
channels = 1  # Start with mono and try stereo if it fails

# Buffers for data acquisition
buffer_size = int(2 * fs * trial_time)
circular_buffer = np.zeros((buffer_size, channels), dtype='float64')
audio_queue = queue.Queue()

# Flag to stop the audio recording
stop_recording_flag = threading.Event()

# Callback for audio recording
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Function that runs in another thread for audio acquisition
def record_audio():
    global channels
    try:
        with sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback):
            while not stop_recording_flag.is_set():
                try:
                    indata = audio_queue.get(timeout=0.1)
                    if indata is not None:                    
                        global circular_buffer
                        circular_buffer = np.roll(circular_buffer, -len(indata), axis=0)
                        circular_buffer[-len(indata):, :] = indata                     
                except queue.Empty:
                    continue
    except sd.PortAudioError as e:
        print(f"Mono channel error: {e}")
        try:
            channels = 2  # Try with stereo channels
            circular_buffer = np.zeros((buffer_size, channels), dtype='float64')  # Adjust circular buffer size
            with sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback):
                while not stop_recording_flag.is_set():
                    try:
                        indata = audio_queue.get(timeout=0.1)
                        if indata is not None:                    
                            circular_buffer = np.roll(circular_buffer, -len(indata), axis=0)
                            circular_buffer[-len(indata):, :] = indata                     
                    except queue.Empty:
                        continue
        except sd.PortAudioError as e:
            print(f"Stereo channel error: {e}")
            print("Please check your microphone connection and try again.")
            stop_recording_flag.set()

# Function for stopping the audio acquisition
def stop_recording():
    stop_recording_flag.set()
    recording_thread.join()

# Helper function that obtains the last N seconds of audio recording
def get_last_n_seconds(n_seconds):
    samples = int(fs * n_seconds)
    return circular_buffer[-samples:]

# Start data acquisition
recording_thread = threading.Thread(target=record_audio, daemon=True)
recording_thread.start()

# Run experiment
print("********* Experiment in progress *********")    
time.sleep(fixation_cross_time)    

data = []
count = 0
for t in trials:
    # Fixation cross    
    count += 1
    print(f"\n********* Audio {count}/{len(trials)} *********")    
    time.sleep(fixation_cross_time)    
    
    # Preparation time
    print(t[0])
    time.sleep(preparation_time)

    # Task    
    time.sleep(trial_time)
    recording = get_last_n_seconds(trial_time)
    data.append((t[0], t[1], recording))

    # Rest time    
    print("----Rest----")
    time.sleep(rest_time)

# Stop audio acquisition
stop_recording()

# Play records
count = 0    
for t in data:
    count += 1
    print(f"Playing: {t[0]} ({count}/{len(data)})")
    sd.play(t[2], fs)
    sd.wait()

# Save data
now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
outputFile = open(now + '.obj', 'wb')
pickle.dump(data, outputFile)
outputFile.close()
