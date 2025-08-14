import streamlit as st
import numpy as np
import sounddevice as sd
import joblib
from scipy import signal
from python_speech_features import mfcc
import queue
import threading
import time

##### Data acquisition configuration #####

# Device configuration
fs = 44100
channels = 1  # Start with mono and try stereo if it fails
recording_time = 2

# Buffers for data acquisition
buffer_size = int(2 * fs * recording_time)
circular_buffer = np.zeros((buffer_size, channels), dtype='float64')
audio_queue = queue.Queue()

# Flag to stop the audio recording
stop_recording_flag = threading.Event()

# Callback for audio recording
def audio_callback(indata, frames, time, status):
    if status:
        st.error(f"Recording error: {status}")
    audio_queue.put(indata.copy())

# Function that runs in another thread for audio acquisition
def record_audio():
    global channels  # Declare channels as global
    try:
        with sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback):
            start_time = time.time()
            while not stop_recording_flag.is_set():
                if time.time() - start_time >= recording_time:
                    stop_recording_flag.set()
                    break
                try:
                    indata = audio_queue.get(timeout=0.1)
                    if indata is not None:
                        global circular_buffer
                        circular_buffer = np.roll(circular_buffer, -len(indata), axis=0)
                        circular_buffer[-len(indata):, :] = indata                     
                except queue.Empty:
                    continue
    except sd.PortAudioError as e:
        st.error(f"Mono channel error: {e}")
        try:
            channels = 2  # Try with stereo channels
            circular_buffer = np.zeros((buffer_size, channels), dtype='float64')  # Adjust circular buffer size
            with sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback):
                start_time = time.time()
                while not stop_recording_flag.is_set():
                    if time.time() - start_time >= recording_time:
                        stop_recording_flag.set()
                        break
                    try:
                        indata = audio_queue.get(timeout=0.1)
                        if indata is not None:
                            circular_buffer = np.roll(circular_buffer, -len(indata), axis=0)
                            circular_buffer[-len(indata):, :] = indata
                    except queue.Empty:
                        continue
        except sd.PortAudioError as e:
            st.error(f"Stereo channel error: {e}")
            st.error("Please check your microphone connection and try again.")
            stop_recording_flag.set()

# Function for stopping the audio acquisition
def stop_recording():
    stop_recording_flag.set()
    if recording_thread.is_alive():
        recording_thread.join()

# Helper function that obtains the last N seconds of audio recording
def get_last_n_seconds(n_seconds):
    samples = int(fs * n_seconds)
    return circular_buffer[-samples:]

# Load the trained model
model_path = 'trained_model_svm_linear.pkl'
clf_cv = joblib.load(model_path)

# Streamlit app
st.title("Real-time Audio Classification")

if st.button("Start Recording"):
    stop_recording_flag.clear()
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()
    st.write("Recording started...")

    # Wait for the recording to finish
    while recording_thread.is_alive():
        time.sleep(0.1)
    st.write("Recording stopped.")

if st.button("Classify Recording"):
    new_record = get_last_n_seconds(recording_time)

    # Process and classify new record
    # Filter the audio signal
    filt = signal.iirfilter(4, [10, 15000], rs=60, btype='band',
                            analog=False, ftype='cheby2', fs=fs,
                            output='ba')

    if channels == 1:
        ff = signal.filtfilt(filt[0], filt[1], new_record[:, 0], method='gust')
        filtered = np.column_stack((ff, ff))  # Duplicate mono channel to fit expected input format
    else:
        ff1 = signal.filtfilt(filt[0], filt[1], new_record[:, 0], method='gust')
        ff2 = signal.filtfilt(filt[0], filt[1], new_record[:, 1], method='gust')
        filtered = np.column_stack((ff1, ff2))

    # Calculate MFCC features
    mfcc_feat = mfcc(filtered, fs, nfft=2048)
    features = mfcc_feat.flatten()

    # Reshape for a single sample prediction
    features = features.reshape(1, -1)

    # Predict the category
    prediction = clf_cv.predict(features)
    
    # Mapping back to category names
    categories = {1: 'Perro', 2: 'Gato', 3: 'Jirafa', 4: 'Tortuga', 5: 'Avestruz', 6: 'Elefante'}
    predicted_category = categories[int(prediction[0])]

    st.write(f"Predicted category: {predicted_category}")

# Stop audio acquisition when the app is closed
if st.button("Exit"):
    stop_recording()
    st.write("Application stopped.")
