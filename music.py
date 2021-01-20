import librosa
import sys
import numpy as np
from math import log2, pow

A4 = 440
C0 = A4*pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
def pitch(freq):
    notes_count = {"C":0, "C#":0, "D":0, "D#":0, "E":0, "F":0, "F#":0, "G":0, "G#":0, "A":0, "A#":0, "B":0}
    for i in freq:
        if i>0:
            h = round(12*log2(i/C0))
            octave = h // 12
            n = h % 12
            notes_count[name[n]] += 1
    return notes_count

np.set_printoptions(threshold=sys.maxsize)

filename = 'melody-1.wav'
Fs = 44100
clip, sample_rate = librosa.load(filename, sr=Fs)

n_fft = 1024 
start = 0 

hop_length=512

X = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)

t_samples = np.arange(clip.shape[0]) / Fs
t_frames = np.arange(X.shape[1]) * hop_length / Fs

f_hertz = np.fft.rfftfreq(n_fft, 1 / Fs)         

print('Time (seconds) :', len(t_samples))

print('Frequency (Hz) of each note [first 20 values]: ', (list(f_hertz[0:20])))

print('Number of frames : ', len(t_frames))

print('Frequency of each note in octave music : ', pitch(list(f_hertz)))