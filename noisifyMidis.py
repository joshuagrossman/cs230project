import librosa
import matplotlib.pyplot as plt
import numpy as np
import random
import pretty_midi
from scipy.signal import butter, lfilter, filtfilt, freqz
import constants
import os

def noisifyMidi(midiPath):
    midi_data = pretty_midi.PrettyMIDI(midiPath)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Set random velocity change to replicate variation in performance
            velocityChange = random.randint(-30, 30)
            newVelocity = min(max(note.velocity + velocityChange, 20), 127)
            note.velocity = newVelocity

            # Set pitch bend for the duration of this note to replicate inexact piano tuning
            pitchBendAmount = random.randint(-2, 2)
            instrument.pitch_bends.append(pretty_midi.PitchBend(pitchBendAmount, note.start))

    # Overwrite MIDI
    midi_data.write(midiPath)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def noisifyWav(wavPath):
    order = 6

    data, sr = librosa.load(path=wavPath, sr=None)

    # Set random low pass variable to replicate differences in piano timbre and microphone input
    lowpassCutoff = 2**np.random.uniform(10.5, 14.4)
    y = butter_lowpass_filter(data, lowpassCutoff, sr, order)

    # Set random static variable to replicate the graininess of recordings
    staticMagnitude = np.random.uniform(0, 0.0005)
    for d in range(len(y)):
        y[d] += random.choice([-staticMagnitude, staticMagnitude])

    # Overwrite WAV
    librosa.output.write_wav(wavPath, y, sr)

if __name__ == '__main__':
    isMidi = False

    if isMidi:
        for midiFilename in os.listdir(constants.WTC1_MIDI_DIR):
            midiFilenamePath = os.path.join(constants.WTC1_MIDI_DIR, midiFilename)
            noisifyMidi(midiFilenamePath)
            print midiFilename
        for midiFilename in os.listdir(constants.WTC2_MIDI_DIR):
            midiFilenamePath = os.path.join(constants.WTC2_MIDI_DIR, midiFilename)
            noisifyMidi(midiFilenamePath)
            print midiFilename
    else:
        for wavFilename in os.listdir(constants.WTC1_WAV_DIR):
            wavFilenamePath = os.path.join(constants.WTC1_WAV_DIR, wavFilename)
            noisifyWav(wavFilenamePath)
            print wavFilename

        for wavFilename in os.listdir(constants.WTC2_WAV_DIR):
            wavFilenamePath = os.path.join(constants.WTC2_WAV_DIR, wavFilename)
            noisifyWav(wavFilenamePath)
            print wavFilename
