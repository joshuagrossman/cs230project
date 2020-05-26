import os.path
import csv
import constants
import librosa
import librosa.display
import numpy as np
import pdb

def convert(wavPath, csvPath):
    y, sr = librosa.load(path=wavPath, sr=None, offset=0)
    print("Loaded WAV " + wavPath)
    binsPerOctave = 48
    C = np.abs(librosa.cqt(y, sr=sr, n_bins=constants.CONTEXT_WINDOW_ROWS, bins_per_octave=binsPerOctave, filter_scale=1))

    with open(csvPath, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for row in C:
            csvRow = [float(val) for val in row]
            writer.writerow(csvRow)
