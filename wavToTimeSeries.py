import os.path
import csv
import constants
import librosa
import librosa.display
import numpy as np
import pdb

# NOTE: Converts a WAV to a CQT matrix
def convert(wavPath, csvPath):
    y, sr = librosa.load(path=wavPath, sr=None, offset=0)
    print("Loaded WAV " + wavPath)
    # NOTE: This is where the magic happens. We might want to play around with this CQT generation process.
    C = np.abs(librosa.cqt(y, sr=sr, n_bins=constants.CONTEXT_WINDOW_ROWS, bins_per_octave=48, filter_scale=1))

    # NOTE: As with the pianoroll, we save the CQT matrix as a CSV.
    with open(csvPath, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for row in C:
            csvRow = [float(val) for val in row]
            writer.writerow(csvRow)
