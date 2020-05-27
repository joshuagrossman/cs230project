import csv
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import constants
import pdb

# NOTE: Detect when new notes are played based on peaks in the average energy of the upper harmonics.
def onsetDetection(C):
    # Base onset detection on onset detection with top two octaves=top 60 bins (n_bins=60*6, bins_per_octave=12*4)
    gradientAllowance = 3 # number of frames over which a transition is allowed
    onsetDetectionThreshold = 1.3 # we have an onset if we increase by more than a factor of X over 4 frames or less
    # NOTE: Just look at the top 24 frequency bins.
    onsetDetectionData = C[-24:, :]

    averageOnsetDetectionData = np.sum(onsetDetectionData, axis=0)

    # NOTE: 1D array that stores when an onset is detected
    onsets = np.array([False] * len(averageOnsetDetectionData))

    # NOTE: Circular buffer that keeps track of the last three total energy values
    buff = [-1] * gradientAllowance
    index = 0
    while index < len(averageOnsetDetectionData):
        d = averageOnsetDetectionData[index]
        # NOTE: Add to buffer
        buff = buff[1:] + [d]

        # Reject smallest sample
        mean = float(sum(buff) - min(buff)) / (len(buff) - 1)
        # NOTE: Call this an onset if this energy level is a certain amount higher than before.
        if d / mean > onsetDetectionThreshold and d > 0.02:
            onsets[index] = True
            # NOTE: Skip ahead 15 frames, because notes typically don't go this fast
            index += 15
        index += 1

    # Sanity check display CQT
    # NOTE: This allows you to see onset detection working.
    # librosa.display.specshow(librosa.amplitude_to_db(C[:, :100], ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')

    return onsets

def writeToFile(cqtSlice, cqtSliceDir, pieceID, millisecondsOffset):
    csvPath = (constants.CQT_SLICE_NAME_TEMPLATE % (cqtSliceDir + "/" + pieceID, str(millisecondsOffset))).replace("(", "").replace(")", "")
    csvContents = np.asarray([[float(val) for val in row] * 9 for row in cqtSlice])
    np.savetxt(csvPath, csvContents, fmt='%.10f', delimiter="\t")

    return csvPath

# NOTE: Creates CQT slices for input to the model, saves them as a file, and returns their path.
def convert(csvPath, cqtSliceDir, pieceID):
    C = []
    cqtSlicePaths = []

    with open(csvPath) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            C.append([float(val) for val in row])
    
    C = np.array(C)

    onsets = onsetDetection(C)

    print(str(len(onsets)) + " onsets found for " + pieceID + ".")

    for i, onset in enumerate(onsets):
        if onset:
            # NOTE: If we see an onset here, use a slice starting at the NEXT frame just to be safe.
            cqtSlice = C[:, i+1:(i+1+constants.CQT_SLICE_WIDTH_IN_PIXELS)]
            millisecondsOffset = int(math.ceil(float(i) / constants.CQT_SAMPLING_RATE * 1000))
            csvPath = writeToFile(cqtSlice, cqtSliceDir, pieceID, millisecondsOffset)
            cqtSlicePaths.append(csvPath)

    return cqtSlicePaths


def convertTest(csvPath, cqtSliceDir, pieceID):
    C = []

    with open(csvPath) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            C.append([float(val) for val in row])
    
    return C
