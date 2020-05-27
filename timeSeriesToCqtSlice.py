import csv
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import constants
import pdb
import os

def onsetDetection(C):
    # Base onset detection on onset detection with top two octaves=top 60 bins (n_bins=60*6, bins_per_octave=12*4)
    gradientAllowance = 3 # number of frames over which a transition is allowed
    onsetDetectionThreshold = 1.3 # we have an onset if we increase by more than a factor of X over 4 frames or less
    onsetDetectionData = C[-24:, :]

    averageOnsetDetectionData = np.sum(onsetDetectionData, axis=0)

    onsets = np.array([False] * len(averageOnsetDetectionData))

    buff = [-1] * gradientAllowance
    index = 0
    while index < len(averageOnsetDetectionData):
        d = averageOnsetDetectionData[index]
        buff = buff[1:] + [d]

        # Reject smallest sample
        mean = float(sum(buff) - min(buff)) / (len(buff) - 1)
        if d / mean > onsetDetectionThreshold and d > 0.02:
            onsets[index] = True
            index += 15
        index += 1

    # Sanity check display CQT
    # librosa.display.specshow(librosa.amplitude_to_db(C[:, :100], ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')

    return onsets

def writeToFile(cqtSlice, cqtSliceDir, pieceID, millisecondsOffset):
    csvPath = (cqtSliceDir + "/" + pieceID + "_" + str(millisecondsOffset) + ".csv").replace("(", "").replace(")", "")
    csvContents = np.asarray([[float(val) for val in row] * 9 for row in cqtSlice])
    np.savetxt(csvPath, csvContents, fmt='%.10f', delimiter="\t")

    return csvPath

def convertTest(csvPath, cqtSliceDir, pieceID):
    C = []

    with open(csvPath) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            C.append([float(val) for val in row])
    
    return C

def convert(csvPath, cqtSliceDir):
    C = []
    cqtSlicePaths = []
    
    pieceID = os.path.basename(csvPath).replace(".csv", "")
    
    with open(csvPath) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            C.append([float(val) for val in row])
    
    C = np.array(C)

    onsets = onsetDetection(C)

    print(str(len(onsets)) + " onsets found for " + pieceID + ".")

    for i, onset in enumerate(onsets):
        if onset:
            cqtSlice = C[:, i+1:(i+1+constants.CQT_SLICE_WIDTH_IN_PIXELS)]
            millisecondsOffset = int(math.ceil(float(i) / constants.CQT_SAMPLING_RATE * 1000))
            csvPath = writeToFile(cqtSlice, cqtSliceDir, pieceID, millisecondsOffset)
            cqtSlicePaths.append(csvPath)

    return cqtSlicePaths

def convert_dir(time_dir, cqt_dir):
    for timeFilename in os.listdir(time_dir):
        timePath = os.path.join(time_dir, timeFilename).replace("\\", "/")
        csvPath = cqt_dir + "/" + timeFilename
        convert(timePath, cqt_dir)