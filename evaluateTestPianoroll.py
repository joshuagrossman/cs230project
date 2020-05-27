import numpy as np
import csv
import constants

# NOTE: This is a piecewise function for the loss of a note based on it being early/late.
l = lambda distInMs: 0 if distInMs < 20 else distInMs**0.8

# NOTE: In the evaluation process, detected notes are "searched" for the N-hundred milliseconds before and after the note.
# NOTE: If the note isn't found, just assume it's missing.
MAX_SEARCH_DIST_IN_MS = 600.0

# NOTE: The loss on a single note if it's not found within the search distance.
NOT_FOUND_LOSS = 167.0

# NOTE: Converts CQT time axis to milliseconds
MILLISECONDS_PER_STEP = 1.0 / constants.CQT_SAMPLING_RATE * 1000

# Converts output pianoroll into just a true followed by falses for onsets.
def format(outputPianoroll):
    # NOTE: Iterate through keyboard note by keyboard note
    for i, row in enumerate(outputPianoroll):
        isPrevOn = False
        # NOTE: Iterate through time slices for just this note's pianoroll
        for j, isOn in enumerate(row):
            # TODO: don't penalize an erroneous rearticulation as much as any note that doesn't belong
            if isOn:
                if isPrevOn:
                    outputPianoroll[i, j] = False
                isPrevOn = True
            else:
                isPrevOn = False

# NOTE: Delineates an algorithm for the performance of a complete piece of music.
# NOTE: pianoroll1 is assumed to be the truth, and pianoroll2 is being evaluated against it.
# NOTE: Depending on how you order your arguments, you can get numbers for our custom-defined
# NOTE: precision and recall. The output of this function is [0, 1] and measures how bad performance is.
def calculatePercentLoss(pianoroll1, pianoroll2, length):
    totalLoss = 0.0
    # Calculate max loss by pretending no notes are found
    maxTotalLoss = 0.0

    for rowIndex, row in enumerate(pianoroll1):
        # Calculate loss independently within each row
        for i, isOn in enumerate(row):
            if i == length: break
            if isOn:
                loss = 0.0
                isFound = False

                # Search right
                j = i
                while j < i + MAX_SEARCH_DIST_IN_MS / MILLISECONDS_PER_STEP:
                    if j >= len(pianoroll2[rowIndex]):
                        loss = NOT_FOUND_LOSS
                        break
                    if pianoroll2[rowIndex][j]:
                        loss = l((j - i) * MILLISECONDS_PER_STEP)
                        isFound = True
                        break
                    j += 1

                if not isFound:
                    # Search left
                    j = i - 1
                    while j > i - MAX_SEARCH_DIST_IN_MS / MILLISECONDS_PER_STEP:
                        if j < 0:
                            loss = NOT_FOUND_LOSS
                            break
                        if pianoroll2[rowIndex][j]:
                            # If the note is closer to the left, use that loss
                            loss = min(loss, l((i - j) * MILLISECONDS_PER_STEP))
                            isFound = True
                            break
                        j -= 1

                if not isFound:
                    loss = NOT_FOUND_LOSS

                totalLoss += loss
                maxTotalLoss += NOT_FOUND_LOSS

    return totalLoss / maxTotalLoss

# NOTE: Reads in two matching pianoroll files for evaluation.
def evaluate(outputPianorollPath, goldenPianorollPath):
    # Evaluate on precision and recall

    outputPianoroll = []
    goldenPianoroll = []
    with open(outputPianorollPath) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            outputPianoroll.append(map(int, row))
    with open(goldenPianorollPath) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            goldenPianoroll.append(map(int, row))

    length = min(len(goldenPianoroll[0]), len(outputPianoroll[0]))

    # Precision--iterate over output
    precision = 1.0 - calculatePercentLoss(outputPianoroll, goldenPianoroll, length)

    # Recall-iterate over golden
    recall = 1.0 - calculatePercentLoss(goldenPianoroll, outputPianoroll, length)

    return (precision, recall)