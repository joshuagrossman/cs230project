import os
import re
import collections
import time
import constants
import wavToTimeSeries
import timeSeriesToCqtSlice
import midiToPianoroll
import pdb

def convert_midi_to_pianoroll(cqt_dir, midi_dir, pianoroll_dir):
    # Convert all MIDI to pianoroll labeled the same as CQT slices
    
    # Construct dict of pieceID -> list of ms offsets
    pianorollIndicesToSample = collections.defaultdict(list)
    
    for cqtSlicePath in os.listdir(cqt_dir):
        
        # ignore hidden files and non-csv
        if (cqtSlicePath[0] == "." or cqtSlicePath[-4:] != ".csv" ):
            continue
            
        cqtSlicePath = os.path.join(cqt_dir, cqtSlicePath).replace("\\", "/")
        p = re.compile("\/(.+)\_([0-9]+)\.csv$")
        file_match = p.search(cqtSlicePath)
        try:
            pieceID = file_match.group(1)
            sliceTimeInMs = int(file_match.group(2))
            pianorollIndicesToSample[pieceID].append(sliceTimeInMs)
        except:
            print("Could not match CQT file " + cqtSlicePath)
            continue

    # Get the relevant pianoroll slices
    for pieceID in pianorollIndicesToSample:
        midiFilename = os.path.join(midi_dir, pieceID) + ".mid"
        try:
            pianoroll = midiToPianoroll.get_piano_roll_subdivided_by_ms(midiFilename)
        except:
            print("Could not locate " + midiFilename)
            continue
        for sliceTimeInMs in pianorollIndicesToSample[pieceID]:
            pianorollSlice = midiToPianoroll.get_ground_truth_labels_at_time(pianoroll, sliceTimeInMs)
            if type(pianorollSlice) == bool:
                continue
            pianorollSlicePath = (os.path.join(pianoroll_dir, pieceID) + "_" + str(sliceTimeInMs) + ".csv").replace("\\", "/").replace("(", "").replace(")", "")
            try:
                midiToPianoroll.writePianorollToFile(pianorollSlice, pianorollSlicePath)
            except:
                print("Could not locate " + pianorollSlicePath)
                continue
    return

convert_midi_to_pianoroll("sample-cqt-dir", "sample-midi-dir", "sample-pianoroll-dir")
        
        
# #### Export MIDI as audio

#     # Generate MIDIs from pianoroll for human listening
#     # for pieceID in pieceIDs:
#     #     pianorollFilename = (constants.PIANOROLL_NAME_TEMPLATE % (pieceID, "")).replace("(", "").replace(")", "")
#     #     outputPianorollPath = (constants.TEST_PIANOROLL_OUT_DIR + pianorollFilename).replace("\\", "/")
#     #     midiPath = os.path.join(constants.TEST_MIDI_OUT_DIR, constants.MIDI_OUT_NAME_TEMPLATE % pieceID).replace("\\", "/")
#     #     pianoroll = midiToPianoroll.format_piano_roll_for_midi(outputPianorollPath)
#     #     prettyMidi = midiToPianoroll.piano_roll_to_pretty_midi(pianoroll, constants.CQT_SAMPLING_RATE)
#     #     prettyMidi.write(midiPath)
#     # print("MIDIs generated for human listening!\n" 