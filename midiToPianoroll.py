from __future__ import division
import sys
import argparse
import numpy as np
import pretty_midi
import librosa
import constants
import csv
import pdb

# library functions:
# pretty_midi.get_piano_roll(fs=100, times=None)

# take in a MIDI path name, time in milliseconds
# return 88 * 1

# regex 


def get_ground_truth_labels_at_time(piano_roll, time):
    """
    Takes in a piano roll (created with 1000Hz sampling rate) representation of a MIDI 
    and a time in milliseconds and returns a (88, 1) vector of ground truth labels.
    """
    # NOTE: The result is a vector of 1s and 0s
    if time >= len(piano_roll[0]):
        return False
    return piano_roll[:,time]


def get_piano_roll_subdivided_by_ms(midi_path, sampling_rate=constants.SAMPLE_RATE_IN_HZ):
    """
    Returns an np array that is a piano roll representation of a midi file, cropping the
    128 note values in MIDI to the 88 notes of a piano keyboard.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    raw_piano_roll = midi_data.get_piano_roll(fs=sampling_rate, times=None)
    piano_roll = raw_piano_roll > 0
    # 21 extra at bottom of range (C-1 -> G#0), 19 extra at top (C#8 -> G9)
    return np.asarray(piano_roll[19:-21]).astype(int) # TODO: look at piano check to make sure we're removing the 
                                    # right one (not flipped top/bottom)... else [21:-19]

def get_test_piano_roll(midi_filename):
    """
    Returns a np array that is a golden piano roll representation of a midi, divided into 1/87s slices
    """
    return get_piano_roll_subdivided_by_ms(midi_filename, constants.CQT_SAMPLE_RATE)

def writePianorollToFile(pianoroll, path):
    np.savetxt(path, pianoroll, fmt='%i', delimiter='\t')
    
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
        except Exception as e:
            print(e)
            continue
    
    # Get the relevant pianoroll slices
    for pieceID in pianorollIndicesToSample:
        midiFilename = os.path.join(midi_dir, pieceID) + ".midi"
        try:
            pianoroll = midiToPianoroll.get_piano_roll_subdivided_by_ms(midiFilename)
        except Exception as e:
            print(e)
            continue
        for sliceTimeInMs in pianorollIndicesToSample[pieceID]:
            pianorollSlice = midiToPianoroll.get_ground_truth_labels_at_time(pianoroll, sliceTimeInMs)
            
            # why is this here?
            if type(pianorollSlice) == bool:
                continue
                
            pianorollSlicePath = (os.path.join(pianoroll_dir, pieceID) + "_" + str(sliceTimeInMs) + ".csv").replace("\\", "/").replace("(", "").replace(")", "")
            try:
                midiToPianoroll.writePianorollToFile(pianorollSlice, pianorollSlicePath)
            except Exception as e:
                print(e)
                continue
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Converts a directory of MIDI to Pianorolls')
    parser.add_argument('cqt', help='Directory of CQT Slices to Match')
    parser.add_argument('midi', help='Input directory of MIDIs')
    parser.add_argument('pianoroll', help='Output directory of Pianorolls')
    args = parser.parse_args()
    convert_midi_to_pianoroll(args.cqt, args.midi, args.pianoroll)

    
    
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

# ### LOOK AT THIS
# def format_piano_roll_for_midi(piano_roll_path):
#     """
#     Takes in path to piano roll csv and reads csv into numpy array. Adds 19 rows pertaining 
#     to the highest values and 21 rows pertaining to the lowest values to the piano roll.
#     """
#     # NOTE: This is used to prepare a pianoroll matrix for conversion back into a midi, for listening.
#     small_piano_roll = np.genfromtxt(piano_roll_path, delimiter='\t')
#     piano_roll_length = small_piano_roll.shape[1]
#     piano_roll = np.insert(small_piano_roll, 0, np.zeros((19,piano_roll_length)), axis=0)
#     piano_roll = np.insert(piano_roll, piano_roll.shape[0], np.zeros((21,piano_roll_length)), axis=0)
#     return piano_roll.astype(int)

# ## LOOK AT THIS
# def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0): 
#     """
#     Convert a Piano Roll array into a PrettyMidi object
#     with a single instrument.

#     From https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py

#     Parameters
#     ----------
#     piano_roll : np.ndarray, shape=(128,frames), dtype=int
#         Piano roll of one instrument
#     fs : int
#         Sampling frequency of the columns, i.e. each column is spaced apart
#         by ``1./fs`` seconds.
#     program : int
#         The program number of the instrument.
#     Returns
#     -------
#     midi_object : pretty_midi.PrettyMIDI
#         A pretty_midi.PrettyMIDI class instance describing
#         the piano roll.
#     """
#     notes, frames = piano_roll.shape
#     pm = pretty_midi.PrettyMIDI()
#     instrument = pretty_midi.Instrument(program=program)

#     # pad 1 column of zeros so we can acknowledge inital and ending events
#     piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

#     # use changes in velocities to find note on / note off events
#     velocity_changes = np.nonzero(np.diff(piano_roll).T)

#     # keep track on velocities and note on times
#     prev_velocities = np.zeros(notes, dtype=int)
#     note_on_time = np.zeros(notes)

#     for time, note in zip(*velocity_changes):
#         # use time + 1 because of padding above
#         velocity = piano_roll[note, time + 1]
#         time = time / fs
#         if velocity > 0:
#             if prev_velocities[note] == 0:
#                 note_on_time[note] = time
#                 prev_velocities[note] = velocity
#         else:
#             pm_note = pretty_midi.Note(
#                 velocity=prev_velocities[note],
#                 pitch=note,
#                 start=note_on_time[note],
#                 end=time)
#             instrument.notes.append(pm_note)
#             prev_velocities[note] = 0
#     pm.instruments.append(instrument)
#     return pm

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
