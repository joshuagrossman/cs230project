import os.path
import constants
import librosa
import librosa.display
import numpy as np
import pdb
import argparse
import re
import math
import os
import multiprocessing as mp

def convert_wav_to_cqt(wavPath):
    """
    Converts a WAV file to a CQT representation
    """
    y, sr = librosa.load(path=wavPath, sr=None, offset=0)
    print("Loaded WAV " + wavPath)
    piece_id = os.path.basename(wavPath).replace(".wav", "")
    cqt = np.abs(librosa.cqt(y, sr=sr, n_bins=constants.CONTEXT_WINDOW_ROWS, 
                           bins_per_octave=constants.BINS_PER_OCTAVE, 
                           hop_length=constants.HOP_LENGTH, 
                           filter_scale=constants.FILTER_SCALE))
    return piece_id, cqt

def save_cqt_slice(cqtSlice, cqtSliceDir, pieceID, millisecondsOffset):
    """
    Saves a CQT slice as a binary file.
    """
    slice_path = (cqtSliceDir + "/" + pieceID + "_" + str(millisecondsOffset) + ".bin")
    pdb.set_trace()
    cqtSlice.tofile(slice_path)
    return

def make_cqt_slices(wavPath, cqtSliceDir):
    """
    Converts a WAV file into a directory of CQT slices
    """
    piece_id, cqt = convert_wav_to_cqt(wavPath)
    cqtSlicePaths = []
    
    tot_time = cqt.shape[1]
    radius = constants.CQT_SLICE_RADIUS_IN_PIXELS
    offset = constants.CQT_SLICE_OFFSET_IN_PIXELS
    
    print("Saving slices for " + os.path.basename(wavPath))
    for t in range(radius, tot_time-offset-radius-1, 2*radius+1):
        cqtSlice = cqt[:, (t+offset-radius):(t+offset+radius+1)]
        millisecondsOffset = int(math.ceil(float(t) / constants.CQT_SAMPLING_RATE * 1000))
        save_cqt_slice(cqtSlice, cqtSliceDir, piece_id, millisecondsOffset)

    return 

def convert_dir(wav_dir, cqt_dir):
    """
    Converts a directory of WAV files into a directory of CQT slices
    """
    wav_re = re.compile("\.wav$")
    for wavFilename in os.listdir(wav_dir):
        if not wav_re.search(wavFilename):
            print("Invalid file: " + wavFilename)
            continue
        wavPath = os.path.join(wav_dir, wavFilename)
        make_cqt_slices(wavPath, cqt_dir)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Converts a directory of WAV to CQT Slices')
    parser.add_argument('wav_dir', help='Input directory of WAV')
    parser.add_argument('cqt_dir', help='Output directory of CQT Slices')
    args = parser.parse_args()
    
    #with mp.Pool(int(mp.cpu_count() / 2)) as pool:
    
    convert_dir(args.wav_dir, args.cqt_dir)
    
    
    