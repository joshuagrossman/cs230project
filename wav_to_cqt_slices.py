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
import time
import h5py

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

def make_cqt_slices(wav_path, cqt_dir):
    """
    Converts a WAV file into a subdir of CQT slices to be stored in cqt_dir
    """
    wav_re = re.compile("\.wav$")
    if not wav_re.search(wav_path):
        print("Invalid WAV file: " + wav_path)
        return
    
    piece_id, cqt = convert_wav_to_cqt(wav_path)
    
    tot_time = cqt.shape[1]
    radius = constants.CQT_SLICE_RADIUS_IN_PIXELS
    offset = constants.CQT_SLICE_OFFSET_IN_PIXELS
    
    if not os.path.exists(cqt_dir):
        os.makedirs(cqt_dir)
    
    print("Saving slices for " + os.path.basename(wav_path))
    
    h5_name = os.path.join(cqt_dir, piece_id) + ".h5"
    with h5py.File(h5_name, 'w') as hf:
        
        starts = np.array(range(radius, tot_time-offset-radius-1, 2*radius+1)) 
        
        for t in starts:
            cqtSlice = cqt[:, (t+offset-radius):(t+offset+radius+1)]
            millisecondsStart = int(math.ceil(float(t) / constants.CQT_SAMPLING_RATE * 1000))
            hf.create_dataset(str(millisecondsStart), data=cqtSlice)

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Converts a directory of WAV to CQT Slices')
    parser.add_argument('wav_dir', help='Input directory of WAV files')
    parser.add_argument('cqt_dir', help='Output directory of CQT Slices')
    args = parser.parse_args()
    
    start_time = time.time()
    wav_paths = [os.path.join(args.wav_dir, wav_file) for wav_file in os.listdir(args.wav_dir)]
    
    # TURN DOWN CPU COUNT AS NECESSARY
    with mp.Pool(mp.cpu_count()) as pool:
        processes = [pool.apply_async(make_cqt_slices, args=(wav_path, args.cqt_dir)) for wav_path in wav_paths]
        [process.get() for process in processes]
    
    print(str(round(time.time() - start_time)) + " seconds to convert " + str(len(wav_paths)) + " WAV files to CQT slices.")
    