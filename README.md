# Machine Music Transcription

Automatic Music Transcription (AMT) is the process of computing the symbolic musical representation of a musical audio recording. Out of the existing AMT solutions, none have reached the human-level accuracy on this task of nearly 100%. Although the problem formulation is simple, it requires a complex model to accurately detect individual notes from a noisy signal. We propose an approach that transforms this task to sequential image classification and leverages deep learning's superior ability to learn the structure within images and across sequences. Specifically, we constructed a bidirectional LSTM network with convolutional and pooling layers to symbolically classify audio representations of classical piano music.

This code works on the MAESTRO dataset, which contains classical piano performances recorded in both WAV and MIDI format, which we use to train our model.

## To get started

#### To train:

`python main.py <processed h5 file directory> train`

#### To get a sample MIDI transcription:

`python main.py <processed h5 file directory> midi --h5-file <h5 file to convert>`

The name of the generated file is specified in the console output, which you can scp from your local machine and play back with your default audio player.

#### To see the full list of options:

`python main.py -h`

## File structure

`main.py` - main program loop

`cnn.py` - specifies CNN model

`train.py` - trains the CNN

`evaluate.py` - functions related to evaluation, including MIDI generation

`constants.py` - all constants

`util.py` - utility functions

`clean_data.py` - data processing to generate WAV and CQT slices for neural network input

`noisifyMidis.py` - currently unused functions for noisifying training data
