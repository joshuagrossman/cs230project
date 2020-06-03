#!/bin/bash
# Convert WAV -> CQT and MIDI -> Pianoroll

python3 wav_to_cqt_slices.py sample-wav-dir test-cqt
python3 midiToPianoroll.py test-cqt/starts sample-midi-dir test-pianoroll