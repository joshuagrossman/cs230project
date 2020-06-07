from constants import *
from util import *
import cnn
import clean_data
import numpy as np
import pretty_midi
import h5py


def generate_midi(filename, is_WAV):
	if is_WAV:
		assert(is_valid_file(filename, "wav"))
		cqt = convert_wav_to_cqt(wavPath)
	else:
		assert(is_valid_file(filename, "h5"))
		hf = h5py.File(filename, "r")
		cqt = np.array(hf.get("cqt"))

	# Create a PrettyMIDI object
	pm = pretty_midi.PrettyMIDI()
	# Create an Instrument instance for a cello instrument
	piano_program = pretty_midi.instrument_name_to_program('Piano')
	piano = pretty_midi.Instrument(program=piano_program)

	model = load_best_model()

	pianoroll = predicted_pianoroll(model, cqt)

	notes_list = get_notes(pianoroll) # each note is formatted as (note_name, start, end)

	for note_name, start, end in notes_list:
	    note_number = pretty_midi.note_name_to_number(note_name)
	    note = pretty_midi.Note(velocity=100, pitch=note_number, start=start, end=end)
	    piano.notes.append(note)

	# Add the piano instrument to the PrettyMIDI object
	pm.instruments.append(piano)
	# Write out the MIDI data
	pm.write('%s.mid' % filename.replace(".h5", ""))