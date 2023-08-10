import os
from magenta.models.performance_rnn import performance_sequence_generator
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2

import magenta.music as mm

# Constants.
BUNDLE_DIR = '/tmp/bundles'
MODEL_NAME = 'performance_with_dynamics'
BUNDLE_NAME = MODEL_NAME + '.mag'

mm.notebook_utils.download_bundle(BUNDLE_NAME, BUNDLE_DIR)
bundle = mm.sequence_generator_bundle.read_bundle_file(os.path.join(BUNDLE_DIR, BUNDLE_NAME))
generator_map = performance_sequence_generator.get_generator_map()
generator = generator_map[MODEL_NAME](checkpoint=None, bundle=bundle)
generator.initialize()
generator_options = generator_pb2.GeneratorOptions()
generator_options.args['temperature'].float_value = 1.0  # Higher is more random; 1.0 is default. 
generate_section = generator_options.generate_sections.add(start_time=0, end_time=30)
sequence = generator.generate(music_pb2.NoteSequence(), generator_options)

# Play and view this masterpiece.
mm.plot_sequence(sequence)
mm.play_sequence(sequence, mm.midi_synth.fluidsynth)

