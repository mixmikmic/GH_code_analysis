import tensorflow as tf
import numpy as np
import sys

from magenta.models.rl_tuner import rl_tuner
from magenta.models.rl_tuner import rl_tuner_ops

# Place to save your model checkpoints and composi|
SAVE_PATH = "/tmp/rl_tuner/"

# Model parameter settings
ALGORITHM = 'q'
REWARD_SCALER = 1
OUTPUT_EVERY_NTH = 50000
NUM_NOTES_IN_COMPOSITION = 32
PRIME_WITH_MIDI = False

rl_tuner_hparams = tf.contrib.training.HParams(random_action_probability=0.1,
                                               store_every_nth=1,
                                               train_every_nth=5,
                                               minibatch_size=32,
                                               discount_rate=0.5,
                                               max_experience=100000,
                                               target_network_update_rate=0.01)

reload(rl_tuner_ops)
reload(rl_tuner)
rl_tuner.reload_files()

rl_net = rl_tuner.RLTuner(SAVE_PATH, 
                          dqn_hparams=rl_tuner_hparams, 
                          algorithm=ALGORITHM,
                          reward_scaler=REWARD_SCALER,
                          output_every_nth=OUTPUT_EVERY_NTH,
                          num_notes_in_melody=NUM_NOTES_IN_COMPOSITION)

# Generate initial music sequence before training with RL
rl_net.generate_music_sequence(visualize_probs=True, title='pre_rl', length=32)

rl_net.train(num_steps=1000000, exploration_period=500000)

# Plot the rewards received during training. Improves as chance of random exploration action decreases.
rl_net.plot_rewards()

# Plot rewards received during calls to evaluation function throughout training. 
# Does not include exploration or random actions.
rl_net.plot_evaluation()

rl_net.generate_music_sequence(visualize_probs=True, title='post_rl')

# If you're happy with the model, save a version!
rl_net.save_model(SAVE_PATH, 'my_cool_model')

# Compute statistics about how well the model adheres to the music theory rules.
stat_dict = rl_net.evaluate_music_theory_metrics(num_compositions=100)

