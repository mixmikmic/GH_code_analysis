from effect_demo_setup import *
from concise.models import single_layer_pos_effect as concise_model
import numpy as np

# Generate training data for the model, use a 1000bp sequence
param, X_feat, X_seq, y, id_vec = load_example_data(trim_seq_len = 1000)

# Generate the model
dc = concise_model(pooling_layer="sum",
                   init_motifs=["TGCGAT", "TATTTAT"],
                   n_splines=10,
                   n_covariates=0,
                   seq_length=X_seq.shape[1],
                   **param)

# Train the model
dc.fit([X_seq], y, epochs=1,
       validation_data=([X_seq], y))
    
# In order to select the right output of a potential multitask model we have to generate a list of output labels, which will be used alongside the model itself.
model_output_annotation = np.array(["output_1"])

import h5py

dataset_path = "%s/data/sample_hqtl_res.hdf5"%concise_demo_data_path
dataset = {}
with h5py.File(dataset_path, "r") as ifh:
    ref = ifh["test_in_ref"].value
    alt = ifh["test_in_alt"].value
    dirs = ifh["test_out"]["seq_direction"].value
    
    # This datset is stored with forward and reverse-complement sequences in an interlaced manner
    assert(dirs[0] == b"fwd")
    dataset["ref"] = ref[::2,...]
    dataset["alt"] = alt[::2,...]
    dataset["ref_rc"] = ref[1::2,...]
    dataset["alt_rc"] = alt[1::2,...]
    dataset["y"] = ifh["test_out"]["type"].value[::2]
    
    # The sequence is centered around the mutatiom with the mutation occuring on position when looking at forward sequences
    dataset["mutation_position"] = np.array([500]*dataset["ref"].shape[0])

from concise.effects.ism import ism
from concise.effects.gradient import gradient_pred
from concise.effects.dropout import dropout_pred

ism_result = ism(model = dc, 
                 ref = dataset["ref"], 
                 ref_rc = dataset["ref_rc"], 
                 alt = dataset["alt"], 
                 alt_rc = dataset["alt_rc"], 
                 mutation_positions = dataset["mutation_position"], 
                 out_annotation_all_outputs = model_output_annotation, diff_type = "diff")
gradient_result = gradient_pred(model = dc, 
                                ref = dataset["ref"], 
                                ref_rc = dataset["ref_rc"], 
                                alt = dataset["alt"], 
                                alt_rc = dataset["alt_rc"], 
                                mutation_positions = dataset["mutation_position"], 
                                out_annotation_all_outputs = model_output_annotation)
dropout_result = dropout_pred(model = dc, 
                              ref = dataset["ref"], 
                              ref_rc = dataset["ref_rc"], 
                              alt = dataset["alt"], alt_rc = dataset["alt_rc"], mutation_positions = dataset["mutation_position"], out_annotation_all_outputs = model_output_annotation)

gradient_result

from concise.effects.snp_effects import effect_from_model

# Define the parameters:
params = {"methods": [gradient_pred, dropout_pred, ism],
         "model": dc,
         "ref": dataset["ref"],
         "ref_rc": dataset["ref_rc"],
         "alt": dataset["alt"],
         "alt_rc": dataset["alt_rc"],
         "mutation_positions": dataset["mutation_position"],
         "extra_args": [None, {"dropout_iterations": 60},
         		{"rc_handling" : "maximum", "diff_type":"diff"}],
         "out_annotation_all_outputs": model_output_annotation,
         }

results = effect_from_model(**params)

print(results.keys())



