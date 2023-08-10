from concise.data import attract

dfa = attract.get_metadata()
dfa

dfa_pum2 = dfa[dfa.Gene_name.str.match("PUM2") &                dfa.Organism.str.match("Homo_sapiens") &                (dfa.Database != "PDB") &                (dfa.Experiment_description == "genome-wide in vivo immunoprecipitation")]

dfa_pum2

from concise.utils.pwm import PWM
pwm_list = attract.get_pwm_list(dfa_pum2.PWM_id.unique())

for pwm in pwm_list:
    pwm.plotPWM(figsize=(8,1.5))

import pandas as pd

from concise.preprocessing import encodeDNA, EncodeSplines

def load(split="train", st=None):
    dt = pd.read_csv("../data/RBP/PUM2_{0}.csv".format(split))
    # DNA/RNA sequence
    xseq = encodeDNA(dt.seq) 
    # distance to the poly-A site
    xpolya = dt.polya_distance.as_matrix().reshape((-1, 1))
    # response variable
    y = dt.binding_site.as_matrix().reshape((-1, 1)).astype("float")
    return {"seq": xseq, "dist_polya_raw": xpolya}, y

def data():
    
    train, valid, test = load("train"), load("valid"), load("test")
    
    # transform the poly-A distance with B-splines
    es = EncodeSplines()
    es.fit(train[0]["dist_polya_raw"])
    train[0]["dist_polya_st"] = es.transform(train[0]["dist_polya_raw"])
    valid[0]["dist_polya_st"] = es.transform(valid[0]["dist_polya_raw"])
    test[0]["dist_polya_st"] = es.transform(test[0]["dist_polya_raw"])
    
    #return load("train"), load("valid"), load("test")
    return train, valid, test

train, valid, test = data()

import concise.layers as cl
import keras.layers as kl
import concise.initializers as ci
import concise.regularizers as cr
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping

train[0]["seq"].shape

train[0].keys()

def model(train, filters=1, kernel_size=9, pwm_list=None, lr=0.001, use_splinew=True, ext_dist=False):
    seq_length = train[0]["seq"].shape[1]
    if pwm_list is None:
        kinit = "glorot_uniform"
        binit = "zeros"
    else:
        kinit = ci.PSSMKernelInitializer(pwm_list, add_noise_before_Pwm2Pssm=True)
        binit = "zeros"
        
    # sequence
    in_dna = cl.InputDNA(seq_length=seq_length, name="seq")
    inputs = [in_dna]
    x = cl.ConvDNA(filters=filters, 
                   kernel_size=kernel_size, 
                   activation="relu",
                   kernel_initializer=kinit,
                   bias_initializer=binit,
                   name="conv1")(in_dna)
    if use_splinew:
        x = cl.SplineWeight1D(n_bases=10, l2_smooth=0, l2=0, name="spline_weight")(x)
        x = kl.GlobalAveragePooling1D()(x)
    else:
        x = kl.AveragePooling1D(pool_size=4)(x)
        x = kl.Flatten()(x)
        
    if ext_dist:    
        # distance
        in_dist = kl.Input(train[0]["dist_polya_st"].shape[1:], name="dist_polya_st")
        x_dist = cl.SplineT()(in_dist)
        x = kl.concatenate([x, x_dist])
        inputs += [in_dist]
    
    x = kl.Dense(units=1)(x)
    m = Model(inputs, x)
    m.compile(Adam(lr=lr), loss="binary_crossentropy", metrics=["acc"])
    return m

m = model(train, filters=10, use_splinew=False, ext_dist=True)

m.fit(train[0], train[1], epochs=50, validation_data=valid, 
     callbacks=[EarlyStopping(patience=5)])

m.get_layer("conv1").plot_weights(plot_type="motif_pwm_info")

m = model(train, filters=16, pwm_list=pwm_list, use_splinew=False, ext_dist=True)

m.fit(train[0], train[1], epochs=50, validation_data=valid, 
     callbacks=[EarlyStopping(patience=5)])

m.get_layer("conv1").plot_weights(plot_type="motif_pwm_info")

# we can see that the performance is better using the initialized motif

get_ipython().set_next_input('m.get_layer("spline_weight").*');get_ipython().magic('psearch *')

m.get_layer("spline_weight").get_weights()



