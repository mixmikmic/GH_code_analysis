import pandas as pd
import sys
sys.path.append('../Src')
from nn_extractor import NNExtractor

network = NNExtractor()

network.config["dataset"]

network.config["network"]

network.output_image_dir

features = network.extract_features()

features = features.transpose().reset_index()

features["i"] = features["index"].str.slice(0,5)
features["j"] = features["index"].str.slice(6,10)

features["i"]=pd.to_numeric(features["i"])
features["j"]=pd.to_numeric(features["j"])
features.head()

features.to_csv("../Data/Features/{}_{}_{}_{}.csv".format(network.config["dataset"]["source"],                                                        network.config["dataset"]["country"],                                                        network.config["network"]["model"],                                                        network.config["network"]["layer"] ))



