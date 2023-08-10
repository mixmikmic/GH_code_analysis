get_ipython().run_cell_magic('capture', '', '\nimport os\nimport pickle\nimport numpy as np\n\nimport src.model_ensemble as ensemble\n\nfrom src.data.loaders import load_and_clean_data\nfrom src.definitions import ROOT_PATH\nfrom src.definitions import TEST_PATH\n\nfrom scipy.optimize import minimize\n\nROWS = 10000\n\nmodels = ensemble.init_models()\n\ntokenizer_path = os.path.join(\n    ROOT_PATH, "models/{}".format("tokenizer.pkl"))\n\nwith open(tokenizer_path, "rb") as file:\n    tokenizer = pickle.load(file)\n\n# Load validation reviews\nval_samples, val_labels = load_and_clean_data(path=TEST_PATH, nrows=ROWS)\nsequences = tokenizer.texts_to_sequences(val_samples)')

predictions = ensemble.models_prediction(sequences, val_labels, models)
accuracies = np.array([np.mean(np.round(pred) == val_labels) for pred in predictions])

SCALE_FACTOR = -100.0

def objective_function(x):
    ensemble_predictions = ensemble.ensemble_prediction(predictions, weights=x)
    ensemble_accuracy = np.mean(ensemble_predictions == val_labels)
    
    value = SCALE_FACTOR * ensemble_accuracy
    grads = -accuracies
    return value, grads

x0 = np.zeros((len(predictions), 1)) / len(predictions)
bounds = [(0, 1)] * len(predictions)
constraints = [{
    'type': 'eq',
    'fun': lambda x: 1.0 - np.sum(x) 
}]

result = minimize(objective_function, 
                  x0, 
                  jac=True, 
                  method='SLSQP', 
                  bounds=bounds,
                  constraints=constraints,
                  tol=1e-7, 
                  options={'disp': True})

print(result.x)
print(result.success)
print(result.message)

test_samples, test_labels = load_and_clean_data(path=TEST_PATH)
sequences = tokenizer.texts_to_sequences(test_samples)
model_predictions = ensemble.models_prediction(sequences, test_labels, models)

ensemble_prediction = ensemble.ensemble_prediction(model_predictions)
mean_ensemble_accuracy = np.mean(ensemble_prediction == test_labels)
print("Mean ensemble accuracy: {:.5f}".format(mean_ensemble_accuracy))

ensemble_prediction = ensemble.ensemble_prediction(model_predictions, weights=result.x)
weighted_ensemble_accuracy = np.mean(ensemble_prediction == test_labels)
print("Weighted mean ensemble accuracy: {:.5f}".format(weighted_ensemble_accuracy))

