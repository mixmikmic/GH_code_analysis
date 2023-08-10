# IO:
import pickle
# Dataset:
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# Preprocessing:
from sklearn.preprocessing import StandardScaler
# Modeling:
import tensorflow as tf
# Evaluation:
from sklearn.metrics import r2_score

# Load dataset:
dataset = fetch_california_housing()
# Features and targets:
features = dataset.data
targets = dataset.target

# Split training & testing sets:
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, 
    test_size = 0.20, 
    random_state=42
)

