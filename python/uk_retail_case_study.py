import featuretools as ft
from utils import (find_training_examples, load_uk_retail_data, 
                   engineer_features_uk_retail, preview, feature_importances)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
ft.__version__

item_purchases, invoices, items, customers = load_uk_retail_data()

entities = {
        "item_purchases": (item_purchases, "item_purchase_id", "InvoiceDate" ),
        "items": (items, "StockCode"),
        "customers": (customers,"CustomerID"),
        "invoices":(invoices,"InvoiceNo", "first_item_purchases_time")
        }

relationships = [("customers", "CustomerID","invoices", "CustomerID"), 
                ("invoices", "InvoiceNo","item_purchases", "InvoiceNo"),
                ("items", "StockCode","item_purchases", "StockCode")]

label_times = find_training_examples(item_purchases, invoices,
                                     prediction_window=pd.Timedelta("14d"),
                                     training_window=pd.Timedelta("21d"),
                                     lead=pd.Timedelta("7d"),
                                     threshold=5)

preview(label_times, 5)

feature_matrix = engineer_features_uk_retail(entities, relationships,
                                             label_times, training_window='21d')

preview(feature_matrix, 10)

label_times[["CustomerID"]]
X_y = feature_matrix.merge(label_times[["CustomerID", 'purchases>threshold']], right_on="CustomerID", left_index=True)
y = X_y.pop('purchases>threshold')
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, 
                                                    y, test_size=0.35)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(X_train)
X_train_imp = imp.transform(X_train)

clf = RandomForestClassifier(random_state=0, n_estimators=100,
                             class_weight="balanced", verbose=True)
clf.fit(X_train_imp, y_train)

X_test_imp = imp.transform(X_test)
predicted_labels = clf.predict(X_test_imp)

tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels).ravel()

tn, fp, fn, tp

feature_importances(clf, feature_matrix.columns, n=15)

