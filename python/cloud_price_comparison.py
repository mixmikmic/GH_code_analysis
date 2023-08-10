import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')

catalyst_dataset = pd.read_csv("dataset/Rigorious Cloud price comparison - Exportable Catalyst prices.csv")
google_dataset = pd.read_csv("dataset/Rigorious Cloud price comparison - Exportable Google prices.csv")
aws_dataset = pd.read_csv("dataset/Rigorious Cloud price comparison - Exportable AWS prices.csv")
azure_dataset = pd.read_csv("dataset/Rigorious Cloud price comparison - Exportable Azure prices.csv")

catalyst_dataset.head(6)

def split_dataset (dataset):
    x = dataset[["vCPUs", "RAM (GiB)", "HDD (GB)", "SSD (GB)"]].values
    y = dataset["Price per hour (NZD)"].values
    
    return (x, y)

catalyst_x, catalyst_y = split_dataset(catalyst_dataset)
google_x, google_y = split_dataset(google_dataset)
aws_x, aws_y = split_dataset(aws_dataset)
azure_x, azure_y = split_dataset(azure_dataset)

from IPython.core.display import display, HTML
display(HTML('<iframe width="840" height="472" src="https://www.youtube-nocookie.com/embed/GAmzwIkGFgE?rel=0" frameborder="0" allowfullscreen></iframe>'))

# Initialise regressors
catalyst_linear = LinearRegression()
google_linear = LinearRegression()
aws_linear = LinearRegression()
azure_linear = LinearRegression()

# Train regressors
catalyst_linear.fit(catalyst_x, catalyst_y)
google_linear.fit(google_x, google_y)
aws_linear.fit(aws_x, aws_y)
azure_linear.fit(azure_x, azure_y)

# Predict Catalyst X
google_cata_price = google_linear.predict(catalyst_x)
aws_cata_price = aws_linear.predict(catalyst_x)
azure_cata_price = azure_linear.predict(catalyst_x)

# Predict Google X
aws_goog_price = aws_linear.predict(google_x)
azure_goog_price = azure_linear.predict(google_x)
catalyst_goog_price = catalyst_linear.predict(google_x)

# Predict AWS X
google_aws_price = google_linear.predict(aws_x)
azure_aws_price = azure_linear.predict(aws_x)
catalyst_aws_price = catalyst_linear.predict(aws_x)

# Predict Azure X
google_azure_price = google_linear.predict(azure_x)
aws_azure_price = aws_linear.predict(azure_x)
catalyst_azure_price = catalyst_linear.predict(azure_x)

def graph_pred (names, predictions):
    flavors_num = predictions[0].shape[0]
    for index, name in enumerate(names):
        plt.plot(range(flavors_num), predictions[index], label=names[index])
    plt.legend(loc=2)

graph_pred([
        "Catalyst", "Google", "AWS", "Azure"
    ], [
        catalyst_y, google_cata_price, aws_cata_price, azure_cata_price
    ])

graph_pred([
        "Catalyst", "Google", "AWS", "Azure"
    ], [
        catalyst_goog_price,
        google_y,
        aws_goog_price,
        azure_goog_price,
        
    ])

graph_pred([
        "Catalyst", "Google", "AWS", "Azure"
    ], [
        catalyst_aws_price,
        google_aws_price,
        aws_y,
        azure_aws_price,
    ])

graph_pred([
        "Catalyst", "Google", "AWS", "Azure"
    ], [
        catalyst_azure_price,
        google_azure_price,
        aws_azure_price,
        azure_y
    ])

