import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def plot_high_prices():
    df = pd.read_csv("data/MSFT.csv")
    # print df['High']
    df ['High'].plot()
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title('High prices for MSFT')
    plt.show()

if __name__ == "__main__":
    plot_high_prices()

