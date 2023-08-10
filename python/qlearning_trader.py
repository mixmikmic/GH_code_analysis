# Import libraries
import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
from datetime import datetime as dt

# Define functions for gathering price data

def fill_missing_values(df):
    """
    Fill missing values in data frame, in place.
    """
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    
    
def get_price_data(symbol_list, start_date, end_date):
    """
    Read stock data (adjusted close) for given symbols and time period.
    """ 
    # Initialize dataframe and index values
    df = pd.DataFrame(index=pd.date_range(start_date, end_date))
    df.index.name = "date" # name index
    
    # Add SPY for reference, if absent
    if 'SPY' not in symbol_list:
        symbol_list.insert(0, "SPY")

    # Iterate through symbols and download data
    for symbol in symbol_list:
        try:
            df_temp = pdr.DataReader(symbol, 'yahoo', start_date, end_date)
            df_temp.rename(columns = {'Adj Close': symbol}, inplace=True)
            df_temp = df_temp[symbol]                 
            df = df.join(df_temp)
            print "Downloaded ticker: {}".format(symbol)
            if symbol == "SPY":
                df = df.dropna(subset=["SPY"]) # remove none trading days
        except:
            print "Cant find ticker symbol: {}".format(symbol) # print error if not in yahoo database
            
    print "Price data downloaded for {} stocks.".format(len(symbol_list))
    
    # Fill forward, backward missing values
    fill_missing_values(df)
    
    # Split SPY data from the rest
    price_df = df
    spy_df = price_df.pop('SPY')
    
    # Export price_data and spy_data for records
    price_df.to_pickle('data/price_data.pkl')
    spy_df.to_pickle('data/spy_data.pkl')

    return price_df, spy_df

# Process price data

# Initialize symbol_list
# XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU
symbol_list = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']

# Initialize start and end dates
start_date = "1999-02-01"
end_date = dt.today().strftime("%Y-%m-%d")

# Run function to get price data
price_df, spy_df = get_price_data(symbol_list, start_date, end_date)

# Explore price data
print "price_df head:\n"
print price_df.head()
print "*"*100
print "spy_df head:\n"
print spy_df.head()
print "*"*100
print "price_df summary statistics:\n"
print price_df.describe()

# Define functions for computing technical data

def get_simple_moving_average(price_df, window):
    """
    Returns the percentage difference between the current price and the simple
    moving average for different windows of time to be used as a technical indicator.
    """
    # Compute rolling mean
    ma = price_df.rolling(window=window).mean()
    
    # Compute distance from rolling mean
    sma_df = price_df / ma - 1
    sma_df = sma_df.fillna(value=0)
    sma_df = sma_df.round(2)
    
    return sma_df


def get_bollinger_bands(price_df, window):
    """
    Returns whether price is above or below bollinger_bands.
    1  - above upper band
    0  - inbetween bands
    -1 - below lower band
    """
    # Compute rolling mean and std    
    ma = price_df.rolling(window=window).mean()
    sd = price_df.rolling(window=window).std()
    
    # Get bands
    upper_band = ma + sd * 2
    lower_band = ma - sd * 2
    
    # Set conditional values in dataframe
    above_upper = price_df > upper_band
    below_lower = price_df < lower_band
    above_upper = above_upper.applymap(lambda x: 1.0 if x else 0.0)
    below_lower = below_lower.applymap(lambda x: 1.0 if x else 0.0)
    
    # Return dataframe
    bband_df = above_upper - below_lower
    
    return bband_df


# Function to combine techincal indicators into a single DF with tuple values
def combine_data(price_df, sma_50, sma_200, bband_50, bband_200):
    """
    Returns dataframe of tuples storing technical data for every symbol.
    """
    # Initialize new DF with same index as price_df
    new_df = pd.DataFrame(index=price_df.index) 
    
    # Iterate through column names to form a new column in a DF consisting
    # of tuples containing technical indicators
    for column in price_df:
        new_df[column] = zip(sma_50[column], sma_200[column], bband_50[column], bband_200[column])
    
    # Export techincal_data for records
    new_df.to_pickle('data/technical_data.pkl')
    
    return new_df

# Calculate technical data

# Calculate SMA indicator for 25 day and 100 day window
sma_25 = get_simple_moving_average(price_df, 25)
sma_100 = get_simple_moving_average(price_df, 100)

# Calculate BBAND indicator for 50 day and 200 day window
bband_50 = get_bollinger_bands(price_df, 50)
bband_200 = get_bollinger_bands(price_df, 200)

# Combine technical indicators into a single DF
tech_df = combine_data(price_df, sma_25, sma_100, bband_50, bband_200)

# Explore technical data
print "tech_df tail:\n"
print tech_df.ix[-10:,]

# 1. Create a function to manage data loading for different training and testing periods.

def load_data(window=201, test=False):
    """
    Load training and test sets. Use after data has been preprocessed and
    exported. Window sets the start date. Always use one year of training data
    followed by one month of testing data.
    """ 
    # Load data
    prices = pd.read_pickle('data/price_data.pkl')
    spy = pd.read_pickle('data/spy_data.pkl')
    tech = pd.read_pickle('data/technical_data.pkl')
    
    # Set date variables
    test_win = 175
    train_win = 25
    start = window
    t_start = window - test_win
    end = window - test_win - train_win
    
    # Subset training data
    prices_train = prices.iloc[-start:-t_start,]
    tech_train = tech.iloc[-start:-t_start,]
    spy_train = spy.iloc[-start:-t_start,]

    # Subset testing data
    prices_test = prices.iloc[-t_start:-end,]
    tech_test = tech.iloc[-t_start:-end,]
    spy_test = spy.iloc[-t_start:-end,]

    # Return dataset
    if test == True:
        return tech_test, prices_test, spy_test
    else:
        return tech_train, prices_train, spy_train

# 2. Create a portfolio class to keep track of our models performance.

# Create helper functions
def normalize_data(df):
    """
    Normalize stock prices using the first row of dataframe.
    """
    return df / df.ix[0, :]


def plot_data(df, epoch, title="Stock prices", xlabel="Date", ylabel="Price",
              legend=True, test=0):
    """
    Plot stock prices with a custom title and axis labels. Export plots to file
    based on epoch given a dataframe.
    """
    # Initalize plot
    ax = df.plot(title=title, fontsize=9, figsize=(11, 8))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc='best')

    # Save plot to file (plots folder must exist)
    if test == 0:
        plt.savefig('plots/train_plot_{}.png'.format(epoch))
    elif test == 1:
        plt.savefig('plots/test_plot_{}.png'.format(epoch))
    else:
        plt.savefig('plots/eval_plot_{}.png'.format(epoch))
        plt.show()
    
    # Close plots to save memory
    plt.close('all')
    

# Create portfolio class
class Portfolio(object):
    def __init__(self, initial_value, df_price):
        # Initial value of portfolio and time_step 1
        self.initial_value = initial_value
        
        # Dataframe to track portfolio throughout learning and testing
        self.port_tracker = pd.DataFrame(index=df_price.index)
        self.port_tracker['value'] = np.nan
        self.port_tracker['value'][0] = initial_value
        
        
    def get_value(self, price_data, weights, time_step):
        """
        Returns the value of a constant portfolio on a given day.
        Used to calculate return between different time periods and weights.
        """
        # Get prices and normalize them
        norm_price = normalize_data(price_data)
    
        # Multiply normalized prices by portfolio weights
        alloced = norm_price * weights
        
        # Multiple eighted portfolio by initial value        
        pos_vals = alloced * self.initial_value
    
        # Sum accross position values to get portfolio value
        port_val = pos_vals.sum(axis=1)
        
        # Get time period that we are on
        port_val = port_val[time_step-1]
    
        return port_val
    
        
    def statistics(self, benchmark, samples_per_year=252, riskfree=.02): 
        """
        Calculate statistics on given portfolio of symbols given a benchmark.
        cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio
        """
        # First get daily returns on portfolio    
        daily_ret = (self.port_tracker['value'] /
                     self.port_tracker['value'].shift(1)) - 1
        daily_ret.ix[0] = 0 # set daily returns for row 0 to 0
        daily_ret = daily_ret.ix[1:] #drop first value, it is NaN
        
        # Compute portfolio statistics
        cum_ret = (self.port_tracker['value'][-2] /
                   self.port_tracker['value'][0]) - 1
        avg_daily_ret = daily_ret.mean()
        std_daily_ret = daily_ret.std()
        k = np.sqrt(samples_per_year)
        adj_risk_free = np.power((1 + riskfree), (1 / samples_per_year)) - 1
        sharpe_ratio = k * ((avg_daily_ret - adj_risk_free) / std_daily_ret)
        
        # Compute benchmark daily returns
        daily_ret_spy = (benchmark / benchmark.shift(1)) - 1
        daily_ret_spy.ix[0] = 0 # set daily returns for row 0 to 0
        daily_ret_spy = daily_ret_spy.ix[1:] #drop first value, it is NaN
        
        # Compute benchmark statistics        
        avg_spy_ret = daily_ret_spy.mean()
        std_spy_ret = daily_ret_spy.std()
        sharpe_ratio_spy = k * ((avg_spy_ret - adj_risk_free) / std_spy_ret)
        cum_spy = (benchmark[-2] / benchmark[0]) - 1
    
        return cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy
    
        
    def plot_results(self, benchmark, epoch, test=0):
        """
        Plot portfolio results and compare them with benchmark.
        """
        # Combine benchmark and portfolio data
        df_temp = pd.concat([self.port_tracker, benchmark[:-1]], axis=1)
        df_temp.rename(columns = {0: 'PORT_VAL'}, inplace=True)
        
        # Plot
        if test == 0:
            plot_data(normalize_data(df_temp), epoch,
                      title="Daily Portfolio and SPY value (train: {})".format(epoch),
                      xlabel="Date", ylabel="Value", test=test)
        else:
            plot_data(normalize_data(df_temp), epoch,
                      title="Daily Portfolio and SPY value (test: {})".format(epoch),
                      xlabel="Date", ylabel="Value", test=test)

# 3. Create an evaluation function to test our model on the testing set at the end of each epoch.

def evaluate_qlearner(test_data, model, test_price, test_spy, epoch):
    """
    Evaluate the performance of the learner at each epoch. Returns statistics
    and saves a log of actions and a plot of performance.
    """
    state = init_state(test_data)
    status = 1
    terminal_state = 0
    time_step = 1
    port_test = Portfolio(100000, test_price)
    open('logs/test_action_log_{}.txt'.format(epoch+1), 'w').close()

    while(status == 1):
        # Run Q-function model on initial state and get rewards for actions
        qval = model.predict(state[0], batch_size=1)

        # Get action from qval
        action = (np.argmax(qval))

        # Take action and get new state
        new_state, time_step, terminal_state = take_action(state, test_data,
                                                           action, time_step)
        
        # Get reward
        reward = get_reward(port_test, test_price, state, new_state, time_step)
        
        # Update state
        state = new_state
        
        # Update status
        if terminal_state == 1: #terminal state
            status = 0
            
        # Log date, actions, and reward to text file. (logs folder must exist)
        date = test_price.iloc[time_step - 1].name
        with open('logs/test_action_log_{}.txt'.format(epoch+1), 'a') as file:
            file.write("{} ACTION: {} ({:.3f})\n".format(date, action, reward))
    
    # Extract statistics on test and save plot            
    cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy = port_test.statistics(test_spy)
    port_test.plot_results(test_spy, epoch+1, test=1)
    
    return cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy

# 4. Design our Q-learning functions (state, action, reward, new state).

def init_state(tech_data):
    """
    Get the initial state for our Qlearner. State holds technical data and the
    current portfolio weights.
    """
    # Intialize weights to be even across portfolio
    init_weight_guess = np.ones(tech_data.shape[1],
                                dtype=np.float64) * 1.0 / tech_data.shape[1]
    
    # State will hold our technical data and the current portfolio weights
    state = [np.array([[np.array(tech_data.ix[0,:].values,
                                 dtype=('f4,f4,i4,i4'))]]), init_weight_guess]
    
    return state


def take_action(state, tech_data, action, time_step):
    """
    Executes an action (0-9) by mapping it to a weight. Actions are limited to
    overweighting one sector or holding all sectors equally.
    """
    # Increase time_step
    time_step += 1
    
    # Map actions to portfolio weights
    if action == 0:
        weight = np.array([0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11])
    elif action == 1:
        weight = np.array([0.28,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09])
    elif action == 2:
        weight = np.array([0.09,0.28,0.09,0.09,0.09,0.09,0.09,0.09,0.09])
    elif action == 3:
        weight = np.array([0.09,0.09,0.28,0.09,0.09,0.09,0.09,0.09,0.09])
    elif action == 4:
        weight = np.array([0.09,0.09,0.09,0.28,0.09,0.09,0.09,0.09,0.09])
    elif action == 5:
        weight = np.array([0.09,0.09,0.09,0.09,0.28,0.09,0.09,0.09,0.09])
    elif action == 6:
        weight = np.array([0.09,0.09,0.09,0.09,0.09,0.28,0.09,0.09,0.09])
    elif action == 7:
        weight = np.array([0.09,0.09,0.09,0.09,0.09,0.09,0.28,0.09,0.09])
    elif action == 8:
        weight = np.array([0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.28,0.09])
    else:
        weight = np.array([0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.28])
    
    # Set termenal state to 1 if this is the last time_step
    if time_step + 1 == tech_data.shape[0]:
        terminal_state = 1
    else:
        terminal_state = 0

    # Set the new states technical data and weights and return t
    new_state = [np.array([[np.array(tech_data.ix[time_step-1,:].values,
                                     dtype=('f4,f4,i4,i4'))]]), weight]

    return new_state, time_step, terminal_state

    
def get_reward(port, price_data, initial_state, new_state, time_step):
    """
    Gets reward based on difference in value between initial_state and
    new_state. Updates portfolio tracker with values.
    """
    # Calculate percent change from initial_state to _new_state
    value_new = port.get_value(price_data, new_state[1], time_step)
    value_init = port.get_value(price_data, initial_state[1], time_step-1)    
    reward = (value_new / value_init) - 1
    
    # Update portfolio tracker
    port.port_tracker['value'][time_step-1] = port.port_tracker['value'][time_step - 2] * (reward + 1)

    # Using portfolio tracker values, set reward to be the cummulative return
    # Over a 20 day window. Otherwise since inception. 
    if time_step < 20:
        reward = (port.port_tracker['value'][time_step-1] /
                  port.port_tracker['value'][0])-1
        # Modify reward to weight gains and losses differently
        if reward > 0:
            reward = (reward + 1)
        else:
            reward = (reward - 1)**3
    else:
        reward = (port.port_tracker['value'][time_step-1] /
                  port.port_tracker['value'][time_step-20])-1
        # Modify reward to weight gains and losses differently
        if reward > 0:
            reward = (reward + 1)
        else:
            reward = (reward - 1)**3

    return reward

# 5. Build the Q-function out of a neural net. This will map states to rewards.

# Import libraries
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam

# Set dimension variables
st_dim = init_state(pd.read_pickle('data/technical_data.pkl'))
st_dim = st_dim[0]
steps = st_dim.shape[1]
num_features = st_dim.shape[2]

# Initialize sequential model
model = Sequential()

# Build first layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build second layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build third layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=False,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build fourth layer
model.add(Dense(10, init='lecun_uniform'))
model.add(Activation('linear'))

# Build optimizer and compile
adam = Adam()
model.compile(loss='mse', optimizer=adam)

# 6. Train and test in a Q-learning loop.
# Model will be fed batches of an experience replay to be trained on.

# Import Libraries
import random
import timeit

# Initialize start time
start_time = timeit.default_timer()

# Initialize learning variables
epochs = 25 # number of passes on the training data
gamma = 0.95 # weight of future rewards
epsilon = 1.25 # measure of randomness for exploration (ten random epochs then constant decrease)
batch_size = 10 # size of training batches for our neural net

# Initialize experience variables
memory = [] # stores training data for neural net in efficient chunks
mem_limit = 20
mem_counter = 0

# Initialize training window start
window = 201

# Initialize testing data
test_data, test_price, test_spy = load_data(window=window, test=True)

# Initialize training data
train_data, price_data, benchmark = load_data(window=window)

# Open file to log portfolio results after epoch (logs folder must exist)
open('logs/master_test_log.txt', 'w').close()
open('logs/master_train_log.txt', 'w').close()

for i in range(epochs):
    # Initialize variables
    state = init_state(train_data)
    port = Portfolio(100000, price_data)
    status = 1
    terminal_state = 0
    time_step = 1
    
    # Open file to log training actions (logs folder must exist)
    open('logs/train_action_log_{}.txt'.format(i+1), 'w').close()
    
    # Train while status = 1
    while status == 1:
        # Run Q-function model on initial state and get rewards for actions
        qval = model.predict(state[0], batch_size=1)
        
        # Using epsilon, add random exporation to the trainer
        # Otherwise find action with Q-function model.
        if random.random() < epsilon:
            action = np.random.randint(1,10)
        else:
            action = (np.argmax(qval))
            
        # Take action and get new state
        new_state, time_step, terminal_state = take_action(state, train_data, action, time_step)
        
        # Get reward
        reward = get_reward(port, price_data, state, new_state, time_step)
        
        # Log date, actions, and reward to text file. (logs folder must exist)
        date = price_data.iloc[time_step - 1].name
        with open('logs/train_action_log_{}.txt'.format(i+1), 'a') as file:
                file.write("{} ACTION: {} ({:.3f})\n".format(date, action, reward))
        
        # If memory not full, add to memory
        if len(memory) < mem_limit:
            memory.append((state, action, reward, new_state))
            state = new_state # set state for next loop
            
        # Otherwise overwrite memory
        else:
            if mem_counter < (mem_limit - 1):
                mem_counter +=1
            else:
                mem_counter = 0
            memory[mem_counter] = (state, action, reward, new_state)
            state = new_state # set state for next loop
            
            # Begin training model on memory experience
            mem_batch = random.sample(memory, batch_size)
            X_train = []
            y_train = []

            for mem in mem_batch:
                # Get Q-values with Q-function
                prior_state, action, reward, next_state = mem
                prior_qval = model.predict(prior_state[0], batch_size=1)
                next_qval = model.predict(next_state[0], batch_size=1)
                
                # Get reward for next state
                next_reward = np.max(next_qval)
                
                # Calculate reward based on gamma, if terminal just use reward
                if terminal_state == 0:
                    reward_update = (reward + (gamma * next_reward))
                else: #terminal state
                    reward_update = reward
                    
                # Get training data together
                y = np.zeros((1,10))
                y[:] = prior_qval[:]
                y[0][action] = reward_update

                X_train.append(prior_state[0]) # Input technical data
                y_train.append(y.reshape(10,)) # Output reward for actions
                
            # Reshape training batch to fit model input
            X_train = np.squeeze(np.array(X_train), axis=(1))
            y_train = np.array(y_train)
            
            # Train model on memory batch
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)

        # update status
        if terminal_state == 1:
            status = 0
            
    # Update epsilon for next epoch
    epsilon -= (1.25/(epochs-1))
            
    # Evaluate epoch results on training data
    cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy = port.statistics(benchmark)
    port.plot_results(benchmark, i+1)
    with open('logs/master_train_log.txt'.format(i+1), 'a') as file:
                file.write("Train {}/{}: Relative Return/Sharpe: {:.4%}/{:.2f}\n".format(i+1, epochs,
                                                                                         cum_ret - cum_spy,
                                                                                         sharpe_ratio - sharpe_ratio_spy))
                
    # Evaluate epoch results on testing data
    elapsed = np.round((timeit.default_timer() - start_time)/60, decimals=2)
    cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy = evaluate_qlearner(test_data, model, test_price, test_spy, i)
    print "Test {}/{} completed in {:.2f} minutes - Relative Return/Sharpe: {:.4%} / {:.2f}".format(i+1, epochs, elapsed,
                                                                                                    cum_ret - cum_spy,
                                                                                                    sharpe_ratio - \
                                                                                                    sharpe_ratio_spy)
    with open('logs/master_test_log.txt'.format(i+1), 'a') as file:
                file.write("Test {}/{}: Relative Return/Sharpe: {:.4%}/{:.2f}\n".format(i+1, epochs, cum_ret - cum_spy,
                                                                                        sharpe_ratio - sharpe_ratio_spy))
    #print "*"*100
    
elapsed = np.round((timeit.default_timer() - start_time)/60/60, decimals=2)
print "Epochs completed in {:.2f} hours".format(elapsed)

get_ipython().magic('matplotlib inline')

df = test_price
plot_data(normalize_data(df), "train", title="Normalized Performance", test=3)

# Re-initialize and run Q-learning loop on random time window (Same code as above)

# Neural Net Q-Function
# Initialize sequential model
model = Sequential()

# Build first layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build second layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build third layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=False,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build fourth layer
model.add(Dense(10, init='lecun_uniform'))
model.add(Activation('linear'))

# Build optimizer and compile
adam = Adam()
model.compile(loss='mse', optimizer=adam)

# Q-Learning loop
# Initialize start time
start_time = timeit.default_timer()

# Initialize learning variables
epochs = 25 # number of passes on the training data
gamma = 0.95 # weight of future rewards
epsilon = 1.25 # measure of randomness for exploration (ten random epochs then constant decrease)
batch_size = 10 # size of training batches for our neural net

# Initialize experience variables
memory = [] # stores training data for neural net in efficient chunks
mem_limit = 20
mem_counter = 0

# Initialize random training window start
window = np.random.randint(201,4000)
print "Window: {}".format(window)

# Initialize testing data
test_data, test_price, test_spy = load_data(window=window, test=True)

# Initialize training data
train_data, price_data, benchmark = load_data(window=window)

# Open file to log portfolio results after epoch (logs folder must exist)
open('logs/master_test_log.txt', 'w').close()
open('logs/master_train_log.txt', 'w').close()

for i in range(epochs):
    # Initialize variables
    state = init_state(train_data)
    port = Portfolio(100000, price_data)
    status = 1
    terminal_state = 0
    time_step = 1
    
    # Open file to log training actions (logs folder must exist)
    open('logs/train_action_log_{}.txt'.format(i+1), 'w').close()
    
    # Train while status = 1
    while status == 1:
        # Run Q-function model on initial state and get rewards for actions
        qval = model.predict(state[0], batch_size=1)
        
        # Using epsilon, add random exporation to the trainer
        # Otherwise find action with Q-function model.
        if random.random() < epsilon:
            action = np.random.randint(1,10)
        else:
            action = (np.argmax(qval))
            
        # Take action and get new state
        new_state, time_step, terminal_state = take_action(state, train_data, action, time_step)
        
        # Get reward
        reward = get_reward(port, price_data, state, new_state, time_step)
        
        # Log date, actions, and reward to text file. (logs folder must exist)
        date = price_data.iloc[time_step - 1].name
        with open('logs/train_action_log_{}.txt'.format(i+1), 'a') as file:
                file.write("{} ACTION: {} ({:.3f})\n".format(date, action, reward))
        
        # If memory not full, add to memory
        if len(memory) < mem_limit:
            memory.append((state, action, reward, new_state))
            state = new_state # set state for next loop
            
        # Otherwise overwrite memory
        else:
            if mem_counter < (mem_limit - 1):
                mem_counter +=1
            else:
                mem_counter = 0
            memory[mem_counter] = (state, action, reward, new_state)
            state = new_state # set state for next loop
            
            # Begin training model on memory experience
            mem_batch = random.sample(memory, batch_size)
            X_train = []
            y_train = []

            for mem in mem_batch:
                # Get Q-values with Q-function
                prior_state, action, reward, next_state = mem
                prior_qval = model.predict(prior_state[0], batch_size=1)
                next_qval = model.predict(next_state[0], batch_size=1)
                
                # Get reward for next state
                next_reward = np.max(next_qval)
                
                # Calculate reward based on gamma, if terminal just use reward
                if terminal_state == 0:
                    reward_update = (reward + (gamma * next_reward))
                else: #terminal state
                    reward_update = reward
                    
                # Get training data together
                y = np.zeros((1,10))
                y[:] = prior_qval[:]
                y[0][action] = reward_update

                X_train.append(prior_state[0]) # Input technical data
                y_train.append(y.reshape(10,)) # Output reward for actions
                
            # Reshape training batch to fit model input
            X_train = np.squeeze(np.array(X_train), axis=(1))
            y_train = np.array(y_train)
            
            # Train model on memory batch
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)

        # update status
        if terminal_state == 1:
            status = 0
            
    # Update epsilon for next epoch
    epsilon -= (1.25/(epochs-1))
            
    # Evaluate epoch results on training data
    cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy = port.statistics(benchmark)
    port.plot_results(benchmark, i+1)
    with open('logs/master_train_log.txt'.format(i+1), 'a') as file:
                file.write("Train {}/{}: Relative Return/Sharpe: {:.4%}/{:.2f}\n".format(i+1, epochs,
                                                                                         cum_ret - cum_spy,
                                                                                         sharpe_ratio - sharpe_ratio_spy))
                
    # Evaluate epoch results on testing data
    elapsed = np.round((timeit.default_timer() - start_time)/60, decimals=2)
    cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy = evaluate_qlearner(test_data, model, test_price, test_spy, i)
    print "Test {}/{} completed in {:.2f} minutes - Relative Return/Sharpe: {:.4%} / {:.2f}".format(i+1, epochs, elapsed,
                                                                                                    cum_ret - cum_spy,
                                                                                                    sharpe_ratio - \
                                                                                                    sharpe_ratio_spy)
    with open('logs/master_test_log.txt'.format(i+1), 'a') as file:
                file.write("Test {}/{}: Relative Return/Sharpe: {:.4%}/{:.2f}\n".format(i+1, epochs, cum_ret - cum_spy,
                                                                                        sharpe_ratio - sharpe_ratio_spy))
    #print "*"*100
    
elapsed = np.round((timeit.default_timer() - start_time)/60/60, decimals=2)
print "Epochs completed in {:.2f} hours".format(elapsed)

df = test_price
plot_data(normalize_data(df), "train", title="Normalized Performance", test=3)

# Re-initialize and run Q-learning loop on random time window (Same code as above)

# Neural Net Q-Function
# Initialize sequential model
model = Sequential()

# Build first layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build second layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build third layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=False,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build fourth layer
model.add(Dense(10, init='lecun_uniform'))
model.add(Activation('linear'))

# Build optimizer and compile
adam = Adam()
model.compile(loss='mse', optimizer=adam)

# Q-Learning loop
# Initialize start time
start_time = timeit.default_timer()

# Initialize learning variables
epochs = 25 # number of passes on the training data
gamma = 0.95 # weight of future rewards
epsilon = 1.25 # measure of randomness for exploration (ten random epochs then constant decrease)
batch_size = 10 # size of training batches for our neural net

# Initialize experience variables
memory = [] # stores training data for neural net in efficient chunks
mem_limit = 20
mem_counter = 0

# Initialize random training window start
window = np.random.randint(201,4000)
print "Window: {}".format(window)

# Initialize testing data
test_data, test_price, test_spy = load_data(window=window, test=True)

# Initialize training data
train_data, price_data, benchmark = load_data(window=window)

# Open file to log portfolio results after epoch (logs folder must exist)
open('logs/master_test_log.txt', 'w').close()
open('logs/master_train_log.txt', 'w').close()

for i in range(epochs):
    # Initialize variables
    state = init_state(train_data)
    port = Portfolio(100000, price_data)
    status = 1
    terminal_state = 0
    time_step = 1
    
    # Open file to log training actions (logs folder must exist)
    open('logs/train_action_log_{}.txt'.format(i+1), 'w').close()
    
    # Train while status = 1
    while status == 1:
        # Run Q-function model on initial state and get rewards for actions
        qval = model.predict(state[0], batch_size=1)
        
        # Using epsilon, add random exporation to the trainer
        # Otherwise find action with Q-function model.
        if random.random() < epsilon:
            action = np.random.randint(1,10)
        else:
            action = (np.argmax(qval))
            
        # Take action and get new state
        new_state, time_step, terminal_state = take_action(state, train_data, action, time_step)
        
        # Get reward
        reward = get_reward(port, price_data, state, new_state, time_step)
        
        # Log date, actions, and reward to text file. (logs folder must exist)
        date = price_data.iloc[time_step - 1].name
        with open('logs/train_action_log_{}.txt'.format(i+1), 'a') as file:
                file.write("{} ACTION: {} ({:.3f})\n".format(date, action, reward))
        
        # If memory not full, add to memory
        if len(memory) < mem_limit:
            memory.append((state, action, reward, new_state))
            state = new_state # set state for next loop
            
        # Otherwise overwrite memory
        else:
            if mem_counter < (mem_limit - 1):
                mem_counter +=1
            else:
                mem_counter = 0
            memory[mem_counter] = (state, action, reward, new_state)
            state = new_state # set state for next loop
            
            # Begin training model on memory experience
            mem_batch = random.sample(memory, batch_size)
            X_train = []
            y_train = []

            for mem in mem_batch:
                # Get Q-values with Q-function
                prior_state, action, reward, next_state = mem
                prior_qval = model.predict(prior_state[0], batch_size=1)
                next_qval = model.predict(next_state[0], batch_size=1)
                
                # Get reward for next state
                next_reward = np.max(next_qval)
                
                # Calculate reward based on gamma, if terminal just use reward
                if terminal_state == 0:
                    reward_update = (reward + (gamma * next_reward))
                else: #terminal state
                    reward_update = reward
                    
                # Get training data together
                y = np.zeros((1,10))
                y[:] = prior_qval[:]
                y[0][action] = reward_update

                X_train.append(prior_state[0]) # Input technical data
                y_train.append(y.reshape(10,)) # Output reward for actions
                
            # Reshape training batch to fit model input
            X_train = np.squeeze(np.array(X_train), axis=(1))
            y_train = np.array(y_train)
            
            # Train model on memory batch
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)

        # update status
        if terminal_state == 1:
            status = 0
            
    # Update epsilon for next epoch
    epsilon -= (1.25/(epochs-1))
            
    # Evaluate epoch results on training data
    cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy = port.statistics(benchmark)
    port.plot_results(benchmark, i+1)
    with open('logs/master_train_log.txt'.format(i+1), 'a') as file:
                file.write("Train {}/{}: Relative Return/Sharpe: {:.4%}/{:.2f}\n".format(i+1, epochs,
                                                                                         cum_ret - cum_spy,
                                                                                         sharpe_ratio - sharpe_ratio_spy))
                
    # Evaluate epoch results on testing data
    elapsed = np.round((timeit.default_timer() - start_time)/60, decimals=2)
    cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy = evaluate_qlearner(test_data, model, test_price, test_spy, i)
    print "Test {}/{} completed in {:.2f} minutes - Relative Return/Sharpe: {:.4%} / {:.2f}".format(i+1, epochs, elapsed,
                                                                                                    cum_ret - cum_spy,
                                                                                                    sharpe_ratio - \
                                                                                                    sharpe_ratio_spy)
    with open('logs/master_test_log.txt'.format(i+1), 'a') as file:
                file.write("Test {}/{}: Relative Return/Sharpe: {:.4%}/{:.2f}\n".format(i+1, epochs, cum_ret - cum_spy,
                                                                                        sharpe_ratio - sharpe_ratio_spy))
    #print "*"*100
    
elapsed = np.round((timeit.default_timer() - start_time)/60/60, decimals=2)
print "Epochs completed in {:.2f} hours".format(elapsed)

df = test_price
plot_data(normalize_data(df), "train", title="Normalized Performance", test=3)

# Re-initialize and run Q-learning loop on random time window (Same code as above)

# Neural Net Q-Function
# Initialize sequential model
model = Sequential()

# Build first layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build second layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build third layer
model.add(LSTM(64,
               input_shape=(steps, num_features),
               return_sequences=False,
               stateful=False))
model.add(Dropout(0.50)) # drop inputs to avoid overfitting

# Build fourth layer
model.add(Dense(10, init='lecun_uniform'))
model.add(Activation('linear'))

# Build optimizer and compile
adam = Adam()
model.compile(loss='mse', optimizer=adam)

# Q-Learning loop
# Initialize start time
start_time = timeit.default_timer()

# Initialize learning variables
epochs = 25 # number of passes on the training data
gamma = 0.95 # weight of future rewards
epsilon = 1.25 # measure of randomness for exploration (ten random epochs then constant decrease)
batch_size = 10 # size of training batches for our neural net

# Initialize experience variables
memory = [] # stores training data for neural net in efficient chunks
mem_limit = 20
mem_counter = 0

# Initialize random training window start
window = np.random.randint(201,4000)
print "Window: {}".format(window)

# Initialize testing data
test_data, test_price, test_spy = load_data(window=window, test=True)

# Initialize training data
train_data, price_data, benchmark = load_data(window=window)

# Open file to log portfolio results after epoch (logs folder must exist)
open('logs/master_test_log.txt', 'w').close()
open('logs/master_train_log.txt', 'w').close()

for i in range(epochs):
    # Initialize variables
    state = init_state(train_data)
    port = Portfolio(100000, price_data)
    status = 1
    terminal_state = 0
    time_step = 1
    
    # Open file to log training actions (logs folder must exist)
    open('logs/train_action_log_{}.txt'.format(i+1), 'w').close()
    
    # Train while status = 1
    while status == 1:
        # Run Q-function model on initial state and get rewards for actions
        qval = model.predict(state[0], batch_size=1)
        
        # Using epsilon, add random exporation to the trainer
        # Otherwise find action with Q-function model.
        if random.random() < epsilon:
            action = np.random.randint(1,10)
        else:
            action = (np.argmax(qval))
            
        # Take action and get new state
        new_state, time_step, terminal_state = take_action(state, train_data, action, time_step)
        
        # Get reward
        reward = get_reward(port, price_data, state, new_state, time_step)
        
        # Log date, actions, and reward to text file. (logs folder must exist)
        date = price_data.iloc[time_step - 1].name
        with open('logs/train_action_log_{}.txt'.format(i+1), 'a') as file:
                file.write("{} ACTION: {} ({:.3f})\n".format(date, action, reward))
        
        # If memory not full, add to memory
        if len(memory) < mem_limit:
            memory.append((state, action, reward, new_state))
            state = new_state # set state for next loop
            
        # Otherwise overwrite memory
        else:
            if mem_counter < (mem_limit - 1):
                mem_counter +=1
            else:
                mem_counter = 0
            memory[mem_counter] = (state, action, reward, new_state)
            state = new_state # set state for next loop
            
            # Begin training model on memory experience
            mem_batch = random.sample(memory, batch_size)
            X_train = []
            y_train = []

            for mem in mem_batch:
                # Get Q-values with Q-function
                prior_state, action, reward, next_state = mem
                prior_qval = model.predict(prior_state[0], batch_size=1)
                next_qval = model.predict(next_state[0], batch_size=1)
                
                # Get reward for next state
                next_reward = np.max(next_qval)
                
                # Calculate reward based on gamma, if terminal just use reward
                if terminal_state == 0:
                    reward_update = (reward + (gamma * next_reward))
                else: #terminal state
                    reward_update = reward
                    
                # Get training data together
                y = np.zeros((1,10))
                y[:] = prior_qval[:]
                y[0][action] = reward_update

                X_train.append(prior_state[0]) # Input technical data
                y_train.append(y.reshape(10,)) # Output reward for actions
                
            # Reshape training batch to fit model input
            X_train = np.squeeze(np.array(X_train), axis=(1))
            y_train = np.array(y_train)
            
            # Train model on memory batch
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)

        # update status
        if terminal_state == 1:
            status = 0
            
    # Update epsilon for next epoch
    epsilon -= (1.25/(epochs-1))
            
    # Evaluate epoch results on training data
    cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy = port.statistics(benchmark)
    port.plot_results(benchmark, i+1)
    with open('logs/master_train_log.txt'.format(i+1), 'a') as file:
                file.write("Train {}/{}: Relative Return/Sharpe: {:.4%}/{:.2f}\n".format(i+1, epochs,
                                                                                         cum_ret - cum_spy,
                                                                                         sharpe_ratio - sharpe_ratio_spy))
                
    # Evaluate epoch results on testing data
    elapsed = np.round((timeit.default_timer() - start_time)/60, decimals=2)
    cum_ret, sharpe_ratio, cum_spy, sharpe_ratio_spy = evaluate_qlearner(test_data, model, test_price, test_spy, i)
    print "Test {}/{} completed in {:.2f} minutes - Relative Return/Sharpe: {:.4%} / {:.2f}".format(i+1, epochs, elapsed,
                                                                                                    cum_ret - cum_spy,
                                                                                                    sharpe_ratio - \
                                                                                                    sharpe_ratio_spy)
    with open('logs/master_test_log.txt'.format(i+1), 'a') as file:
                file.write("Test {}/{}: Relative Return/Sharpe: {:.4%}/{:.2f}\n".format(i+1, epochs, cum_ret - cum_spy,
                                                                                        sharpe_ratio - sharpe_ratio_spy))
    #print "*"*100
    
elapsed = np.round((timeit.default_timer() - start_time)/60/60, decimals=2)
print "Epochs completed in {:.2f} hours".format(elapsed)

df = test_price
plot_data(normalize_data(df), "train", title="Normalized Performance Testing", test=3)

df = price_data
plot_data(normalize_data(df), "train", title="Normalized Performance Training", test=3)

