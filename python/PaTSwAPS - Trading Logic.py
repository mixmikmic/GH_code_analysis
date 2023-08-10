import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter

def initialize(context):
    context.stock_pairs = [(symbol('ABGB'), symbol('FSLR'),
                           (symbol('CSUN'), symbol('ASTI')))]
    context.num_pairs = len(context.stock_pairs)
    
    context.security_list = []
    for first_sec, second_sec in context.stock_pairs:
        context.security_list.append(first_sec)
        context.security_list.append(second_sec)
    
    context.spread = np.ndarray((context.num_pairs, 0))
    context.inLong = [False] * context.num_pairs
    context.inShort = [False] * context.num_pairs
    
    # Strategy specific variables
    context.lookback = 20 # used for regression
    context.z_window = 20 # used for zscore calculation, must be <= lookback
    
    # Construct a Kalman filter
    context.kf = KalmanFilter(transition_matrices = [1],
                              observation_matrices = [1],
                              initial_state_mean = 0,
                              initial_state_covariance = 1,
                              observation_covariance=1,
                              transition_covariance=.01)
        
    # Only do work 30 minutes before close
    schedule_function(func=rebalance, date_rule=date_rules.every_day(), time_rule=time_rules.market_close(minutes=30))

def rebalance(context, data):
    if get_open_orders():
        return
    
    prices = data.history(context.security_list, 'price', 35, '1d').iloc[-context.lookback::]
    
    new_spreads = np.ndarray((context.num_pairs, 1))
    
    for i in range(context.num_pairs):
        (stock_y, stock_x) = context.stock_pairs[i]
        Y = prices[stock_y]
        X = prices[stock_x]
        
        try:
            hedge = hedge_ratio(Y, X)      
        except ValueError as err:
            log.debug(err)
            return
        
        new_spreads[i, :] = Y[-1] - hedge * X[-1]
        
        if context.spread.shape[1] > context.z_window:
            # Consider only the z-score lookback period
            spreads = context.spread[i, -context.z_window:]
            state_means, _ = context.kf.filter(context.spread[i])
            zscore = (spreads[-1] - state_means[-1]) / spreads.std()
            
            if context.inShort[i] and zscore < 0.0 and all(data.can_trade([stock_y,stock_x])):
                order_target(stock_y, 0)
                order_target(stock_x, 0)
                context.inShort[i] = False
                context.inLong[i] = False
                record(X_pct=0, Y_pct=0)
            
            if context.inLong[i] and zscore > 0.0 and all(data.can_trade([stock_y,stock_x])):
                order_target(stock_y, 0)
                order_target(stock_x, 0)
                context.inShort[i] = False
                context.inLong[i] = False
                record(X_pct=0, Y_pct=0)
            
            if zscore < -1.0 and (not context.inLong[i]) and all(data.can_trade([stock_y,stock_x])):
                y_target_shares = 1
                x_target_shares = -hedge
                context.inLong[i] = True
                context.inShort[i] = False
                
                (y_target_pct, x_target_pct) = computeHoldingsPct(y_target_shares, x_target_shares, Y[-1], X[-1] )
                order_target_percent(stock_y, y_target_pct * (1.0/context.num_pairs) / float(context.num_pairs))
                order_target_percent(stock_x, x_target_pct * (1.0/context.num_pairs) / float(context.num_pairs))
                record(Y_pct=y_target_pct, X_pct=x_target_pct)
            
            if zscore > 1.0 and (not context.inShort[i]) and all(data.can_trade([stock_y,stock_x])):
                y_target_shares = -1
                x_target_shares = hedge
                context.inShort[i] = True
                context.inLong[i] = False
                
                (y_target_pct, x_target_pct) = computeHoldingsPct(y_target_shares, x_target_shares, Y[-1], X[-1])
                order_target_percent(stock_y, y_target_pct * (1.0/context.num_pairs) / float(context.num_pairs))
                order_target_percent(stock_x, x_target_pct * (1.0/context.num_pairs) / float(context.num_pairs))
                record(Y_pct=y_target_pct, X_pct=x_target_pct)
    
    context.spread = np.hstack([context.spread, new_spreads])

def hedge_ratio(Y, X):
    model = sm.OLS(Y, X).fit()
    return model.params.values
    
def computeHoldingsPct(yShares, xShares, yPrice, xPrice):
    yDol = yShares * yPrice
    xDol = xShares * xPrice
    notionalDol =  abs(yDol) + abs(xDol)
    y_target_pct = yDol / notionalDol
    x_target_pct = xDol / notionalDol
    return (y_target_pct, x_target_pct)

