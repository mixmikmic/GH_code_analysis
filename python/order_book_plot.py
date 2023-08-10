import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------
#%%  set the parameters 
#Stock name
#-------------------------------------------------------------------------
ticker ="AMZN"


#-------------------------------------------------------------------------
# Levels
#-------------------------------------------------------------------------
lvl= 10

#-------------------------------------------------------------------------
# File names
#-------------------------------------------------------------------------
path="/media/jianwang/b2ba2597-9566-445c-87e6-e801c3aee85d/jian/research/data/order_book/"
path_save="/media/jianwang/New Volume/research/order_book/figure/"
name_book    = 'AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
name_mess    = 'AMZN_2012-06-21_34200000_57600000_message_10.csv'

#--------------------------------------------------------------------------
# Date of files
#--------------------------------------------------------------------------
demo_date    = [2012,6,21]    #year, month, day

#---------------------------------------------------------------------------
# Load Messsage File
#---------------------------------------------------------------------------
#  Load data
t=time.time()
mess = np.array(pd.read_csv(path+name_mess))
print("The time for reading the CSV file",time.time()-t)
#
#
#% Message file information:
#% ----------------------------------------------------------
#%   
#%   - Dimension:    (NumberEvents x 6)
#%
#%   - Structure:    Each row:
#%                   Time stamp (sec after midnight with decimal
#%                   precision of at least milliseconds and 
#%                   up to nanoseconds depending on the period), 
#%                   Event type, Order ID, Size (# of shares), 
#%                   Price, Direction
#%
#%                   Event types: 
#%                       - '1'   Submission new limit order
#%                       - '2'   Cancellation (partial)
#%                       - '3'   Deletion (total order)
#%                       - '4'   Execution of a visible limit order
#%                       - '5'   Execution of a hidden limit order 
#%                               liquidity
#%                       - '7'   Trading Halt (Detailed 
#%                               information below)
#%
#%                   Direction:
#%                       - '-1'  Sell limit order
#%                       - '-2'  Buy limit order
#%                       - NOTE: Execution of a sell (buy) 
#%                               limit order corresponds to 
#%                               a buyer-(seller-) initiated 
#%                               trade, i.e. a BUY (SELL) trade. 
#%
#% ----------------------------------------------------------
#% Data Preparation - Message File
#
#% Trading hours (start & end)

#%% deal with the message data 
#Remove observations outside the official trading hours
# ----------------------------------------------------------

#% Trading hours (start & end)
start_trad   = 9.5*60*60       # 9:30:00 in sec 
                               # after midnight
end_trad     = 16*60*60        # 16:00:00 in sec 
                               # after midnight
# Get index of observations 
time_idx=(mess[:,0]>= start_trad) & (mess[:,0]<= end_trad)
mess = mess[time_idx,:]


##-----------------------------------------------------------
#% Note: As the rows of the message and orderbook file
#%       correspond to each other, the time index of 
#%       the message file can also be used to 'cut' 
#%       the orderbook file.
#     
#
#% Check for trading halts
#% ----------------------------------------------------------
trade_halt_idx = np.where(mess[:,1] == 7)

if (np.size(trade_halt_idx)>0):
    print(['Data contains trading halt! Trading halt, '+
    'quoting resume, and resume of trading indices in tradeHaltIdx'])    
else:
    print('No trading halts detected.')
#
#
#%		When trading halts, a message of type '7' is written into the 
#%		'message' file. The corresponding price and trade direction 
#%		are set to '-1' and all other properties are set to '0'. 
#%		Should the resume of quoting be indicated by an additional 
#%		message in NASDAQ's Historical TotalView-ITCH files, another 
#%		message of type '7' with price '0' is added to the 'message' 
#%		file. Again, the trade direction is set to '-1' and all other 
#%		fields are set to '0'. 
#%		When trading resumes a message of type '7' and 
#%		price '1' (Trade direction '-1' and all other 
#%		entries '0') is written to the 'message' file. For messages 
#%		of type '7', the corresponding order book rows contain a 
#%		duplication of the preceding order book state. The reason 
#%		for the trading halt is not included in the output.
#% 						
#% 			Example: Stylized trading halt messages in 'message' file.				
#% 		
#% 			Halt: 				36023	| 7 | 0 | 0 | -1 | -1
#% 											...
#% 			Quoting: 			36323 	| 7 | 0 | 0 | 0  | -1
#% 											...
#% 			Resume Trading:		36723   | 7 | 0 | 0 | 1  | -1
#% 											...
#% 			The vertical bars indicate the different columns in the  
#% 			message file.
#    
#% Set Bounds for Intraday Intervals
#
#% Define interval length

freq = 6.5*3600/(5*60)+1  # Interval length in sec, according to the python do not include the endpoint
                          # so add 1 in the last 

time_interval=60*6.5/(freq-1)

# Set interval bounds
bounds = np.linspace(start_trad,end_trad,freq,endpoint=True)

# Number of intervals
bl = np.size(bounds,0)

# Indices for intervals
bound_idx = np.zeros([bl,1])



k1 = 0 
for k2 in range(0,np.size(mess,0)):
    if mess[k2,0] >= bounds[k1]:
        bound_idx[k1,0] = k2
        k1 = k1+1    
bound_idx[bl-1]=mess[len(mess)-1,0]
  
    
#    
#% Plot - Number of Executions and Trade Volume by Interval
#
#% Note: Difference between trades and executions
#%
#%       The LOBSTER output records limit order executions 
#%       and not what one might intuitively consider trades. 
#%
#%       Imagine a volume of 1000 is posted at the best ask  
#%       price. Further, an incoming market buy order of 
#%       volume 1000 is executed against the quote.
#%
#%       The LOBSTER output of this trade depends on the 
#%       composition of the volume at the best ask price. 
#%       Take the following two scenarios with the best ask 
#%       volume consisting of ...
#%       (a) 1 sell limit order with volume 1000
#%       (b) 5 sell limit orders with volume 200 each 
#%           (ordered according to time of submission)
#%
#%       The LOBSTER output for case ...
#%       (a) shows one execution of volume 1000. If the 
#%           incoming market order is matched with one 
#%           standing limit order, execution and trade
#%           coincide.
#%       (b) shows 5 executions of volume 200 each with the  
#%           same time stamp. The incoming order is matched 
#%           with 5 standing limit orders and triggers 5  
#%           executions.
#%
#%       Bottom line: 
#%       LOBSTER records the exact limit orders against 
#%       which incoming market orders are executed. What 
#%       might be called 'economic' trade size has to be 
#%       inferred from the executions.
    
#% Collection matrix
trades_info = np.zeros([bl-1,4])
#    % Note: Number visible executions, volume visible 
#    %       trades, number hidden executions, 
#    %       volume hidden trades

   
for k1 in range(0,bl-1):
 

    temp	= mess[bound_idx[k1]+1:bound_idx[k1+1],[1,3]]
	   
    # Visible
    temp_vis = temp[temp[:,0]==4,1]
    
    #% Hidden 
    temp_hid = temp[temp[:,0]==5,1];
    
    # Collect information
    trades_info[k1,:] = [np.size(temp_vis,0), np.sum(temp_vis),np.size(temp_hid,0), np.sum(temp_hid)]
                    
    del temp, temp_vis, temp_hid

#%% plot the data 
#Plot number of executions
#------------------------------------------------------------------------------

get_ipython().magic('matplotlib inline')
fig, ax = plt.subplots()
ind=np.arange(np.size(trades_info,0))
width=1
color=["red","blue"]
   #% Visible ...
ax.bar(ind,trades_info[:,0],width=width, color=color[0],label="Visible",alpha=0.7)
#        title({[ticker ' // ' ...
#            datestr(datenum(demoDate),'yyyy-mmm-dd')] ...
#            ['Number of Executions per ' ...
#            num2str(freq./60) ' min Interval ']});
ax.set_xlabel('Interval')
ax.set_ylabel('Number of Executions')
ax.set_title(ticker+"@"+str(demo_date[0])+"-"+str(demo_date[1])+
"-"+str(demo_date[2])+"\nNumber of Executions per "+str(time_interval)+" minutes interval")
ax.bar(ind,-trades_info[:,2],width=width,color=color[1],label="Hidden");
ax.legend(loc="upper center")
plt.savefig(path_save+"num_exec.png")

#------------------------------------------------------------------------------
#plot the volume of traders
#------------------------------------------------------------------------------
fig, ax = plt.subplots()
ind=np.arange(np.size(trades_info,0))
width=1
color=["red","blue"]
   #% Visible ...
ax.bar(ind,trades_info[:,1]/100,width=width, color=color[0],label="visible",alpha=0.7)

ax.set_xlabel('Interval')
ax.set_ylabel('Number of Trades Trades (X100 shares)')
ax.set_title(ticker+"@"+str(demo_date[0])+"-"+str(demo_date[1])+
"-"+str(demo_date[2])+"\nVolume of trades per "+str(time_interval)+" minutes interval")
ax.bar(ind,-trades_info[:,3]/100,width=width,color=color[1],label="Hidden");
ax.legend(loc="upper center")
plt.savefig("/home/jianwang/data/num_trade.png")
plt.show()

t=time.time()
book = np.array(pd.read_csv(path+name_book,dtype ="float64"))
print("The time for reading the CSV file",time.time()-t)
book = book[time_idx,:]
book[:,::2]=book[:,::2]/10000

#%% plot the snapshot of the limit order book 
#------------------------------------------------------------------------------
#select a random event to show
event_idx= np.random.randint(0, len(book))# note that the randint will not generate the last value

ask_price_pos=list(range(0,lvl*4,4))

# Note: Pick a randmom row/ event from the order book.
# position of variables in the book 

ask_price_pos = list(range(0,lvl*4,4))

ask_vol_pos= [i+1 for i in ask_price_pos]

bid_price_pos=[i+2 for i in ask_price_pos]

bid_vol_pos=[i+1 for i in bid_price_pos]

vol= list(range(1,lvl*4,2))

max_price = book[event_idx, ask_price_pos[lvl-1]]+0.01
min_price=book[event_idx,bid_price_pos[lvl-1]]-0.01

max_vol=max(book[event_idx,vol])

mid=0.5*(sum(book[event_idx,[0,2]],2))

#%%plot the Snapshot of the Limit Order Book
#------------------------------------------------------------------------------
plt.figure()
#ask price
color=["red","blue"]
y_pos=np.arange(11,21)
y_value=book[event_idx,ask_vol_pos]
plt.barh(y_pos, y_value,alpha=0.7,color=color[0],align="center",label="Ask")
#mid price
plt.plot([10,40],[10,10],'<g',markersize=10,fillstyle="full",label="Mid_price")
#bid price
y_pos=np.arange(0,10)
y_value=book[event_idx,bid_vol_pos][::-1]
plt.barh(y_pos,y_value,alpha=0.7,color=color[1],align="center",label="Bid")
#set style
y_pos=np.arange(0,21)
y_ticks=np.concatenate((book[event_idx,bid_price_pos][::-1],np.array([mid]),book[event_idx,ask_price_pos]),0)
plt.yticks(y_pos,y_ticks)
plt.xlabel('Volumne')
plt.title(ticker+"@"+str(demo_date[0])+"-"+str(demo_date[1])+
"-"+str(demo_date[2])+"\nLOB Snapshot -Time: "+str(mess[event_idx,0])+" Seconds")
plt.ylim([-1,21])
plt.legend()
plt.show()

#%%plot the relative depth in the Limit Oeder Book
#------------------------------------------------------------------------------



#% Relative volume ...

#% Ask
book_vol_ask = np.cumsum(book[event_idx,ask_vol_pos])
book_vol_ask = book_vol_ask/book_vol_ask[-1]

#% Bid
book_vol_bid = np.cumsum(book[event_idx,bid_vol_pos])
book_vol_bid = book_vol_bid/book_vol_bid[-1]

plt.figure()
#% Ask
plt.step(list(range(1,11)),book_vol_ask,color="g",label="Ask Depth")

plt.title(ticker+"@"+str(demo_date[0])+"-"+str(demo_date[1])+
"-"+str(demo_date[2])+"\nLOB Relative Depth -Time: "+str(mess[event_idx,0])+" Seconds")

plt.ylabel('% of Volume')
plt.xlabel('Level')

plt.xlim([1,10])

#Bid
plt.step(list(range(1,11)),-book_vol_ask,color="r",label="Bid Depth")

#y_pos=np.arange(0,21)
y_pos=np.linspace(-1,1,11)
plt.yticks(y_pos,[1,0.8,0.6,0.4,0.2,0,0.2,0.4,0.6,0.8,1])
plt.ylim([-1,1])

plt.show()

