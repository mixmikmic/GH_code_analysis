get_ipython().system('pip install redis')
get_ipython().system('pip install wordcloud')
from os import chdir
chdir('../')
get_ipython().magic('matplotlib inline')
from lib import *
suppress_warnings()

conn, cur = conpg(location = 'postgres')
sql_select = '''select created_at, location, tweet_content, cleaned_tweet, hashtags, date, time, date_time  
                from tweets where (date_time >= NOW() - '36 hour'::INTERVAL) and hashtags is not null and hashtags != 'None';'''
cur.execute(sql_select)
rows = cur.fetchall()
conn.close()
df = pd.DataFrame(rows)
df.reset_index(inplace = True)
df['created_datetime'] = pd.to_datetime(df['created_at'])
df['year'] = df.created_datetime.apply(lambda x: x.year)
df['month'] = df.created_datetime.apply(lambda x: x.month)
df['day'] = df.created_datetime.apply(lambda x: x.day)
df['hour'] = df.created_datetime.apply(lambda x: x.hour)
len(df)

df = df[~df['hashtags'].str.contains('amp')]
df = df[~df['hashtags'].str.contains('job|hiring|sales|retail|clerical|jobs|careerarc')]
df = df[~df['hashtags'].str.contains('dallas|plano|irving|richardson|garland|allen|odessa|midland|sanfrancisco|losangeles|irvine|garland')]
len(df)

#df[~df['location'].str.contains('.ca|california')][~df['location'].str.contains('.tx|texas')]
ca_df = df[df['location'].str.contains('.ca|california')]
tx_df = df[df['location'].str.contains('.tx|texas')]

hastages_series = df['hashtags']
len(hastages_series)

from PIL import Image
def word_cloud(state = None): 
    state=state.lower()
    #state_mask = np.array(Image.open("./doc/{}.jpg".format(state)))
    #state = state.lower()
    state_df = df[df['location'].str.contains('.{}'.format(state))]
    words = state_df['hashtags'][(state_df['hashtags'].isnull() == False) & (state_df['hashtags'] != 'None')]
    stopwords = set(STOPWORDS)
    #stopwords.add("amp")
    wc = WordCloud(width=1600, height=1600, background_color='black',                    relative_scaling=1, stopwords=stopwords, max_words=300, colormap='winter').generate(' '.join(i for i in words))
    #wc.to_file("./doc/{}.jpg".format(state))
    #, colormap='copper',mask=state_mask
    plt.figure(figsize=(10,10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    return

count_vectorizer = CountVectorizer(min_df = 1, stop_words='english')
hashtags_countvec = count_vectorizer.fit_transform(hastages_series)
#hashtags_countvec = pickle.dumps(hashtags_countvec_fit)
#r.set('hashtags_countvec_fit_temp', hashtags_countvec)
#hashtags_countvec = pickle.loads(r.get('hashtags_countvec_fit_temp'))

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    
    return result

def new_trend_period(trend_group):        
    new_result = []
    count = 0
    for i in range(len(trend_group)-1):
        if (count > len(trend_group)-2):
            pass
        else:
            gap = ((trend_group[count+1][0]) - (trend_group[count][len(trend_group[count])-1]))
            if gap <.8*min((trend_group[count+1][0]), (trend_group[count][len(trend_group[count])-1])):
                new_period = trend_group[count]+trend_group[count+1]
                new_result.append(new_period)
                count +=2
            else:
                new_result.append(trend_group[count])
                count +=1
    return new_result

def get_arr(hashtag):
    time_delta, min_time, time_window, time_lag, time_gap, windows = get_time()
    hashtag = hashtag.lower()
    arr = []
    start_time = min_time
    for window in range(windows):
        start_time = start_time
        end_time = start_time + time_lag
        subset = df['hashtags'][((df['created_datetime'] < end_time) & (df['created_datetime'] > start_time))]
        h = subset.str.contains(hashtag).mean()
        arr.append(h)
        start_time += time_gap    
    return arr

time_format = '%Y-%h-%d %a %I %p'
from_zone = tz.tzutc()
to_zone = 'US/Pacific'
def hashtag_trend(hashtag):
    time_delta, min_time, time_window, time_lag, time_gap, windows = get_time()
    arr = get_arr(hashtag)
    timeline = []
    w = np.array(range(windows))
    for i in w:
        timeline.append((min_time) + time_gap * i) 
    arr = np.array(arr)  
    grad = np.gradient(arr)
    time_df = pd.DataFrame({'arr':arr,'grad':grad},index=timeline)
    sma_arr = pd.rolling_mean(arr, window=4)
    sma_grad = np.gradient(sma_arr)
    
    arr_df = pd.DataFrame(arr)
    ewm_arr = arr_df.ewm(alpha=0.3,min_periods=4,adjust=False, ignore_na=True).mean()
    ewm_grad = np.gradient(np.array(ewm_arr).reshape(1,-1)[0])
    
    trend_sma = np.argwhere(sma_grad>0.001).reshape(1,-1)[0]
    trend_group_sma = group_consecutives(trend_sma)      
    trend_group_sma1 = new_trend_period(trend_group_sma)
    trend_group_sma2 = [i for i in trend_group_sma1 if (len(i) > 7)]
    
    trend_ewm = np.argwhere(ewm_grad>0.0001).reshape(1,-1)[0]
    trend_group_ewm = group_consecutives(trend_ewm)      
    trend_group_ewm1 = new_trend_period(trend_group_ewm)
    trend_group_ewm2 = [i for i in trend_group_ewm1 if (len(i) > 7)]
    #rolstd_arr = pd.rolling_std(arr, window=4)
    spike = []
    spike.append([[i[0],i[len(i)-1]] for i in trend_group_ewm2])
    spike = np.array(spike)
    spk1 = [i[0] for i in spike[0]]
    spk2 = [i[1] for i in spike[0]] 
    
    forecast, stderr, con_int = get_arima(hashtag)
    forecast = np.append(arr,forecast)
    stderr = np.append(arr,stderr)
    upper = np.append(arr,con_int[:,0])
    lower = np.append(arr,con_int[:,1])
    plt.figure(figsize=(20,10))
    plt.axvline(w[-1], color = 'lightgray',linestyle='dashed')
    plt.plot(forecast, c = 'gray',linewidth = 3, label = 'forecast')
    #plt.plot(stderr, c = 'lightcoral',linestyle='dashed', label = 'Standard Error')
    plt.plot(upper, c = 'gray',linestyle='dashed', label = 'Confidence Interval')
    plt.plot(lower, c = 'gray',linestyle='dashed')
    plt.plot(arr, label='Hashtag Frequency',c = 'dimgray',linewidth = 3)
    plt.plot(ewm_arr, label='Exponential Weighted Moving Avg(EMA)',c = 'salmon',linewidth = 2)
    plt.plot(sma_arr, label='Simple Moving Avg(SMA)',c = 'dodgerblue')
    #plt.plot(rolstd_arr, label='Rolling STDs',c = 'gold')
    plt.plot(sma_grad, label='Gradient over SMA',c = 'mediumblue')
    plt.plot(ewm_grad, label='Gradient over EMA',c = 'red')
    for j,k in zip(spk1,spk2):
        spike1 = (min_time + time_gap * j).replace(tzinfo=from_zone).astimezone(to_zone).strftime(time_format)                   
        plt.axvline(j, color = 'salmon',linestyle='dashed', label = 'Trending: {}'.format(spike1))
        plt.axvspan(j, k, alpha=0.2, color='lightcoral')
    plt.title(hashtag, fontsize=20)    
    plt.legend(fontsize=12) 
    plt.show()
    return  

def get_time():
    time_delta = (max(df['created_datetime'])) - min(df['created_datetime'])
    min_time = min(df['created_datetime'])
    time_window = time_delta.components.days*24 + time_delta.components.hours
    time_lag = timedelta(hours = .5)
    time_gap = timedelta(hours = .5)
    windows = int(round(time_delta/time_gap,0))
    return time_delta, min_time, time_window, time_lag, time_gap, windows

from statsmodels.tsa.arima_model import ARIMA
def fit_arima(arr):
    arima_df = pd.DataFrame({'perf':arr})   
    series = arima_df.dropna()
    model = ARIMA(np.array(series.values), order=(4,1,0))
    model_fit = model.fit()
    return model_fit

## ARIMA
## AR: Autoregression
## I: Intergrated
## MA: Moving Average
## order: p,q,d
## p: # of lag observations included in the model
## d: # of times in the raw observations are differenced
## q: the size of the moving average window
from statsmodels.tsa.arima_model import ARIMA
def get_arima(hashtag):
    arr = get_arr(hashtag)
    model_fit = fit_arima(arr)
    forecast_ = model_fit.forecast(steps=10)
    forecast = forecast_[0]+arr[-1]
    stderr = forecast_[1]+arr[-1]
    con_int = forecast_[2]+arr[-1]
    return forecast, stderr, con_int

from statsmodels.tsa.arima_model import ARIMA
def arima_residuals_plt(hashtag): 
    arr = get_arr(hashtag)
    model_fit = fit_arima(arr)
    print(model_fit.summary())
    # plot residual errors
    residuals =pd. DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())

hashtag_freq_df = pd.DataFrame({'hashtag': count_vectorizer.get_feature_names(), 'occurrences':np.asarray(hashtags_countvec.sum(axis=0)).ravel().tolist()})
hashtag_freq_df['frequency'] = hashtag_freq_df['occurrences']/np.sum(hashtag_freq_df['occurrences'])
hashtag_freq_df.sort_values(by = 'occurrences',ascending = False, inplace = True)

word_cloud('TX')

word_cloud('CA')

time_format = '%Y-%h-%d %a %I %p'
from_zone = tz.tzutc()
to_zone = 'US/Pacific'
time_delta, min_time, time_window, time_lag, time_gap, windows = get_time()
print('\tstart time: ',min_time.replace(tzinfo=from_zone).astimezone(to_zone).strftime(time_format),'\n',      '\tend time:  ', max(df['created_datetime']).replace(tzinfo=from_zone).astimezone(to_zone).strftime(time_format),'\n',      '\ttotal hours: ',time_window,'\n',      '\ttime lag: ', time_lag,'\n',      '\ttime gap: ', time_gap,'\n',      '\ttime windows: ', windows)

hashtag_freq_df.head(10)

[hashtag_trend(i) for i in (hashtag_freq_df['hashtag'].head(10))]

hashtag_trend('funny')

arima_residuals_plt('la')

hashtag_trend('earthquake')

arima_residuals_plt('earthquake')

hashtag_trend('ga06')























