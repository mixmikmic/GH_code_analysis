from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sample_length = 32

aggregate_power = pd.read_csv('../master_data/house_2/channel_1.dat', header = None,
                                names = ["Time","Power"], delim_whitespace = True,
                                index_col = 0)
aggregate_power.index = pd.to_datetime(aggregate_power.index,unit='s')

kettle_power = pd.read_csv('../master_data/house_2/channel_8.dat', header = None,
                                names = ["Time","Power"], delim_whitespace = True,
                                index_col = 0)


kettle_power.head()

def get_activations(power_series, min_off_duration=30, min_on_duration=12,
                        border=1, power_threshold=200):
    activations = []
    signal_start = None
    signal_end = None
    no_signal = 0
    current_signal = []
    series_length = len(power_series)
    series_length_percent = int(len(power_series)/100)
    measurements_processed = 0
    
    power_series.index = pd.to_datetime(power_series.index,unit='s')
    min_off_duration = pd.Timedelta(str(min_off_duration) + 's')
    min_on_duration = pd.Timedelta(str(min_off_duration) + 's')
    timestep = pd.Timedelta('6s')
    
    for time in power_series.index:
        measurement = power_series.loc[time]['Power']
        measurements_processed +=1
        
        if measurements_processed % series_length_percent == 0:
            print(str(measurements_processed/series_length_percent) + '%')
        
        if (measurement >= power_threshold):
            if signal_start is None:
                signal_start = time - timestep
            no_signal = 0
            current_signal.append(measurement)
            signal_end = time + timestep
            
        else:
            if signal_end is not None:
                no_signal = time - signal_end
                if (no_signal >= min_off_duration) and (signal_start is not None):
                    if (signal_end - signal_start) > min_on_duration:
                        mean_power = np.mean(current_signal)
                        activations.append([signal_start,signal_end,mean_power])
                    current_signal = []
                    signal_start = None
                    signal_end = None
    
    return activations

kettle_activations = get_activations(kettle_power)

kettle_activations_pd = pd.DataFrame(kettle_activations, columns = ['start','end','mean_power'])
kettle_activations_pd['start'] = pd.to_datetime(kettle_activations_pd['start'])
kettle_activations_pd['end'] = pd.to_datetime(kettle_activations_pd['end'])

kettle_activations_pd.head()

for _ in range(1):
    dice = np.random.randint(0,100)
    start = kettle_activations_pd.iloc[dice]['start']
    end = kettle_activations_pd.iloc[dice]['end']
    print(start,end)
    aggregate_power[start:end].plot()
    plt.show()

def clean_power_series(aggregate_power, activations):
    
    temp_ap = aggregate_power.copy()
    
    n = activations.shape[0]

    #pcent = int(n/100)
    ap_sans_activations = pd.DataFrame()
    
    ap_sans_activations.append(temp_ap[:activations['start'][0]])
    
    for i in range(1,n):
        ap_sans_activations = ap_sans_activations.append(temp_ap[activations['end'][i-1]:activations['start'][i]])
    
    ap_sans_activations.append(temp_ap[activations['end'][i]:])
    
    return ap_sans_activations


def create_training_set(aggregate_power, activations, ap_sans_activations = None, sample_length = 128, pad = 1024):
    training_set = pd.DataFrame(columns = np.linspace(0,1,sample_length))
    training_set_response = pd.DataFrame(columns = ['start','end','mean_power'])

    #aggregate power sans the activation intervals
    if ap_sans_activations is None:
        ap_sans_activations = clean_power_series(aggregate_power,activations)

    n = aggregate_power.shape[0]
    j = 0
    for i in range(activations.shape[0]):
        if i % 100 == 0:
            print(i)
        while True:
            dice = np.random.random()
            if dice < 0.5:
                start = activations['start'][i]
                end = activations['end'][i]
                #randomly place the chosen signal in a window.
                #note that the entire signal is always in the window 
                sample_start = start - pd.Timedelta(np.random.randint(0,pad),unit='s')
                sample_end = end + pd.Timedelta(np.random.randint(0,pad),unit='s')
                sample = aggregate_power[sample_start:sample_end]
                while sample.shape[0] > sample_length:
                    if np.random.random() > 0.5 and end < sample.index[-1]:
                        sample = sample.iloc[:-1]
                    elif start > sample.index[0]:
                        sample = sample.iloc[1:]
                    else:
                        sample = sample.iloc[0:1]
                
                if not sample.shape[0] < sample_length:
                    
                    #sample = sample.reindex(columns=training_set.columns, method='nearest')
                    start_fraction = (start - sample.index[0])/(sample.index[-1] - sample.index[0])
                    end_fraction = (end - sample.index[0])/(sample.index[-1] - sample.index[0])
                    training_set.loc[j] = sample.T.values[0]
                    training_set_response.loc[j] = [start_fraction,end_fraction,activations['mean_power'][i]]
                    j+=1
                
                break
                
                
            ri = np.random.randint(0,ap_sans_activations.shape[0]-sample_length)
            sample = ap_sans_activations.iloc[ri: ri+sample_length]
            training_set.loc[j] = sample.T.values[0]
            training_set_response.loc[j] = [0,0,0]
            j+=1
            
    return training_set, training_set_response


#downsample the input:
aggregate_power_15m = aggregate_power.resample('15min').mean()
print(aggregate_power.shape, aggregate_power_15m.shape)
aggregate_power_15m.head()



aggregate_power_15m = aggregate_power_15m.dropna()
aggregate_power_15m.shape

clean_ap = clean_power_series(aggregate_power_15m,kettle_activations_pd)

X, y = create_training_set(aggregate_power_15m, kettle_activations_pd,
                                            ap_sans_activations = clean_ap,
                                            sample_length = sample_length, pad = 40000)
X.head()

from keras.models import load_model

model = load_model('../master_data/nilm/models/model0122ds15.h5')

normalization = pd.read_csv('../master_data/nilm/normalization_params.csv', header=0, delim_whitespace=True)
normalization.head()

X_np = np.array(X).reshape((X.shape[0],X.shape[1],1))
y_np = np.array(y).reshape((y.shape[0],y.shape[1]))

#mean = normalization['mean'].values[0]
mean = X_np.mean(axis=1).reshape(X_np.shape[0],1,1)
X_np = X_np - mean
sd = normalization['sd'].values[0]
#rand_sd = rand_sd.sample(frac=1).reset_index(drop=True)
X_np /= sd
print("Mean: ", X_np.mean())
print("Std: ", X_np.std())


pred = model.predict(X_np)

#Scale experiment:
pred[:,0] = pred[:,0]/1000
pred[:,1] = pred[:,1]/1000

diff_power = pred[:,2] - y_np[:,2]
diff_power = diff_power.reshape(diff_power.shape[0],1)
diff_clean = pred[:,2][y_np[:,2] == 0] - y_np[:,2][y_np[:,2] == 0]
diff_clean = diff_clean.reshape(diff_clean.shape[0],1)
diff_signal = pred[:,2][y_np[:,2] != 0] - y_np[:,2][y_np[:,2] != 0]
diff_signal = diff_signal.reshape(diff_signal.shape[0],1)

print(diff_power.shape)

plt.hist(diff_power,bins=50, color='teal')
plt.title("Distribution of prediction - measurement")
plt.show()

plt.hist(diff_clean,bins=50, color='teal')
plt.title("Distribution of prediction - measurement with no signal present")
plt.show()

plt.hist(diff_signal,bins=50, color='teal')
plt.title("Distribution of prediction - measurement with signal present")
plt.show()

#F1 score = 2*(recall*precision)/(recall+precision)
signal_present_gold = np.zeros(pred.shape[0])
signal_present_gold[np.where(y_np[:,2] > 0)] = 1
accs = []
thresholds = np.arange(20,2000,10)

for i in thresholds:
    signal_present_pred = np.zeros(pred.shape[0])
    signal_present_pred[np.where(pred[:,2] >= i)] = 1
    
    precision = signal_present_gold[signal_present_gold + signal_present_pred == 2].shape[0]/np.sum(signal_present_pred)
    recall = signal_present_gold[signal_present_gold + signal_present_pred == 2].shape[0]/np.sum(signal_present_gold)
    f1 = 2*(recall*precision)/(recall+precision)
    
    accs.append(f1)
    
print("Best F1: ", max(accs))
plt.plot(thresholds,accs)
plt.xticks = thresholds
plt.show()

#Error in duration:

duration_pred = pred[:,1] - pred[:,0]
duration_measured = y_np[:,1] - y_np[:,0]
sane_index = np.where(np.logical_and(duration_pred < 1,duration_pred > 0))
sane_duration_pred  = duration_pred[sane_index]
sane_duration_measured = duration_measured[sane_index]

duration_mae = np.abs(duration_pred - duration_measured)
duration_mae = duration_mae.reshape(duration_mae.shape[0],1)
sane_duration_mae = np.abs(sane_duration_pred - sane_duration_measured).mean()

plt.clf()
plt.hist(duration_mae,bins = 50, color='teal')
plt.title("Distribution of predicted signal duration error")
plt.show()

start_diff = np.abs(pred[:,0] - y_np[:,0])
start_diff = start_diff.reshape(start_diff.shape[0],1)

plt.hist(start_diff,bins = 50, color='teal')
plt.title("Distribution of predicted signal start error")
plt.show()

end_diff = np.abs(pred[:,1] - y_np[:,1])
end_diff = end_diff.reshape(end_diff.shape[0],1)

plt.hist(end_diff,bins = 50, color='teal')
plt.title("Distribution of predicted signal end error")
plt.show()

