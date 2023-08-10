import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

house1_aggregate = pd.read_csv('../../master_data/house_1/channel_1.dat', header = None,
                                    names = ["Time","Power"], delim_whitespace = True,
                                    index_col = 0)
house2_aggregate = pd.read_csv('../../master_data/house_2/channel_1.dat', header = None,
                                    names = ["Time","Power"], delim_whitespace = True,
                                    index_col = 0)
kettle_power = pd.read_csv('../../master_data/house_1/channel_10.dat', header = None,
                                    names = ["Time","Power"], delim_whitespace = True,
                                    index_col = 0)
kettle_power_h2 = pd.read_csv('../../master_data/house_2/channel_8.dat', header = None,
                                    names = ["Time","Power"], delim_whitespace = True,
                                    index_col = 0)



def get_signatures(aggregate_power, power_series, min_off_duration=30, min_on_duration=12, timestep = '6s',
                   power_threshold=200, pad = 1024, sample_length = 128):
    signatures = []
    aggregate_intervals = []
    signal_intervals = []
    signal_start = None
    signal_end = None
    no_signal = 0
    series_length = len(power_series)
    series_length_percent = int(len(power_series)/100)
    measurements_processed = 0

    response = pd.DataFrame(columns = [i/sample_length for i in range(sample_length)])
    #response.loc[0] = [0 for _ in range(sample_length)]
    zeros = np.zeros(sample_length)
    
    
    power_series = power_series.astype(np.float64)
    aggregate_power = aggregate_power.astype(np.float64)
    
    aggregate_power.index = pd.to_datetime(aggregate_power.index,unit='s') 
    power_series.index = pd.to_datetime(power_series.index,unit='s')
    min_off_duration = pd.Timedelta(str(min_off_duration) + 's')
    min_on_duration = pd.Timedelta(str(min_off_duration) + 's')
    timestep = pd.Timedelta(timestep)
    
    i=0
    
    for time in power_series.index:
        measurement = power_series.loc[time]['Power']
        measurements_processed +=1
        
        if measurements_processed % series_length_percent == 0:
            print(str(measurements_processed/series_length_percent) + '%, discovered ', i)
        
        if (measurement >= power_threshold):
            if signal_start is None:
                signal_start = time - 2*timestep
            no_signal = 0
            signal_end = time + 2*timestep
            
        else:
            if signal_end is not None:
                no_signal = time - signal_end
                if (no_signal > min_off_duration) and (signal_start is not None):
                    if (signal_end - signal_start) > min_on_duration:
                        
                        #pad_start = np.random.randint(0,pad)
                        #sample_start = signal_start - pd.Timedelta(pad_start, unit='s')
                        #sample_end = signal_end + pd.Timedelta(pad - pad_start, unit='s')
                        
                        #sample = aggregate_power[sample_start:sample_end]
                        
                        sample_start = np.argmax(aggregate_power.index > signal_start)
                        sample_end = np.argmax(aggregate_power.index > signal_end)
                        
                        signal = power_series[signal_start:signal_end]['Power']
                        signal.index = np.arange(signal.shape[0])
                        signal = signal.reindex(np.arange(sample_end - sample_start),method='nearest').values
                        
                        diff = sample_length - (sample_end-sample_start)
                        dice = np.random.randint(0,diff)
                        
                        sample = aggregate_power.iloc[sample_start - dice: sample_start - dice + sample_length]
                    
                        response = np.zeros(sample_length)

                        if dice + signal.shape[0] > sample_length:
                            continue #due to breaks in the aggregate data. shouldn't be too big of an issue.
                            
                        response[dice:dice + signal.shape[0]] += signal
                        
                        i+=1
                        signatures.append(response)
                        signal_intervals.append([signal_start,signal_end])
                        aggregate_intervals.append(sample['Power'].values)
                    
                    current_signal = []
                    signal_start = None
                    signal_end = None
    
    return signatures, aggregate_intervals, signal_intervals

kettle_signatures, kettle_aggregates, signals = get_signatures(house1_aggregate, kettle_power, min_on_duration = 12,
                                                              power_threshold = 1000, min_off_duration=0)

kettle_signatures_test, kettle_aggregates_test, signals_test = get_signatures(house2_aggregate, kettle_power_h2,
                                                                              min_on_duration = 12,
                                                                              power_threshold = 1000, min_off_duration=0)

dice = np.random.randint(len(kettle_signatures_test))
print(dice)
plt.plot(kettle_signatures_test[dice])
plt.plot(kettle_aggregates_test[dice])
plt.show()

kettle_signatures_pd = pd.DataFrame(kettle_signatures, columns = np.linspace(0,1,128))
kettle_signatures_pd.head()

kettle_aggregates_pd = pd.DataFrame(kettle_aggregates, columns = np.linspace(0,1,128))
kettle_aggregates_pd.head()

signals_pd = pd.DataFrame(signals, columns = ['start','end'])
signals_pd.head()

kettle_signatures_test_pd.head()

kettle_signatures_pd.to_csv('../../master_data/nilm/kettle_signatures.dat', sep = ' ')
kettle_aggregates_pd.to_csv('../../master_data/nilm/kettle_input.dat', sep = ' ')
signals_pd.to_csv('../../master_data/nilm/kettle_signals.dat', sep = ' ')

kettle_signatures_test_pd = pd.DataFrame(kettle_signatures_test, columns = np.linspace(0,1,128))
kettle_aggregates_test_pd = pd.DataFrame(kettle_aggregates_test, columns = np.linspace(0,1,128))
signals_test_pd = pd.DataFrame(signals_test)

kettle_signatures_test_pd.to_csv('../../master_data/nilm/kettle_signatures_test.dat', sep = ' ')
kettle_aggregates_test_pd.to_csv('../../master_data/nilm/kettle_input_test.dat', sep = ' ')
signals_test_pd.to_csv('../../master_data/nilm/kettle_signals_test.dat', sep = ' ')

def clean_power_series(aggregate_power, signal_intervals):
    #aggregate_power.index = pd.to_datetime(aggregate_power.index,unit='s')
    #signal_intervals['start'] = pd.to_datetime(signal_intervals['start'],unit='s')
    #signal_intervals['end'] = pd.to_datetime(signal_intervals['end'],unit='s')
    
    temp_ap = aggregate_power.copy()
    
    n = signal_intervals.shape[0]

    pcent = int(n/100)
    ap_sans_application = pd.DataFrame()
    
    ap_sans_application.append(temp_ap[:signal_intervals['start'][0]])
    
    for i in range(1,n):
        if i % pcent == 0:
            print(str(i/pcent) + '%')
        ap_sans_application = ap_sans_application.append(temp_ap[signal_intervals['end'][i-1]:signal_intervals['start'][i]])
    
    ap_sans_application.append(temp_ap[signal_intervals['end'][i]:])
    
    return ap_sans_application

        
def make_syntethic_data(aggregate_power, signal_intervals, app_signatures, n = 10000,
                        ap_sans_application = None, sample_length = 128):
    
    if ap_sans_application is None:
        ap_sans_application = clean_power_series(aggregate_power, signal_intervals)
    
    sign_len = len(app_signatures)
    
    syntethic_data = pd.DataFrame(columns=np.linspace(0,1,sample_length))
    syntethic_response = pd.DataFrame(columns=np.linspace(0,1,sample_length))
    
    pcent = n/100
    
    
    for i in range(n):
        
        if i % pcent == 0:
            print(i/pcent)
        
        syntethic_response.loc[i] = [0 for _ in range(sample_length)]
        dice = np.random.random()
        
        app_no = np.random.randint(0,sign_len-1)
        sample_start = np.random.randint(0,ap_sans_application.shape[0]-(sample_length+1))
        sample_end = sample_start + sample_length - 1
        
        sample_start_time = ap_sans_application.index[sample_start]
        sample_end_time = ap_sans_application.index[sample_end] 
        
        sample = ap_sans_application.loc[sample_start_time:sample_end_time]
        signal = app_signatures.loc[app_no].dropna()
    
        if dice > 0.5:
            signal_start = np.random.randint(0,sample_length - (signal.shape[0] + 1))
            signal_end = signal_start + signal.shape[0]
            
            sample.iloc[signal_start:signal_end] = sample.iloc[signal_start:signal_end].values[0] + signal.values
            
            signal_start_time = sample.index[signal_start]
            signal_end_time = sample.index[signal_end]
            
            start_fraction = (signal_start_time - sample.index[0])/(sample.index[-1] - sample.index[0])
            end_fraction = (signal_end_time - sample.index[0])/(sample.index[-1] - sample.index[0])

            syntethic_data.loc[i] = sample.T.values[0]
            syntethic_response.loc[i].iloc[signal_start:signal_end] = signal.values
        
        else:
            syntethic_data.loc[i] = sample.T.values[0]
            #syntethic_response.loc[i] = [0 for _ in range(sample_length)]
            
    return syntethic_data, syntethic_response
    

clean_data = clean_power_series(house1_aggregate,signals_pd)

house1_aggregate['Power'] = house1_aggregate['Power'].astype(np.float64)
kettle_signatures_pd = kettle_signatures_pd.astype(np.float64)

syntethic_kettle_data, syntethic_kettle_response = make_syntethic_data(house1_aggregate, signals_pd,
                                                                       kettle_signatures_pd,n=10000,
                                                                       ap_sans_application = clean_data)

print(syntethic_kettle_data.isnull().any(axis=1).any())
print(syntethic_kettle_response.isnull().any(axis=1).any())
dice = np.random.randint(0,syntethic_kettle_data.shape[0])
syntethic_kettle_data.loc[dice].plot()
plt.show()
syntethic_kettle_response.loc[dice].plot()
plt.show()

syntethic_kettle_response.head()

syntethic_kettle_data.to_csv('../../master_data/nilm/syntethic_kettle_dae.dat', sep = ' ')
syntethic_kettle_response.to_csv('../../master_data/nilm/syntethic_kettle_dae_response.dat', sep = ' ')

