def initialize_datasets(df):
    """this function takes in a dataframe generated from the Matlab files. For the partial discharge curves and 
    initializes the data so that cycle 1 corresponds to the beginning of the partial cycling experiments, and
    not to full charge/discharge data."""
    # Initializing each parameter with the first datasets

    # Will need to do a separate for loop to see if where we initalize the code is correct.
    # Coordinate the count and the index so that if the data series in the count is < 3000, pass and
    # move onto the next index. If it finally gets initialized, then do the for loop to continue in adding
    # all the other mumbo jumbo

    count = 2
    for idx in range(1, len(df[data])):
        time_sum = df['#subsystem#'][0][0][0][count][0][0].flatten()
        if len(time_sum) < 3500:
            count = count + 7
        else:
            datetime_sum = df['#subsystem#'][0][0][0][count][0][1].flatten()
            step_sum = df['#subsystem#'][0][0][0][count][0][2].flatten()
            cycle_sum = df['#subsystem#'][0][0][0][count][0][3].flatten()
            current_sum = df['#subsystem#'][0][0][0][count][0][4].flatten()
            voltage_sum = df['#subsystem#'][0][0][0][count][0][5].flatten()
            charge_ah_sum = df['#subsystem#'][0][0][0][count][0][6].flatten()
            discharge_ah_sum = df['#subsystem#'][0][0][0][count][0][7].flatten()
            count = count + 7
            break
    
    return count, idx, time_sum, datetime_sum, step_sum, cycle_sum, current_sum, voltage_sum, charge_ah_sum, discharge_ah_sum

def single_pd_matlab_partialdata(filename):
    """this function will be used to load in the partial discharge data because it requires a unique
    initialization"""
    import numpy as np
    import pandas as pd
    
    df = import_matlab_data(filename)
    keys = df.keys()

    for i in keys:
            if i == '#subsystem#':
                pass
            else:
                data = i

    count, end_idx, time_sum, datetime_sum, step_sum, cycle_sum, current_sum, voltage_sum, charge_ah_sum, discharge_ah_sum = initialize_datasets(df)
    start_ind = end_idx + 1

    for idx in range(int(start_ind), len(df[data])):
        length_test = (df['#subsystem#'][0][0][0][count][0][0].flatten())

        if df[data][idx][2].shape == (0,0):
            name = df[data][idx][0][0][0]
            name = name.split(' ')
            number = float(name[0])
            cycle_sum = np.append(cycle_sum, (cycle_sum[-1] + number))
        elif len(length_test) < 3500:
            count = count + 7
        else:
            time_idx = (df['#subsystem#'][0][0][0][count][0][0].flatten()) + time_sum[-1]
            time_sum = np.concatenate((time_sum, time_idx), axis = 0)

            datetime_idx = df['#subsystem#'][0][0][0][count][0][1].flatten()
            datetime_sum = np.concatenate((datetime_sum, datetime_idx), axis = 0)

            step_idx = df['#subsystem#'][0][0][0][count][0][2].flatten()
            step_sum = np.concatenate((step_sum, step_idx), axis = 0)

            if len(cycle_sum) > len(current_sum):
                diff = len(cycle_sum) - len(current_sum)
                cycle_prior = cycle_sum[-1]
                for i in range(0,diff):
                    cycle_sum = np.delete(cycle_sum, -1)
                cycle_idx = (df['#subsystem#'][0][0][0][count][0][3].flatten()) + cycle_prior
                cycle_test = (df['#subsystem#'][0][0][0][count][0][3].flatten())
                cycle_sum = np.concatenate((cycle_sum, cycle_idx), axis = 0)
            else:
                cycle_prior = cycle_sum[-1]
                cycle_idx = (df['#subsystem#'][0][0][0][count][0][3].flatten()) + cycle_prior
                cycle_test = (df['#subsystem#'][0][0][0][count][0][3].flatten())
                cycle_sum = np.concatenate((cycle_sum, cycle_idx), axis = 0)

            current_idx = df['#subsystem#'][0][0][0][count][0][4].flatten()
            current_sum = np.concatenate((current_sum, current_idx), axis = 0)

            voltage_idx = df['#subsystem#'][0][0][0][count][0][5].flatten()
            voltage_sum = np.concatenate((voltage_sum, voltage_idx), axis = 0)

            charge_ah_idx = df['#subsystem#'][0][0][0][count][0][6].flatten()
            charge_ah_sum = np.concatenate((charge_ah_sum, charge_ah_idx), axis = 0)

            discharge_ah_idx = df['#subsystem#'][0][0][0][count][0][7].flatten()
            discharge_ah_sum = np.concatenate((discharge_ah_sum, discharge_ah_idx), axis = 0)

            count = count + 7

            if count > (len(df['#subsystem#'][0][0][0])-7):
                break


    host_df = pd.DataFrame()

    host_df['time'] = time_sum
    host_df['datetime'] = datetime_sum
    host_df['step'] = step_sum
    host_df['cycle'] = cycle_sum
    host_df['current_amp'] = current_sum
    host_df['voltage'] = voltage_sum
    host_df['charge_ah'] = charge_ah_sum
    host_df['discharge_ah'] = discharge_ah_sum
    
    return host_df

