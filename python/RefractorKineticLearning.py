import pandas as pd
from IPython.display import display
from scipy.signal import savgol_filter
import numpy as np
from tpot import TPOTRegressor
from scipy.interpolate import interp1d

#Decorators
def evenly_space(fun,times):
    '''Decorate Functions that require even spacing.'''
    
    pass

def format_dataframe(raw_df,states,controls,impute=True,time='Hour',strain='Strain',augment=None):
    '''Put DataFrame into the TSDF format.
    
    The input csv or dataframe should have a 
    column for time and ever state and control
    variable input for that time. Optional Columns are
    "Replicate" and "Strain".
    
    '''
    
    #Remove Unused Columns
    raw_df = raw_df[[strain,time] + states+controls]
    
    #Impute NaN Values using Interpolation
    if impute:
        raw_df = raw_df.set_index([strain,time])
        tsdf = raw_df.groupby(strain).apply(lambda group: group.interpolate())
    
    #Format Columns to Only Include States and Controls
    columns = [('states',state) for state in states] + [('controls',control) for control in controls]
    tsdf.columns = pd.MultiIndex.from_tuples(columns)    
        
    return tsdf

def augment_data(tsdf,n=200,strain='Strain',time='Hour'):
    '''Augment the time series data for improved fitting.
    
    The time series data points are interpolated to create
    smooth curves for each time series and fill in blank 
    values.
    '''
    
    def augment(df):
        #Find New Times
        times = df.index.get_level_values(1)
        new_times = np.linspace(min(times),max(times),n)
        
        #Build New Indecies
        strain_name = set(df.index.get_level_values(0))
        new_indecies = pd.MultiIndex.from_product([strain_name,new_times])
        
        #Reindex the Data Frame & Interpolate New Values
        df = df.reindex(df.index.union(new_indecies))
        df.index.names = [strain,time]
        df = df.interpolate()
        
        #Remove Old Indecies
        df.index = df.index.droplevel(0)
        times_to_remove = set(times) - (set(times) & set(new_times))
        df = df.loc[~df.index.isin(times_to_remove)]
        return df
            
    tsdf = tsdf.groupby(strain).apply(augment)
    return tsdf


def estimate_state_derivative(tsdf,time='Hour',strain='Strain'):
    '''Estimate the Derivative of the State Variables'''
    
    #Check if a vector is evenly spaced
    evenly_spaced = lambda x: max(set(np.diff(x))) - min(set(np.diff(x))) < 10**-5
    
    #Find the difference between elements of evenly spaced vectors
    delta = lambda x: np.diff(x)[0]

    #Find Derivative of evenly spaced data using the savgol filter
    savgol = lambda x: savgol_filter(x,7,2,deriv=1, delta=delta(x))

    def estimate_derivative(tsdf):
        state_df = tsdf['states']
        times = state_df.index.get_level_values(1)
    
        if evenly_spaced(times):
            state_df = state_df.apply(savgol)      
        else:     
            state_df = state_df.apply(savgol_uneven)
            
        #Add Multicolumn
        state_df.columns = pd.MultiIndex.from_product([['derivatives'],state_df.columns])

        #Merge Derivatives Back
        tsdf = pd.merge(tsdf, state_df,left_index=True, right_index=True,how='left')

        return tsdf
    
        
    tsdf = tsdf.groupby(strain).apply(estimate_derivative)
    return tsdf

class dynamic_model(object):
    '''A MultiOutput Dynamic Model created from TPOT'''
    
    def __init__(self,tsdf):
        self.tsdf = tsdf

    
    def search(self):
        '''Find the best model that fits the data with TPOT.'''
        
        X = self.tsdf[['states','controls']].values
        
        def fit_single_output(row):
            tpot = TPOTRegressor(generations=2, population_size=5, verbosity=2,n_jobs=1)
            fit_model = tpot.fit(X,row).fitted_pipeline_
            return fit_model
    
        self.model_df = self.tsdf['derivatives'].apply(fit_single_output).to_frame()
        display(self.model_df)

    def fit(self,tsdf):
        '''Fit the Dynamical System Model.
        
        Fit the dynamical system model and
        return the map f.
        '''
        
        #update the data frame
        self.tsdf = tsdf
        X = self.tsdf[['states','controls']].values
        
        #Fit the dataframe data to existing models
        #self.model_df.apply(lambda model: print(model),axis=1)
        self.model_df = self.model_df.apply(lambda model: model[0].fit(X,self.tsdf['derivatives'][model.name]),axis=1)
    
    
    def predict(self,X):
        '''Return a Prediction'''
        y = self.model_df.apply(lambda model: model[0].predict(X),axis=1).values
        
        return y 
    
    
    def fit_report(self):
        '''Report the Quality of the Fit in Plots'''
        
        pass

def learn_dynamics(df,states,controls,data_augmentation=None):
    '''Find system dynamics Time Series Data.
    
    Take in a Data Frame containing time series data 
    and use that to find the dynamics x_dot = f(x,u).
    '''
    
    #Clean the data and get dataframe into correct format
    tsdf = format_dataframe(df,states,controls)
    
    #Augment the data using an interpolation scheme
    if data_augmentation is not None:
        tsdf = augment_data(tsdf,n=data_augmentation)
    
    #Estimate the Derivative
    tsdf = estimate_state_derivative(tsdf)
    
    #Fit Model
    model = dynamic_model(tsdf)
    model.search()
    
    return model


def simulate_dynamics(model,strain_df,time_points=None):
    '''Use Learned Dynamics to Generate a Simulated Trajectory in the State Space'''
    
    times = strain_df.index.get_level_values(1)
    
    #Get Controls as a Function of Time Using Interpolations
    u_df = strain_df['controls'].apply(lambda y: interp1d(times,y))
    display(u_df)
    #Get Initial Conditions from the Strain Data Frame
    
    
    #Solve Differential Equation For Same Time Points
    
    
    #Return DataFrame with Predicted Trajectories
    
    
    return trajectory_df

#Import Limonene Data and Format Data Frame
limonene_df = pd.read_csv('data/limonene_data.csv')
#display(limonene_df)

controls = ['AtoB', 'GPPS', 'HMGR', 'HMGS', 'Idi','Limonene Synthase', 'MK', 'PMD', 'PMK']
states = ['Acetyl-CoA','HMG-CoA', 'Mevalonate', 'Mev-P', 'IPP/DMAPP', 'Limonene']


model = learn_dynamics(limonene_df,states,controls,data_augmentation=200)

tsdf = format_dataframe(limonene_df,states,controls)
tsdf = augment_data(tsdf)
tsdf = estimate_state_derivative(tsdf)

model.predict(tsdf.sample()[['states','controls']].values)

strain_df = tsdf.loc[tsdf.index.get_level_values(0)=='L1']
simulate_dynamics(model,strain_df)

