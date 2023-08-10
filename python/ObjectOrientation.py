get_ipython().magic('matplotlib notebook')
import os.path
import pandas as pd
import pylab as plt

class BaseData(object):
    ''' Keep simulation or measurement data
    
        Parameters
        -----------
        filename : str
            path to filename
            
        name : str, optional [None]
            It is always a good idea to label your data, if None, name will
            be determined from filename
        
        Attributes
        -----------
        data : pd.Series
            data read from result file, c
            
        savedir : str
            path were results/figures etc. should be saved, defaults to 'D:/'
    '''
    
    def __init__(self, filename, name=None):
        self.filename = filename
        self.savedir = 'D:/'
        self.data = self.read_data()
        #If no name is provided, the name will be created from the filename
        if name is None:
            _dummy = (os.path.basename(filename).split('.')[:-1]) # Get the filename without suffix
            self.name = ''.join(_dummy) # Rebuild a string from resulting list
        else:
            self.name = name
               
    def read_data(self):
        ''' Read data from input file
        
            Returns:
            ----------
            pd.DataFrame
        '''
        data = pd.Series.from_csv(self.filename, header=0)
        return data
    
    def plot_data(self, label=None, fig=None, ax=None):
        ''' Plot the data into a figure
        
            Parameters:
            ----------
            label : str, optional [self.name]
                Identifier for this function (used in the legend)
            fig : matplotlib.figure instance, optional [None]
                Figure to plot into, will be created if None
            ax : matplotlib.axes instance, optional [None]
                Axes to plot into, will be created if None
        
            Returns:
            ----------
            None
            
            Notes:
            ----------
            In the EBC-Python Library exist an advanced helper to create figures and axes
            (the first few lines in this code), see base_functions.reusable_helpers.helper_figures
        
        '''
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            fig = ax.get_figure()
        elif ax is None:
            ax = fig.add_axes()
        if label is None:
            label = self.name
        self.data.plot(ax=ax, label=label)
        ax.set_title('Power vs. travel' )
        ax.set_ylabel('Power in kW')
        ax.legend()
        
a = BaseData('Referenz.csv')
b = BaseData('Messung1.csv')

fig, ax = plt.subplots() # Create empty figure with on pair of axes
a.plot_data('Referenz',fig=fig, ax=ax)# Usage of optional parameters
b.plot_data(fig=fig, ax=ax) # If no label is provided, the class's .name attribute will be used

class NewData(BaseData):
    ''' Modified Version of BaseData, only reading Method is changed
    '''
          
    def read_data(self):
        ''' Read data from input file
        
            Returns:
            ----------
            pd.DataFrame
        '''
        data = pd.read_excel(self.filename, index_col = 0)
        data=data.iloc[:,0] # Transform pd.DataFrame to pd.Series
        return data
c = NewData('Messung2.xlsx')
c.plot_data()

class Comparison(object):
    """ Compare results of the Measurements contained in BaseData and derived classes
    
        Parameters
        ----------
        None
        
        Attributes
        ----------
        members : list of BaseData instances (and derived), 
            if you add an result by the add_result method, it is added as an member to this list
            
        members_by_name : dict of BaseData instances
            also added by calling the add_result method, key will be the classes .name
            
        reference : BaseData instance
            a reference measurement (e.g. for scaling)
    """
    def __init__(self):
        self.members = [] # Keep a list of the added results (in contrast to the dictionary below, this will keep the order the data were added)
        self.members_by_name = {} # If you would like to access them by name directly
        self.reference = None
        
    def add_result(self, obj):
        """ Add results of one the measurement the attribute `members`.

        Parameters:
        ----------
        obj : Instance of BaseData (or inherited)

        Returns:
        ----------
        None

        """
        # make sure it is the appropriate format
        assert isinstance(obj, BaseData), 'Obj must be an instance of BaseData, but got {0}'.format(type(obj))
        # don't add two measurements with the same name
        assert obj.name not in list(self.members_by_name.keys()), 'Measurement with name {0} already exists'.format(obj.name)
        self.members.append(obj)
        self.members_by_name[obj.name] = obj
        
    def set_reference(self, obj):
        """ Define an reference value
        
        Parameters
        ----------
        obj : Instance of BaseData(or inherited)
        
        """
        assert isinstance(obj, BaseData), 'Obj must be an instance of BaseData, but got {0}'.format(type(obj))
        self.reference = obj
        
        
    def plot_relative(self, result, fig, ax):
        ''' Plot the measured data relative to the reference data
        
            Parameters:
            ----------
            result : BaseData object
                The result that should be plotted
                
            fig : matplotlib.figure instance
                Figure to plot into, will be created if None
                
            ax : matplotlib.axes instance
                Axes to plot into, will be created if None
        '''
        rel = result.data / self.reference.data
        rel.plot(ax=ax, label = result.name)
        ax.set_ylabel('Relative Leistung zur Referenz')
        ax.legend()

# Daten zur Klasse hinzufügen
all_results = Comparison()
all_results.add_result(a)
# all_results.add_result(a) # Adding this a second time is prevented by the assert statement, uncomment to test
all_results.add_result(b)
all_results.add_result(c)
all_results.set_reference(all_results.members_by_name['Referenz'])
print(all_results.members)

# Alle Ergebnisse in einen Plot
fig, ax = plt.subplots()
for member in all_results.members:
    all_results.plot_relative(member, fig, ax)
    
# Alternativ als Plot nebeneinander
fig, ax = plt.subplots(2, sharex=True)
for i, member in enumerate(all.members[1:]):
    all_results.plot_relative(member, fig, ax[i])
ax[0].set_xlabel('') # Doppeltes Label an der X-Achse entfernen

class AdvancedBaseData(BaseData):
    def __str__(self):
        return 'Class {0}, named {1} from file {2}'.format(type(self), self.name, os.path.abspath(self.filename))
        
d = AdvancedBaseData('Messung1.csv')
print(d)

#for result in all_results:
#    print(result)
    
#all_results[1]

len(all_results)

class AdvancedComparison(Comparison):
    def __len__(self):
        return len(self.members)
    
    def __iter__(self):
        for item in self.members:
            yield(item)
            
    def __getitem__(self, i):
        return self.members[i]
    
ac = AdvancedComparison()
ac.add_result(AdvancedBaseData('Referenz.csv', 'Reference'))
ac.add_result(AdvancedBaseData('Messung1.csv', 'Measurement 1'))
ac.add_result(c)
fig, ax = plt.subplots()
for result in ac:
    result.plot_data(ax=ax)
    
print('Zweites Element im Vergleichsobjekt: ' + str(ac[1]))

len(ac)



