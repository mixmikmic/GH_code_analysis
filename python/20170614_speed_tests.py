import netcomp as nc

# Import things we'll use to analyze the timing
from sklearn.linear_model import LinearRegression
get_ipython().magic('load_ext line_profiler')

def complexity_plot(log_range,times,label=None):
    "Make a logarithmic complexity plot, report best-fit line slope."
    logtime = np.log(times).reshape(-1,1)
    logn = np.log(log_range).reshape(-1,1)
    regr = LinearRegression()
    regr.fit(logn,logtime)

    slope = float(regr.coef_)
    fit_line = np.exp(regr.predict(logn))
    
    plt.figure();

    plt.loglog(log_range,times,'o');
    plt.loglog(log_range,fit_line,'--');
    plt.xlabel('Size of Problem');
    plt.ylabel('Time Elapsed');
    if label is not None:
        plt.title('Complexity of ' + label)
        print('Best fit line for {} has slope {:0.03f}.'.format(label,slope))
    else:
        print('Best fit line has slope {:.03f}.'.format(slope))

data_dict = pickle.load(open('graph_distance_timing.p','rb'))

print(data_dict['description'])

df = data_dict['results_df']

df

labels = df.columns.unique()

df_dict = {}

df['Edit'].T

n = 100
for label in labels:
    df_temp = df[label].T
    df_temp.index = range(100)
    df_dict[label] = df_temp

df_total = pd.concat(df_dict,axis=1)

df_total

ran = [10,30,100,300,1000,3000]
for label in labels:
    complexity_plot(ran[2:],np.array(df_total[label].median())[2:],label=label)
    plt.title('Complexity of ' + label)



