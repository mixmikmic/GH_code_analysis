import pandas as pd
import numpy as np
from IPython.display import display

import plotly
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

csv_input = input()

results = pd.read_csv(csv_input, low_memory=False)
results = results.set_index("index")
print("Total number of entries: ", len(results))
results.head(10)

directories = results['directory'].unique()

#Let us take any 'english' dataset

analyze_results = results[results['directory'].str.contains('english')]
analyze_results = analyze_results[analyze_results['RESAMPLING']==True]
print(len(analyze_results))

analyze_results['fscore_avg'] = (analyze_results['fscore_dup'] + analyze_results['fscore_nov'])/2.0
analyze_results['precision_avg'] = (analyze_results['precision_dup'] + analyze_results['precision_nov'])/2.0
analyze_results['recall_avg'] = (analyze_results['recall_dup'] + analyze_results['recall_nov'])/2.0

print("Overall best scores:")
print("")
print('Best average F1 score: ', max(analyze_results['fscore_avg']))
print('Best average precision score: ',max(analyze_results['precision_avg']))
print('Best average recall score: ',max(analyze_results['recall_avg']))
print("")
print('Best duplicate F1 score: ', max(analyze_results['fscore_dup']))
print('Best duplicate precision score: ',max(analyze_results['precision_dup']))
print('Best duplicate recall score: ',max(analyze_results['recall_dup']))
print("")
print('Best novel F1 score: ', max(analyze_results['fscore_nov']))
print('Best novel precision score: ',max(analyze_results['precision_nov']))
print('Best novel recall score: ',max(analyze_results['recall_nov']))

# What is the spread of the variables we are most interested in?
analyze_results[['fscore_avg','precision_avg','recall_avg','fscore_dup','precision_dup','recall_dup','fscore_nov',
                'precision_nov','recall_nov']].describe()

# First get the top results and look to see what their performance is accross the board and which algorithm was used
top_results = analyze_results.sort_values(by='fscore_avg', ascending=False).head(10)
top_results[['algorithms','fscore_avg','precision_avg','recall_avg','fscore_dup','precision_dup','recall_dup',
             'fscore_nov','precision_nov','recall_nov', 'run_time']]

# It looks like XGB is doing the best - we can see how different each algorithm performs...
grouped_data = analyze_results.groupby('algorithms')[['fscore_avg','precision_avg','recall_avg','fscore_dup','precision_dup','recall_dup',
             'fscore_nov','precision_nov','recall_nov']]

grouped_data.mean()

grouped_data.count()

grouped_data.std()

def box_plot_results(column):
    import scipy.stats as s
    non_grouped = analyze_results[['algorithms','fscore_avg','precision_avg','recall_avg','fscore_dup',
                                   'precision_dup','recall_dup', 'fscore_nov','precision_nov','recall_nov']]
    log_reg_data = non_grouped[non_grouped['algorithms']=='LOG_REG'][column].values
    xgb_data = non_grouped[non_grouped['algorithms']=='XGB'][column].values
    
    log_reg = go.Box(y=log_reg_data, name = 'Logistic Regression')
    xgb = go.Box(y=xgb_data, name = 'XGB')
    data = [xgb, log_reg]
    layout = go.Layout(title='Average F1 Score by Algorithm' )
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
    
    #print(s.kruskal(xgb_data, log_reg_data))

box_plot_results('fscore_avg')

# Let us also plot the fscore as it was closer in our summary statistic 
# You may want to try out the zoom feature as thre are a lot of outliers for this.  
# Logistic regression sometimes failed to find any novel examples
box_plot_results('fscore_dup')

#'MEM_NET',
toggle_columns = ['BOW_APPEND','BOW_COS','BOW_DIFFERENCE','BOW_PRODUCT','BOW_TFIDF','CNN_APPEND','CNN_COS','CNN_DIFFERENCE',
'CNN_PRODUCT','LDA_APPEND','LDA_COS','LDA_DIFFERENCE','LDA_PRODUCT','MEM_NET',
 'ST_APPEND','ST_COS','ST_DIFFERENCE','ST_PRODUCT','W2V_ABS','W2V_APPEND',
'W2V_AVG','W2V_COS','W2V_DIFFERENCE','W2V_PRETRAINED','W2V_PRODUCT',
 'RESAMPLING','OVERSAMPLING','REPLACEMENT','STEM']

top_results = analyze_results.sort_values(by='fscore_avg', ascending=False).head(10)
toggle_counts =[]
for c in toggle_columns:
    #We want to get counts on the totaly number of times the toggle was true, how many times that was in the top 10
    #If the spreadsheet did not have NaN values then a simpler function could be used...
    num_true = len(analyze_results[analyze_results[c]==True][c])
    num_top_true = len(top_results[top_results[c]==True][c])
    avg_f_score = np.mean(analyze_results[analyze_results[c]==True]['fscore_avg'])
    std_f_score = np.std(analyze_results[analyze_results[c]==True]['fscore_avg'])
    c_res = {
        'name': c,
        'Times_True' : num_true,
        'Times_True_Top10' : num_top_true,
        'PCT_Top10' : num_top_true/10.*100,
        'fscore_avg': avg_f_score,
        'fscore_std': std_f_score
    }
    toggle_counts.append(c_res)
    
toggle_counts = pd.DataFrame(toggle_counts)
toggle_counts.set_index('name')
toggle_counts.sort_values(by='Times_True_Top10', ascending=False)

def option_plots(options, column):
    eval_columns = ['fscore_avg','precision_avg','recall_avg','fscore_dup',
                        'precision_dup','recall_dup', 'fscore_nov','precision_nov','recall_nov']
    
    non_grouped = analyze_results[eval_columns+options]
    
    boxes = []
    for option in options:
        # Mostly true false except for MEM_MASK_MODE...
        if option!='MEM_MASK_MODE':
            boxT = go.Box(y=non_grouped[non_grouped[option]==True][column].values,
                        name = "T_"+option, marker=dict(color='#2D9C6E'))
            boxF = go.Box(y=non_grouped[non_grouped[option]==False][column].values,
                name = "F_"+option, marker=dict(color='#AE37DB'))
            boxes.append(boxT)
            boxes.append(boxF)
        else:
            box1 = go.Box(y=non_grouped[non_grouped[option]=='skip_thought'][column].values,
                name = "ST_"+option, marker=dict(color='#37DB97'))
            box2 = go.Box(y=non_grouped[non_grouped[option]=='word2vec'][column].values,
                name = "W2V_"+option, marker=dict(color='#AE37DB'))
            boxes.append(box1)
            boxes.append(box2)

    data = [b for b in boxes]
    layout = go.Layout(title='Average %s' %column)
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

bow_options = ['BOW_APPEND','BOW_COS','BOW_DIFFERENCE','BOW_PRODUCT','BOW_TFIDF']
lda_options = ['LDA_APPEND','LDA_COS','LDA_DIFFERENCE','LDA_PRODUCT']
param_options = ['RESAMPLING','OVERSAMPLING','REPLACEMENT', 'STEM']
w2v_options = ['W2V_ABS','W2V_APPEND','W2V_AVG','W2V_COS','W2V_MAX', 'W2V_MIN', 'W2V_DIFFERENCE',
               'W2V_PRETRAINED','W2V_PRODUCT']
st_options = ['ST_APPEND','ST_COS','ST_DIFFERENCE','ST_PRODUCT']
#cnn_options = ['CNN_APPEND','CNN_COS','CNN_DIFFERENCE','CNN_PRODUCT']
#mem_options = ['MEM_NET','MEM_MASK_MODE']
option_plots(param_options, 'fscore_avg')

toggle_types = ['BOW', 'LDA', 'W2V', 'ST']
def meta_option_plots(column):
    eval_columns = ['fscore_avg','precision_avg','recall_avg','fscore_dup',
                        'precision_dup','recall_dup', 'fscore_nov','precision_nov','recall_nov', 'run_time']
    
    non_grouped = analyze_results
    
    boxes = []
        
    bow_y_true = analyze_results[analyze_results[['BOW_APPEND','BOW_COS','BOW_DIFFERENCE','BOW_PRODUCT','BOW_TFIDF']]
                    .isin([True]).any(axis=1)]    
    bowT = go.Box(y=bow_y_true[column].values, name = 'Any BOW', marker=dict(color='#2D9C6E'))
    bow_y_false = analyze_results[analyze_results[['BOW_APPEND','BOW_COS','BOW_DIFFERENCE','BOW_PRODUCT','BOW_TFIDF']]
                    .isin([False]).all(axis=1)]    
    bowF = go.Box(y=bow_y_false[column].values, name = 'W/O BOW', marker=dict(color='#AE37DB'))    
    boxes.append(bowT)
    boxes.append(bowF)
    
    lda_y_true = analyze_results[analyze_results[['LDA_APPEND','LDA_COS','LDA_DIFFERENCE','LDA_PRODUCT']]
                    .isin([True]).any(axis=1)]    
    ldaT = go.Box(y=lda_y_true[column].values, name = 'Any LDA', marker=dict(color='#2D9C6E'))
    lda_y_false = analyze_results[analyze_results[['LDA_APPEND','LDA_COS','LDA_DIFFERENCE','LDA_PRODUCT']]
                    .isin([False]).all(axis=1)]    
    ldaF = go.Box(y=lda_y_false[column].values, name = 'W/O LDA', marker=dict(color='#AE37DB'))    
    boxes.append(ldaT)
    boxes.append(ldaF)
    
    w2v_y_true = analyze_results[analyze_results[['W2V_APPEND','W2V_COS','W2V_DIFFERENCE','W2V_PRODUCT']]
                    .isin([True]).any(axis=1)]    
    w2vT = go.Box(y=w2v_y_true[column].values, name = 'Any W2V', marker=dict(color='#2D9C6E'))
    w2v_y_false = analyze_results[analyze_results[['W2V_APPEND','W2V_COS','W2V_DIFFERENCE','W2V_PRODUCT']]
                    .isin([False]).all(axis=1)]    
    w2vF = go.Box(y=w2v_y_false[column].values, name = 'W/O W2V', marker=dict(color='#AE37DB'))    
    boxes.append(w2vT)
    boxes.append(w2vF)

    st_y_true = analyze_results[analyze_results[['ST_APPEND','ST_COS','ST_DIFFERENCE','ST_PRODUCT']]
                    .isin([True]).any(axis=1)]    
    stT = go.Box(y=st_y_true[column].values, name = 'Any Skip_Thought', marker=dict(color='#2D9C6E'))
    st_y_false = analyze_results[analyze_results[['ST_APPEND','ST_COS','ST_DIFFERENCE','ST_PRODUCT']]
                    .isin([False]).all(axis=1)]    
    stF = go.Box(y=st_y_false[column].values, name = 'W/O Skip_Thought', marker=dict(color='#AE37DB'))    
    boxes.append(stT)
    boxes.append(stF)
    
    data = [b for b in boxes]
    layout = go.Layout(title='Average %s For All High Level Features' %column)
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

meta_option_plots('fscore_avg')

toggle_types = ['BOW', 'LDA', 'W2V', 'ST']
def meta_option_plots(column):
    eval_columns = ['fscore_avg','precision_avg','recall_avg','fscore_dup',
                        'precision_dup','recall_dup', 'fscore_nov','precision_nov','recall_nov', 'run_time']
    
    non_grouped = analyze_results
    
    boxes = []
        
    bow_y_true = analyze_results[analyze_results[['BOW_APPEND','BOW_COS','BOW_DIFFERENCE','BOW_PRODUCT','BOW_TFIDF']]
                    .isin([True]).any(axis=1)]    
    bowT = go.Box(y=bow_y_true[column].values, name = 'BOW')
 
    boxes.append(bowT)
    #boxes.append(bowF)
    
    lda_y_true = analyze_results[analyze_results[['LDA_APPEND','LDA_COS','LDA_DIFFERENCE','LDA_PRODUCT']]
                    .isin([True]).any(axis=1)]    
    ldaT = go.Box(y=lda_y_true[column].values, name = 'LDA', )
  
    boxes.append(ldaT)
    #boxes.append(ldaF)
    


    st_y_true = analyze_results[analyze_results[['ST_APPEND','ST_COS','ST_DIFFERENCE','ST_PRODUCT']]
                    .isin([True]).any(axis=1)]    
    stT = go.Box(y=st_y_true[column].values, name = 'Skip_Thought')

    boxes.append(stT)
    #boxes.append(stF)
    #Here only append, difference, product, and cos are used as avg, max, min, abs are decorators on these
    w2v_y_true = analyze_results[analyze_results[['W2V_APPEND','W2V_COS','W2V_DIFFERENCE','W2V_PRODUCT']]
                    .isin([True]).any(axis=1)]    
    w2vT = go.Box(y=w2v_y_true[column].values, name = 'W2V')
  
    boxes.append(w2vT)
    #boxes.append(w2vF)
    
    data = [b for b in boxes]
    layout = go.Layout(title='Average F1 Score For All High Level Features')
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

meta_option_plots('fscore_avg')

toggles = analyze_results[['BOW_APPEND','BOW_COS','BOW_DIFFERENCE','BOW_PRODUCT','BOW_TFIDF','CNN_APPEND','CNN_COS',
'CNN_DIFFERENCE','CNN_PRODUCT','LDA_APPEND','LDA_COS','LDA_DIFFERENCE','LDA_PRODUCT',
 'ST_APPEND','ST_COS','ST_DIFFERENCE','ST_PRODUCT','W2V_ABS','W2V_APPEND', 'W2V_MAX', 'W2V_MIN',
'W2V_AVG','W2V_COS','W2V_DIFFERENCE','W2V_PRODUCT']]

t_count = np.sum(toggles, axis=1)
analyze_results['num_features'] = t_count

plot_data_ = go.Scatter(x=analyze_results['num_features'], 
                    y=analyze_results['fscore_avg'],
                    mode = 'markers', 
                    )

plot_data = [plot_data_]

plotly.offline.iplot(plot_data)

top_results = analyze_results.sort_values(by='fscore_avg', ascending=False).head(10)
top_results['num_features'].values

meta_option_plots('run_time')

#Are the top results slower/faster than the other results?
top_results['run_time'].values

print("Top 10 avg time: ", np.mean(top_results['run_time'].values))
print("All Results avg time: ", np.mean(analyze_results['run_time'].values))



