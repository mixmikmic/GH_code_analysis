import pickle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (6,4.5)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rc('pdf', fonttype=42)
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = 'Times'
plt.rcParams['font.family'] = 'serif'

histories = pickle.load(open('../results/model_histories.p', 'rb'))

for i, k in enumerate(list(histories.keys())): 
    plt.plot(list(range(1, len(histories[k]['loss'])+1)), histories[k]['loss'])
    plt.plot(list(range(1, len(histories[k]['loss'])+1)), histories[k]['val_loss'])
    plt.title('Model '+str(i+1)+': Mean squared error by training epoch')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc='upper left')
    plt.show()

for i, k in enumerate(list(histories.keys())): 
    plt.plot(list(range(5, len(histories[k]['loss'][5:])+5)), histories[k]['loss'][5:])
    plt.plot(list(range(5, len(histories[k]['loss'][5:])+5)), histories[k]['val_loss'][5:])
    plt.title('Model '+str(i+1)+': Mean squared error by training epoch')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc='upper left')
    plt.xlim(5, len(histories[k]['loss'])-1)
    plt.savefig('training_graph_'+str(i+1)+'.pdf')
    plt.show()



