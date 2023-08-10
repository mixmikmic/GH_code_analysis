import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os

main_path = '/fhgfs/users/jbehnken/01_Data/04_Models'

models = []
for folder in os.listdir(main_path):
    data = []
    name =  folder.split('_', 1)[-1]
    
    c = 0
    f = 0
    for letter in name:
        if letter=='c': c+=1
        if letter=='f': f+=1
    name_new = str(c)+'c_'+str(f)+'f'
    
    data.append(name_new)
    path = os.path.join(main_path, folder, name+'_Hyperparameter.csv')
    
    df = pd.read_csv(path)
    data.extend(df[df['Auc']==df['Auc'].max()].values.tolist()[0])
    
    modified_time = int(os.stat(path).st_mtime)
    date = time.localtime(modified_time)[0:6]
    data.extend(date)
    models.append(data)
    
df = pd.DataFrame(models, columns=['Name', 'Learning_Rate', 'Batch_Size', 'Patch_Size', 'Depth', 'Hidden_Nodes', 'Accuracy', 'Auc', 'Steps', 'Early_Stopped', 'Time', 'Title', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'])
df = df[['Name', 'Auc', 'Accuracy', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Learning_Rate', 'Batch_Size', 'Patch_Size', 'Depth', 'Hidden_Nodes', 'Steps', 'Early_Stopped', 'Time', 'Title']].sort_values(by='Auc', ascending=False)


df

df = pd.read_csv('/fhgfs/users/jbehnken/01_Data/04_Models/13_cccfff/cccfff_Hyperparameter.csv')
df = df[df['Title']=='Starting_Test']
print(df['Accuracy'].value_counts())

color_wheel = {1: 'r', 
               2: 'g',}
colors = df['Early_Stopped'].map(lambda x: color_wheel.get(x + 1))

plt.style.use('ggplot')
pd.plotting.scatter_matrix(df[['Batch_Size', 'Patch_Size', 'Depth', 'Hidden_Nodes']], color=colors, diagonal='kde', alpha=1, figsize=(12,12))
plt.tight_layout()
plt.show()


for column in ['Depth', 'Batch_Size', 'Patch_Size', 'Hidden_Nodes']:
    min_depth = df[df['Early_Stopped']==True][column].min()
    max_depth = df[df['Early_Stopped']==True][column].max()
    bins = max_depth - min_depth + 1

    df[df['Early_Stopped']==True][column].hist(bins=bins)
    plt.title(column)
    plt.show()



liste = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)


for i in range(100):
    if liste[i][0]<=60:
        scheibe = liste[i][0]
        break
        
# Cuts 7 frames for the final image
lower = scheibe-33
upper =  scheibe-26

print(lower, upper)
scheibe-30

path = '/fhgfs/users/jbehnken/01_Data/04_Models/16_pre-cccfff/pre-cccfff_Hyperparameter.csv'
df = pd.read_csv(path)
df.tail()





