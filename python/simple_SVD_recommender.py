import pandas as pd
import numpy as np

num_users = 10
num_items = 5
np.random.seed(42)
def generate_users(num_users, num_items):
    data = []
    for i in range(num_users):
        user = [np.random.randint(2) for _ in range(num_items)]
        data.append(user)
    return data
cols = ["item"+str(i) for i in range(num_items)]
rows = ["user"+str(i) for i in range(num_users)]
user_item_mat = pd.DataFrame(generate_users(num_users,num_items), columns=cols)
user_item_mat.index = rows
user_item_mat         

from scipy.linalg import svd

U, Sigma, VT = svd(user_item_mat)

VT = VT[:3,:]
pd.DataFrame(VT)

pd.DataFrame(VT.T)

U = U[:,:3]
pd.DataFrame(U)

Sigma = Sigma[:3]
pd.DataFrame(np.diag(Sigma))

import pandas as pd
data = [[1,1,1,0,0,0],[1,1,1,0,0,0],[1,1,1,0,0,0],[1,1,1,0,0,0],[0,0,0,1,1,1],[0,0,0,1,1,1],[0,0,0,1,1,1],[0,0,0,1,1,1]]
df = pd.DataFrame(data)

cols = ['Star Wars', 'Star Trek', 'Space Balls', 'Diehard','Lethal Weapon', 'Terminator']
rows = ["user"+str(i) for i in range(8)]
df.index = rows
df.columns = cols
df

from scipy.linalg import svd

U_test, Sigma_test, VT_test = svd(df)
VT_test = VT_test[:2,:]
U_test = U_test[:,:2]

movie_concept_test = pd.DataFrame(VT_test, columns=cols)
movie_concept_test.T

user_concept_test = pd.DataFrame(U_test, index=rows)

user_concept_test

user_item_mat

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib inline')
plt.style.use('seaborn')

fig = plt.figure(figsize=(20,16))
ax = fig.gca(projection='3d')
ax.scatter(U[:,0],U[:,1],U[:,2],c='k',s=150);
ax.set_xlabel("D1", fontsize=20, labelpad=20)
ax.set_ylabel("D2", fontsize=20, labelpad=20)
ax.set_zlabel("D3", fontsize=20, labelpad=20);

lbls = user_item_mat.index
offset = 0.01
for i, txt in enumerate(lbls):
    if i not in [6,7]:
        ax.text(U[i,0]+offset,U[i,1],U[i,2],txt, fontsize=20)
    else:
        ax.text(U[i,0]+offset,U[i,1],U[i,2]+5*offset,txt, fontsize=20)

fig = plt.figure(figsize=(20,16))
ax = fig.gca(projection='3d')
ax.scatter(VT.T[:,0],VT.T[:,1],VT.T[:,2],c='b',s=150, label="Items");
ax.scatter(U[:,0],U[:,1],U[:,2],c='r',s=150, label="Users");
ax.set_xlabel("D1", fontsize=20, labelpad=20)
ax.set_ylabel("D2", fontsize=20, labelpad=20)
ax.set_zlabel("D3", fontsize=20, labelpad=20);

lbls = user_item_mat.columns
item_offset = 0.01
for i, txt in enumerate(lbls):
    if i not in [6,7]:
        ax.text(VT.T[i,0],VT.T[i,1]+item_offset,VT.T[i,2],txt, fontsize=20)
    else:
        ax.text(VT.T[i,0],VT.T[i,1]+item_offset,VT.T[i,2]+5*item_offset,txt, fontsize=20)

lbls = user_item_mat.index
offset = 0.01
for i, txt in enumerate(lbls):
    if i not in [6,7]:
        ax.text(U[i,0],U[i,1]+offset,U[i,2],txt, fontsize=20)
    else:
        ax.text(U[i,0],U[i,1]+offset,U[i,2]+6*offset,txt, fontsize=20)
ax.view_init(30,15)
plt.legend(loc="upper left", fontsize=30);

compare_item = 2
for item in range(num_items):
    if item != compare_item:
        print("Item %s & %s: "%(compare_item,item), np.dot(VT.T[compare_item],VT.T[item]))  

compare_user = 6
for user in range(num_users):
    #if user != compare_user:
        print("User %s & %s: "%(compare_user,user), np.dot(U[compare_user],U[user]))

def get_recommends(itemID, VT, num_recom=2):
    recs = []
    for item in range(VT.T.shape[0]):
        if item != itemID:
            recs.append([item,np.dot(VT.T[itemID],VT.T[item])])
    final_rec = [i[0] for i in sorted(recs,key=lambda x: x[1],reverse=True)]
    return final_rec[:num_recom]

print(get_recommends(2,VT,num_recom=2))

def get_recommends_user(userID, U, df):
    userrecs = []
    for user in range(U.shape[0]):
        if user!= userID:
            userrecs.append([user,np.dot(U[userID],U[user])])
    final_rec = [i[0] for i in sorted(userrecs,key=lambda x: x[1],reverse=True)]
    comp_user = final_rec[0]
    print("User #%s's most similar user is User #%s "% (userID, comp_user))
    rec_likes = df.iloc[comp_user]
    current = df.iloc[userID]
    recs = []
    for i,item in enumerate(current):
        if item != rec_likes[i] and rec_likes[i]!=0:
            recs.append(i)
    return recs

user_to_rec = 3
print("Items for User %s to check out: "% user_to_rec, get_recommends_user(user_to_rec,U,user_item_mat))





