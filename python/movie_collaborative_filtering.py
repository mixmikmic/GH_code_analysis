import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

movies = pd.read_table('ml-20m/movies.csv', sep=',',names = ['movieId',"Title","genres"], skiprows=1)

movies

u, m, r, junk = np.loadtxt('ml-20m/ratings_small.csv', delimiter=',', skiprows=1).T
mat = coo_matrix((r, (u-1, m-1)), shape=(u.max(), m.max()))

print(mat.toarray()[0][28])
print(mat.shape)

a = np.zeros((100,100))
a[5,5] = 1
pd.DataFrame(a)

from scipy.sparse.linalg import svds
U, s, VT = svds(mat,k=100)

def find_similar_movies(itemID, VT, num_recom=2):
    recs = []
    for item in range(VT.T.shape[0]):
        if item != itemID:
            recs.append([item+1,np.dot(VT.T[itemID-1],VT.T[item])])
    final_rec = [(i[0],i[1]) for i in sorted(recs,key=lambda x: x[1],reverse=True)]
    return final_rec[:num_recom]

movies[movies.Title == "Toy Story 2 (1999)"]

MOVIE_ID = 3114
for item in find_similar_movies(MOVIE_ID,VT,num_recom=10):
    print(item)
    print(movies[movies['movieId'] == int(item[0])].Title,'\n')

def get_recommends_for_user(userID, U, VT, num_recom=2):
    recs = []
    for item in range(VT.T.shape[0]):
        recs.append([item+1,np.dot(U[userID-1],VT.T[item])])
    final_rec = [(i[0],i[1]) for i in sorted(recs,key=lambda x: x[1],reverse=True)]
    return final_rec[:num_recom]

USERID=25
for item in get_recommends_for_user(USERID,U,VT,10):
    print(item)
    print("Actual Rating: ", mat.toarray()[USERID][item[0]])
    print(movies[movies.movieId == int(item[0])].Title,'\n')

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
print("Items for User %s to check out: "% user_to_rec, get_recommends_user(user_to_rec,U,pd.DataFrame(mat.toarray())))

import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from scipy.sparse.linalg import svds

class svdRec():
    def __init__(self):
        self.U, self.s, self.V = (None,None,None)
        self.user_encoder = None
        self.item_encoder = None
        self.mat = None
        self.decomp = False
    
    def load_csv_sparse(self,filename,delimiter=',',skiprows=None, is_one_indexed=True):
        print("Note: load_csv_sparse expects a csv in the format of: rowID, colID, Value, ...")
        u, m, r = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows, usecols=(0,1,2)).T
        if is_one_indexed:
            self.mat = coo_matrix((r, (u-1, m-1)), shape=(u.max(), m.max())).tocsr()
        else:
            self.mat = coo_matrix((r, (u, m)), shape=(u.max(), m.max())).tocsr()
        print("Created matrix of shape: ",self.mat.shape)
        
    def load_data_numpy(self, array, data_type=float):
        self.mat = csr_matrix(array,dtype=data_type)
        print("Created matrix of shape: ",self.mat.shape)
        
    def load_item_encoder(self, d):
        if type(d) != dict:
            raise TypeError("Encoder must be dictionary with key = itemID and value = Title")
        self.item_encoder = d
        
    def load_user_encoder(self, d):
        if type(d) != dict:
            raise TypeError("Encoder must be dictionary with key = userID and value = Title")
        self.user_encoder = d
        
    def get_item_name(self,itemid):
        if self.item_encoder:
            return self.item_encoder[str(itemid)]
        else:
            return "No ItemId -> Item-name Encoder Built!"
    
    def get_user_name(self,userid):
        if self.item_encoder:
            return self.user_encoder[str(userid)]
        else:
            return "No UserID -> Username Encoder Built!"
    
    def SVD(self, num_dim=None):
        if num_dim==None:
            print("Number of SVD dimensions not requested, using %s dimensions." % (min(self.mat.shape)-1), "To set, use num_dim.")
            num_dim = min(self.mat.shape)-1
        self.U, self.s, self.VT = svds(self.mat,k=num_dim)
        self.decomp = True
    
    def get_cell(self,i,j):
        return self.mat[1,:].toarray()[0,j]
    
    def get_similar_items(self, itemID, num_recom=5, show_similarity=False):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        recs = []
        for item in range(self.VT.T.shape[0]):
                recs.append([item+1,self.item_similarity(itemID-1,item)])
        if show_similarity:
            final_rec = [(i[0],i[1]) for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        else:
            final_rec = [i[0] for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        return final_rec[:num_recom]
    
    def item_similarity(self,item1,item2):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.VT.T[item1],self.VT.T[item2])
    
    def user_similarity(self,user1,user2):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.U[user1],self.U[user2])
    
    def user_item_similarity(self,user,item):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.U[user],self.VT.T[item])
    
    def user_item_predict(self,user,item):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        return np.dot(self.U[user],self.VT.T[item])
        
    def recommends_for_user(self, userID, num_recom=2, show_similarity=False):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        recs = []
        for item in range(self.VT.T.shape[0]):
            recs.append((item+1,self.user_item_predict(userID-1,item)))
        if show_similarity:
            final_rec = [(i[0],i[1]) for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        else:
            final_rec = [i[0] for i in sorted(recs,key=lambda x: x[1],reverse=True)]
        return final_rec[:num_recom]
    
    def recs_from_closest_user(self, userID):
        if not self.decomp:
            raise ValueError("Must run SVD() before making recommendations!")
        userrecs = []
        for user in range(self.U.shape[0]):
            if user!= userID:
                userrecs.append([user,self.user_similarity(userID,user)])
        final_rec = [i[0] for i in sorted(userrecs,key=lambda x: x[1],reverse=True)]
        comp_user = final_rec[0]
        print("User #%s's most similar user is User #%s "% (userID, comp_user))
        data = self.mat.toarray()
        rec_likes = data[comp_user]
        current = data[userID]
        recs = []
        for i,item in enumerate(current):
            if item != rec_likes[i] and rec_likes[i]!=0:
                recs.append(i)
        return recs
        

svd = svdRec()
svd.load_csv_sparse('ml-20m/ratings_small.csv', delimiter=',', skiprows=1)
svd.SVD(num_dim=100)

movies = pd.read_table('ml-20m/movies.csv', sep=',',names = ['movieId',"Title","genres"])
movie_dict = {}
for i, row in movies.iterrows():
    movie_dict.update({row['movieId']: row['Title']})
svd.load_item_encoder(movie_dict)
print(svd.item_encoder['3114'])
print(svd.get_cell(1,2))

MOVIE_ID = 3114
for item in svd.get_similar_items(MOVIE_ID,show_similarity=True):
    print(item)
    print(svd.get_item_name(item[0]),'\n')

USERID=25
for item in svd.recommends_for_user(USERID, num_recom=5, show_similarity=False):
    print("ID: ", item)
    print("Actual Rating: ", svd.mat.toarray()[USERID][item])
    print("Title: ",svd.get_item_name(item),'\n')

user_to_rec = 3
print("Items for User %s to check out: "% user_to_rec, svd.recs_from_closest_user(user_to_rec))

print(svd.user_similarity(0,2)) # UserID 1 and UserID 3 (the userIDs are shifted by one in this dataset)
print(svd.item_similarity(0,3113)) # Toy Story and Toy Story 2
print(svd.user_item_similarity(0,0)) # UserID 1 and Toy Story



