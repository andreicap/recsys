
# coding: utf-8

# # Item Similarity

# In[1]:


import pandas
import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix


import time
import itemSimilarity as Recommenders


# In[2]:


track_playlist_is_model = Recommenders.item_similarity_recommender_py()
track_user_is_model =  Recommenders.item_similarity_recommender_py()


# In[3]:


import time
import pickle
import math
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn import preprocessing

print("succesful import")


# In[4]:


target_playlists = pd.read_csv('datasets/target_playlists.csv', sep='\t')
target_tracks = pd.read_csv('datasets/target_tracks.csv', sep='\t')
tracks_final = pd.read_csv('datasets/tracks_final.csv', sep='\t')
playlists_final = pd.read_csv('datasets/playlists_final.csv', sep='\t')
train_final = pd.read_csv('datasets/train_final.csv', sep='\t')

print('Successfully read data')


# In[5]:


import gc as gc
print(gc.collect())

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


# In[6]:


removed_tracks = tracks_final.query('duration == -1'); 
tracks_final = tracks_final.query('duration != -1');


# In[7]:


combined = train_final.merge(tracks_final, left_on = 'track_id', right_on = 'track_id', how = 'left')


# In[8]:


combined = combined[combined.track_id.isin(target_tracks.track_id)]


# In[9]:


combined.head(5)


# In[10]:


combined = combined[-combined['playcount'].apply(np.isnan)]


# In[11]:


combined.head(5)


# In[12]:


combined['tags'] = [v[1:-1].split(',') for v in combined['tags']]


# In[13]:


playlistList = combined['playlist_id']
itemList = combined['track_id']
playcount = combined['playcount']
tagList = combined['tags']


# In[14]:


playlists_final.head(10)
#add owner to combined datat
combined = combined.merge(playlists_final.drop(playlists_final.columns[[0, 2, 3, 4]], axis=1), left_on = 'playlist_id', right_on = 'playlist_id', how = 'left')


# In[15]:


combined.head(5)


# In[16]:


track_playlist_is_model = Recommenders.item_similarity_recommender_py()
track_user_is_model =  Recommenders.item_similarity_recommender_py()


# In[17]:


track_playlist_is_model.create(combined, 'playlist_id', 'track_id')


# In[18]:


playlists = playlistList.unique()
len(playlists)


# In[19]:


tracksList = itemList
tracks = tracksList.unique()
len(tracks)


# In[20]:


playlist_id = playlists[5]
playlist_items = track_playlist_is_model.get_user_items(playlist_id)
len(playlists)


# In[ ]:


submissions = []
i = 0
recs = []
row = []

for p in range (5000, 10000):
    try:
        pl_id = target_playlists.values[p]
        row = '{},'.format(pl_id[0])
        recs = track_playlist_is_model.recommend(pl_id[0])
        recs = recs.head(5).song

        recs = [int(i) for i in recs]
        for track_id in recs:
            row += ' {}'.format(track_id)
        row += '\n'
        submissions.append(row)
    except:
        pass
    if i%5==0:
        try:
            print(submissions[-1])
        except:
            pass     
        print('done - {}'.format(i))  
    i+=1


# In[22]:


from multiprocessing import Pool

submissions = []
i = 0
recs = []
row = []

def process_submission(playlist):
    try:
        pl_id = playlist
        row = '{},'.format(pl_id[0])
        recs = track_playlist_is_model.recommend(pl_id[0])
        recs = recs.head(5).song

        recs = [int(i) for i in recs]
        for track_id in recs:
            row += ' {}'.format(track_id)
        row += '\n'
        submissions.append(row)
    except:
        pass
    if i%5==0:
        try:
            print(submissions[-1])
        except:
            pass     
        print('done - {}'.format(i))  
    i+=1
    

if __name__ == '__main__':
    pool = Pool(os.cpu_count())                         # Create a multiprocessing Pool
    pool.map(process_submission, target_playlists.values[5000:10000]) 

