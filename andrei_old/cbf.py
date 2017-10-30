import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors


pd.set_option('display.float_format', lambda x: '%.3f' % x)

playlists_final = pd.read_table("playlists_final.csv", index_col=False, header=0);
target_tracks = pd.read_table("target_tracks.csv", index_col=False, header=0);
tracks_final = pd.read_table("tracks_final.csv", index_col=False, header=0);
train_final = pd.read_table("train_final.csv", index_col=False, header=0);
target_playlists = pd.read_table("target_playlists.csv", index_col=False, header=0);

#remove metadataless
removed_tracks = tracks_final.query('duration == -1'); 
tracks_final = tracks_final.query('duration != -1');


# print(tracks_final.playcount.describe())

# count    97211.000
# mean      2401.901
# std       6741.930
# min          0.000
# 25%         77.000
# 50%        505.000
# 75%       2096.000
# max     367595.000
# Name: playcount, dtype: float64

# print(tracks_final.playcount.quantile(np.arange(.5, 1, .1)))

# 0.900    5903.000
# 0.910    6478.000
# 0.920    7148.200
# 0.930    7976.000
# 0.940    9042.800
# 0.950   10392.500
# 0.960   12167.600
# 0.970   14678.000
# 0.980   18638.800
# 0.990   28037.400
# Name: playcount, dtype: float64

# 0.500    505.000
# 0.600    904.000
# 0.700   1584.000
# 0.800   2834.000
# 0.900   5903.000
# Name: playcount, dtype: float64

# remove lower songs from trainfinal

train_final = (train_final[~train_final.track_id.isin(removed_tracks.track_id)])

# remove zeroplayed songs
tracks_null_playcount = tracks_final.query('playcount == 0')
train_final = (train_final[~train_final.track_id.isin(tracks_null_playcount.track_id)])


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


combined = train_final.merge(tracks_final, left_on = 'track_id', right_on = 'track_id', how = 'left')
# wide_artist_data = usa_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
# wide_artist_data_sparse = csr_matrix(wide_artist_data.values)

#filter only target tracks
combined = combined[combined.track_id.isin(target_tracks.track_id)]

#had to use only top 10% :(
#combined cleaning, only popular songs
unpopular_tracks =  combined.query('playcount < 20000')
combined = combined[~combined.track_id.isin(unpopular_tracks.track_id)]
combined['tags'] = [v[1:-1].split() for v in combined['tags'].values]
# combined.shape - Out[135]: (294900, 7)
print(combined.shape)


# wide_data = combined.pivot(index = 'playlist_id', columns = 'track_id', values = 'playcount').fillna(0)

# wide_data_sparse = csr_matrix(wide_data.values)

# save_sparse_csr('wide_data_sparse.npz', wide_data_sparse)


# model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
# model_knn.fit(wide_data_sparse)

# query_index = np.random.choice(wide_data.shape[0])
# print(query_index)


# distances, indices = model_knn.kneighbors(wide_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

# for i in range(0, len(distances.flatten())):
#     if i == 0:
#         print('Recommendations for {0}:\n'.format(wide_data.index[query_index]))
#     else:
#         print('{0}: {1}, with distance of {2}:'.format(i, wide_data.index[indices.flatten()[i]], distances.flatten()[i]))

import scipy.sparse as sps

URM_all = sps.coo_matrix((combined['playcount'], (combined['playlist_id'], combined['track_id'])))


userList = combined['playlist_id']
itemList = combined['track_id']
ratingList = combined['playcount']
tagList = combined['tags']

userList_icm = list(userList)
itemList_icm = list(itemList)
tagList_icm = list(tagList)

userList_unique = list(set(userList_icm))
itemList_unique = list(set(itemList_icm))

flattened_list = []

#flatten the lis
for x in tagList_icm:
    for y in x:
        flattened_list.append(y)

tagList_unique = list(set(flattened_list))
# tagList_unique = [ for v in combined['tags'].values]

numUsers = len(userList_unique)
numItems = len(itemList_unique)
numTags = len(tagList_unique)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(combined['tags'])

tagList_icm = le.transform(combined['tags'])

print(tagList_icm[0:10])

ones = np.ones(len(tagList_icm))
ICM_all = sps.coo_matrix((ones, (itemList_icm, tagList_icm)))
ICM_all = ICM_all.tocsr()
ICM_all

missing_items = np.zeros((3, numTags))
missing_items = sps.csr_matrix(missing_items)
ICM_all = sps.vstack((ICM_all, missing_items))
ICM_all

features_per_item = (ICM_all > 0).sum(axis=1)
items_per_feature = (ICM_all > 0).sum(axis=0)

print(features_per_item.shape)
print(items_per_feature.shape)

features_per_item = np.sort(features_per_item)
items_per_feature = np.sort(items_per_feature)

train_test_split = 0.80

numInteractions = URM_all.nnz


train_mask = np.random.choice([True,False], numInteractions, [train_test_split, 1-train_test_split])

userList = np.array(userList)
itemList = np.array(itemList)
ratingList = np.array(ratingList)


URM_train = sps.coo_matrix((ratingList[train_mask], (userList[train_mask], itemList[train_mask])))
URM_train = URM_train.tocsr()

test_mask = np.logical_not(train_mask)

URM_test = sps.coo_matrix((ratingList[test_mask], (userList[test_mask], itemList[test_mask])))
URM_test = URM_test.tocsr()

def precision(recommended_items, relevant_items):
    
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    
    return precision_score

def recall(recommended_items, relevant_items):
    
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    
    return recall_score

def MAP(recommended_items, relevant_items):
   
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score

def evaluate_algorithm(URM_test, recommender_object, at=5):
    
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    
    num_eval = 0


    for i,user_id in  enumerate(userList_unique):
        
        if i % 500 == 0:
            print("User %d of %d" % (i, len(userList_unique)))

        relevant_items = URM_test[user_id].indices
        
        if len(relevant_items)>0:
            
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval+=1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)


    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval
    
    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP));


class BasicItemKNNRecommender(object):
    """ ItemKNN recommender with cosine similarity and no shrinkage"""

    def __init__(self, URM, k=50, shrinkage=100, similarity='cosine'):
        self.dataset = URM
        self.k = k
        self.shrinkage = shrinkage
        self.similarity_name = similarity
        if similarity == 'cosine':
            self.distance = Cosine(shrinkage=self.shrinkage)
        elif similarity == 'pearson':
            self.distance = Pearson(shrinkage=self.shrinkage)
        elif similarity == 'adj-cosine':
            self.distance = AdjustedCosine(shrinkage=self.shrinkage)
        else:
            raise NotImplementedError('Distance {} not implemented'.format(similarity))

    def __str__(self):
        return "ItemKNN(similarity={},k={},shrinkage={})".format(
            self.similarity_name, self.k, self.shrinkage)

    def fit(self, X):
        item_weights = self.distance.compute(X)
        
        item_weights = check_matrix(item_weights, 'csr') # nearly 10 times faster
        print("Converted to csr")
        
        # for each column, keep only the top-k scored items
        # THIS IS THE SLOW PART, FIND A BETTER SOLUTION        
        values, rows, cols = [], [], []
        nitems = self.dataset.shape[1]
        for i in range(nitems):
            if (i % 1000 == 0):
                print("Item %d of %d" % (i, nitems))
                
            this_item_weights = item_weights[i,:].toarray()[0]
            top_k_idx = np.argsort(this_item_weights) [-self.k:]-5
            
            values.extend(this_item_weights[top_k_idx])
            rows.extend(np.arange(nitems)[top_k_idx])
            cols.extend(np.ones(self.k) * i)
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.dataset[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        # rank items
        ranking = scores.argsort()[::-1]
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
            
        return ranking[:at]
    
    def _filter_seen(self, user_id, ranking):
        user_profile = self.dataset[user_id]
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


import scipy
class ISimilarity(object):
    """Abstract interface for the similarity metrics"""

    def __init__(self, shrinkage=10):
        self.shrinkage = shrinkage

    def compute(self, X):
        pass


class Cosine(ISimilarity):
    def compute(self, X):
        # convert to csc matrix for faster column-wise operations
        X = check_matrix(X, 'csc', dtype=np.float32)

        # 1) normalize the columns in X
        # compute the column-wise norm
        # NOTE: this is slightly inefficient. We must copy X to compute the column norms.
        # A faster solution is to  normalize the matrix inplace with a Cython function.
        Xsq = X.copy()
        Xsq.data **= 2
        norm = np.sqrt(Xsq.sum(axis=0))
        norm = np.asarray(norm).ravel()
        norm += 1e-6
        # compute the number of non-zeros in each column
        # NOTE: this works only if X is instance of sparse.csc_matrix
        col_nnz = np.diff(X.indptr)
        # then normalize the values in each column
        X.data /= np.repeat(norm, col_nnz)
        print("Normalized")

        # 2) compute the cosine similarity using the dot-product
        dist = X * X.T
        print("Computed")
        
        # zero out diagonal values
        dist = dist - sps.dia_matrix((dist.diagonal()[scipy.newaxis, :], [0]), shape=dist.shape)
        print("Removed diagonal")
        
        # and apply the shrinkage
        if self.shrinkage > 0:
            dist = self.apply_shrinkage(X, dist)
            print("Applied shrinkage")    
        
        return dist

    def apply_shrinkage(self, X, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind * X_ind.T
        # remove the diagonal
        co_counts = co_counts - sps.dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        co_counts_shrink = co_counts.copy()
        co_counts_shrink.data += self.shrinkage
        co_counts.data /= co_counts_shrink.data
        dist.data *= co_counts.data
        return dist

rec = BasicItemKNNRecommender(URM=URM_train, shrinkage=0, k=50)
rec.fit(ICM_all)
#    playlist_id  track_id  artist_id  duration  playcount     album  \
# 0      3271849   2801526     325531    157000   1086.000  [149604]   

#                                      tags  
# 0  [3982, 170251, 189631, 237215, 237214] 
















