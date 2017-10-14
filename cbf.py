import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from scipy.sparse import csr_matrix

pd.set_option('display.float_format', lambda x: '%.3f' % x)

playlists_final = pd.read_table("playlists_final.csv", index_col=False, header=0);
target_tracks = pd.read_table("target_tracks.csv", index_col=False, header=0);
tracks_final = pd.read_table("tracks_final.csv", index_col=False, header=0);
train_final = pd.read_table("train_final.csv", index_col=False, header=0);
target_playlists = pd.read_table("target_playlists.csv", index_col=False, header=0);

#remove metadataless
removed_tracks = tracks_final.query('duration == -1'); 
tracks_final = tracks_final.query('duration != -1');


print(tracks_final.playcount.describe())

# count    97211.000
# mean      2401.901
# std       6741.930
# min          0.000
# 25%         77.000
# 50%        505.000
# 75%       2096.000
# max     367595.000
# Name: playcount, dtype: float64

print(tracks_final.playcount.quantile(np.arange(.5, 1, .1)))

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

#had to use only top 10% :(
#combined cleaning, only popular songs
unpopular_tracks =  combined.query('playcount < 1584')
combined = combined[~combined.track_id.isin(unpopular_tracks.track_id)]

# combined.shape - Out[135]: (294900, 7)


wide_data = combined.pivot(index = 'track_id', columns = 'playlist_id', values = 'playcount').fillna(0)

wide_data_sparse = csr_matrix(wide_data.values)

save_sparse_csr('~/comp/wide_data_sparse.npz', wide_data_sparse)

from sklearn.neighbors import NearestNeighbors


model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(wide_data_sparse)

query_index = np.random.choice(wide_data.shape[0])
print(query_index)


distances, indices = model_knn.kneighbors(wide_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(wide_data.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, wide_data.index[indices.flatten()[i]], distances.flatten()[i]))













































