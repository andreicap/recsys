
# coding: utf-8

# In[7]:


import time
import pickle
import math
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn import preprocessing


# In[10]:


SUBMISSION = False
SUBMISSION_FILENAME = 'submission.csv'
TEST_FILENAME = 'test.csv'


# In[12]:


target_playlists = pd.read_csv('datasets/target_playlists.csv', sep='\t')
target_tracks = pd.read_csv('datasets/target_tracks.csv', sep='\t')
tracks_final = pd.read_csv('datasets/tracks_final.csv', sep='\t')
playlists_final = pd.read_csv('datasets/playlists_final.csv', sep='\t')
train_final = pd.read_csv('datasets/train_final.csv', sep='\t')

print('Successfully read data')


# In[13]:


print('Data info:')
print('Unique tracks count: {}'.format(tracks_final['track_id'].nunique()))
print('Unique playlist count: {}'.format(playlists_final['playlist_id'].nunique()))
print('Target tracks count: {}'.format(target_tracks['track_id'].nunique()))
print('Target playlists count: {}'.format(target_playlists['playlist_id'].nunique()))


# In[ ]:


def str_tags_to_list(str_tags):
    if str_tags == '[]':
        return []
    return list(map(int, str_tags.replace('[', '').replace(']', '').replace(' ', '').split(',')))


def str_album_to_int(album):
    if album == '[]' or album == '[None]':
        return -1
    return int(album.replace('[', '').replace(']', ''))


# In[ ]:


def get_tracks_tags(track_final):
    track_tags_list = str_tags_to_list(track_final['tags'])
    return [[track_final['track_id'], track_tag] for track_tag in track_tags_list]

tracks_tags = pd.concat([pd.DataFrame(data=get_tracks_tags(track_final), columns=['track_id', 'tag']) for index, track_final in tracks_final.iterrows()])
print('tracks_tags {}'.format(tracks_tags.shape))
print(tracks_tags.head(6))


# In[ ]:


def get_track_album(track_final):
    track_album = str_album_to_int(track_final['album'])
    return [[track_final['track_id'], track_album]]

tracks_albums = pd.concat([pd.DataFrame(data=get_track_album(track_final), columns=['track_id', 'album']) for index, track_final in tracks_final.iterrows()])
print('tracks_albums {}'.format(tracks_albums.shape))
print(tracks_albums.head(6))


# In[ ]:


# Remove tracks without album
tracks_albums = tracks_albums[tracks_albums.album != -1]
print('tracks_albums {}'.format(tracks_albums.shape))
print(tracks_albums.head(6))


# In[ ]:


def get_playlist_titles(playlist_final):
    playlist_tags_list = str_tags_to_list(playlist_final['title'])
    return [[playlist_final['playlist_id'], playlist_tag] for playlist_tag in playlist_tags_list]

playlist_titles = pd.concat([pd.DataFrame(data=get_playlist_titles(playlist_final), columns=['playlist_id', 'title']) for index, playlist_final in playlists_final.iterrows()])
print('playlist_titles {}'.format(playlist_titles.shape))
print(playlist_titles.head(6))


# In[ ]:


tracks_artist = pd.DataFrame()
tracks_artist['track_id'] = tracks_final['track_id']
tracks_artist['artist_id'] = tracks_final['artist_id']
print('tracks_artist {}'.format(tracks_artist.shape))
print(tracks_artist.head(6))


# In[ ]:


print('Tracks with album count: {}'.format(tracks_albums['track_id'].nunique()))
print('Unique album count: {}\n'.format(tracks_albums['album'].nunique()))

print('Tracks with tags count: {}'.format(tracks_tags['track_id'].nunique()))
print('Unique tags count: {}\n'.format(tracks_tags['tag'].nunique()))

print('Tracks with artists count: {}'.format(tracks_artist['track_id'].nunique()))
print('Unique artists count: {}\n'.format(tracks_artist['artist_id'].nunique()))

print('Playlists with title count: {}'.format(playlist_titles['playlist_id'].nunique()))
print('Unique titles count: {}\n'.format(playlist_titles['title'].nunique()))


# In[ ]:


print(list(tracks_final['track_id'])[:6])
track_id_le = preprocessing.LabelEncoder()
track_id_le.fit(list(tracks_final['track_id']))
print('track_id_le classes: {}'.format(len(track_id_le.classes_)))
transformed_track_id = track_id_le.transform(list(tracks_tags['track_id']))
print(transformed_track_id[:6])
tracks_tags['transformed_track_id'] = transformed_track_id

print(list(tracks_tags['tag'])[:6])
tags_le = preprocessing.LabelEncoder()
tags_le.fit(list(tracks_tags['tag']))
print('tags_le classes: {}'.format(len(tags_le.classes_)))
transformed_tags = tags_le.transform(list(tracks_tags['tag']))
print(transformed_tags[:6])
tracks_tags['transformed_tag'] = transformed_tags

print(tracks_tags.head(6))


# In[ ]:


album_le = preprocessing.LabelEncoder()
album_le.fit(list(tracks_albums['album']))
print('album_le classes: {}'.format(len(album_le.classes_)))
tracks_albums['transformed_track_id'] = track_id_le.transform(list(tracks_albums['track_id']))
tracks_albums['album'] = list(map(lambda x: x+31900, album_le.transform(list(tracks_albums['album']))))
print(tracks_albums.head(6))


# In[ ]:


artist_le = preprocessing.LabelEncoder()
artist_le.fit(list(tracks_artist['artist_id']))
print('album_le classes: {}'.format(len(artist_le.classes_)))
tracks_artist['transformed_track_id'] = track_id_le.transform(list(tracks_artist['track_id']))
tracks_artist['transformed_artist_id'] = list(map(lambda x: x+31900+27604,                                                  artist_le.transform(list(tracks_artist['artist_id']))))
print(tracks_artist.head(6))


# In[ ]:


# Playlist and tracks that belong to them
target_playlists_and_tracks = pd.merge(target_playlists, train_final, on='playlist_id')
print('target_playlists_and_tracks {}'.format(target_playlists_and_tracks.shape))
print(target_playlists_and_tracks.head(10))


# In[ ]:


def split_training_data(train_final, target_playlists_and_tracks, random_state):
    validation_set = target_playlists_and_tracks.groupby(['playlist_id'])                        .apply(lambda x: x.sample(n=3, random_state=random_state))                        .reset_index(drop=True)
    df_concat = pd.concat([train_final, validation_set])
    training_set = df_concat.drop_duplicates(keep=False)
    return training_set, validation_set

# Split dataset - from all target playlists remove randomly 3 tracks
training_set, validation_set = split_training_data(train_final, target_playlists_and_tracks, random_state=0)
test_target_tracks = validation_set['track_id'].drop_duplicates(keep='first').to_frame()
test_target_tracks['transformed_track_id'] = track_id_le.transform(list(test_target_tracks['track_id']))
target_tracks['transformed_track_id'] = track_id_le.transform(list(target_tracks['track_id']))

print('training_set: {} validation_set: {}'.format(training_set.shape, validation_set.shape))
print(training_set.head(5))
print('training_set: {} validation_set: {}'.format(training_set.shape, validation_set.shape))
print(validation_set.head(5))
print('test_target_tracks: {}'.format(test_target_tracks.shape))
print(test_target_tracks.head(5))


# In[ ]:


tracks_titles = pd.merge(playlist_titles, training_set, on='playlist_id')
tracks_titles['transformed_track_id'] = track_id_le.transform(list(tracks_titles['track_id']))
print(tracks_titles.shape)
print(tracks_titles.head(3))

ones = np.ones(tracks_titles.shape[0])
print('ones shape: {}, vector: {}'.format(ones.shape, ones))
tracks_with_title = scipy.sparse.coo_matrix((ones, (list(tracks_titles['transformed_track_id']), list(tracks_titles['title']))))
tracks_with_title = tracks_with_title.tocsr()

print(tracks_with_title.shape)


# In[ ]:


print('tracks_tags.shape {}'.format(tracks_tags.shape))
print('tracks_albums.shape {}'.format(tracks_albums.shape))
print('tracks_artist.shape {}'.format(tracks_artist.shape))

ones = np.ones(tracks_tags.shape[0] + tracks_albums.shape[0] + tracks_artist.shape[0])
print('ones shape: {}, vector: {}'.format(ones.shape, ones))

ICM_tags = scipy.sparse.coo_matrix((ones, (list(tracks_tags['transformed_track_id']) + list(tracks_albums['transformed_track_id']) + list(tracks_artist['transformed_track_id'])                                          , list(tracks_tags['transformed_tag']) + list(tracks_albums['album']) + list(tracks_artist['transformed_artist_id']))))
ICM_tags = ICM_tags.tocsr()

# Add tracks that do not have tags and its transformed id is bigger than the biggest that has a tag
# missing_items = np.zeros((1, tracks_tags['transformed_tag'].nunique() + tracks_albums['album'].nunique() + tracks_artist['artist_id'].nunique()))
# missing_items = scipy.sparse.csr_matrix(missing_items)
# ICM_tags = scipy.sparse.vstack((ICM_tags, missing_items))

print(ICM_tags.shape)
features_per_item = (ICM_tags > 0).sum(axis=1)
items_per_feature = (ICM_tags > 0).sum(axis=0)

print('features_per_item.shape {}'.format(features_per_item.shape))
print('items_per_feature.shape {}'.format(items_per_feature.shape))


# In[ ]:


features_per_item = np.array(features_per_item).squeeze()
items_per_feature = np.array(items_per_feature).squeeze()

print(features_per_item.shape)
print(items_per_feature.shape)

features_per_item = np.sort(features_per_item)
items_per_feature = np.sort(items_per_feature)


# In[ ]:


import matplotlib.pyplot as pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

pyplot.plot(features_per_item, 'ro')
pyplot.ylabel('Num features ')
pyplot.xlabel('Item Index')
pyplot.show()


# In[ ]:


pyplot.plot(items_per_feature, 'ro')
pyplot.ylabel('Num items ')
pyplot.xlabel('Feature Index')
pyplot.show()


# In[ ]:


if SUBMISSION:
    train_final['count'] = train_final.groupby(['track_id']).transform('count')
    tracks_with_popularity = train_final.groupby(['track_id', 'count']).head(1).sort_values('count', ascending=False)
    target_tracks_with_popularity = pd.merge(target_tracks, tracks_with_popularity, on='track_id').groupby('track_id').head(1)
    target_tracks_with_popularity = target_tracks_with_popularity.sort_values('count').reset_index()
    print(target_tracks_with_popularity.head(5))
else:
    training_set['count'] = training_set.groupby(['track_id']).transform('count')
    tracks_with_popularity = training_set.groupby(['track_id', 'count']).head(1).sort_values('count', ascending=False)
    target_tracks_with_popularity = pd.merge(test_target_tracks, tracks_with_popularity, on='track_id').groupby('track_id').head(1)
    target_tracks_with_popularity = target_tracks_with_popularity.sort_values('count').reset_index()
    print(target_tracks_with_popularity.head(5))


# In[ ]:


# target_tracks_popularity = target_tracks_with_popularity.sort_values('count').reset_index()
print(target_tracks_with_popularity['count'].describe())
pyplot.plot(target_tracks_with_popularity['count'], 'ro')
pyplot.ylabel('Popularity')
pyplot.xlabel('Target tracks')
pyplot.show()


# In[ ]:


popularity_sum = target_tracks_with_popularity['count'].sum()
target_tracks_with_popularity['predictions'] = target_tracks_with_popularity['count'] / popularity_sum * 30000

print(target_tracks_with_popularity.tail(5))
print(target_tracks_with_popularity['predictions'].sum())

print(target_tracks_with_popularity['predictions'].describe())

def round_ceil(x):
    x['predictions'] = int(math.ceil(x['predictions']))
    return x

def reduce_by_one(x):
    if x['predictions'] > 1:
        x['predictions'] = x['predictions'] - 1
    return x

target_tracks_with_predictions = target_tracks_with_popularity.apply(round_ceil, axis=1)
to_reduce = target_tracks_with_predictions['predictions'].sum() - 30000

print(target_tracks_with_predictions.tail(5))
print(target_tracks_with_predictions['predictions'].sum())


# In[ ]:


def is_relevant(recommendation_item, validation_set):
    validation_item = validation_set.loc[validation_set['playlist_id'] == recommendation_item['playlist_id']]
    recommendation_item['recommendation'] = pd.Series(recommendation_item['recommendation'])                                                .isin(list(validation_item['track_id']))
    return recommendation_item


def precision(recommended_items_relevance):
    precision_scores = recommended_items_relevance.sum(axis=1) / recommended_items_relevance.shape[1]
    return precision_scores.mean()


def mAP(recommended_items_relevance):
    p_at_k = recommended_items_relevance.cumsum(axis=1) / (1 + np.arange(recommended_items_relevance.shape[1]))
    recommended_items_mAP = p_at_k.sum(axis=1) / recommended_items_relevance.shape[1]
    return recommended_items_mAP.mean()


def evaluate_recommendations(recommended_items, validation_set):
    items_relevance = recommended_items.apply(lambda recommendation_item: is_relevant(recommendation_item, validation_set), axis=1)
    recommended_items_relevance = pd.DataFrame(list(items_relevance['recommendation']), index=items_relevance['recommendation'].index)
    precision_score = precision(recommended_items_relevance)
    mAP_score = mAP(recommended_items_relevance)
    return precision_score, mAP_score

def evaluate(recommended_items, validation_set):
    print('Evaluating...')
    begin = time.time()
    precision_score, mAP_score = evaluate_recommendations(recommended_items, validation_set)
    print('Precision: {0:.{digits}f}, mAP: {1:.{digits}f}, took {2:.{digits}f}s'
          .format(precision_score, mAP_score, time.time() - begin, digits=5))


# In[ ]:


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, scipy.sparse.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, scipy.sparse.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, scipy.sparse.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, scipy.sparse.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, scipy.sparse.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, scipy.sparse.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, scipy.sparse.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


# In[ ]:


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
#         print(X.indptr)
        
        col_nnz = np.diff(X.indptr)
#         print(col_nnz)
        # then normalize the values in each column
        X.data /= np.repeat(norm, col_nnz)
        print("Normalized")
#         print(X[:2][:2])
#         print(norm)
#         print(col_nnz)

        # 2) compute the cosine similarity using the dot-product
        print("Computing distance")
        dist = X * X.T
        print("Computed")
        
        # zero out diagonal values
#         dist = dist - scipy.sparse.dia_matrix((dist.diagonal()[scipy.newaxis, :], [0]), shape=dist.shape)
#         print("Removed diagonal")
        
        # and apply the shrinkage
#         if self.shrinkage > 0:
#             dist = self.apply_shrinkage(X, dist)
#             print("Applied shrinkage")    
        
        return dist

    def apply_shrinkage(self, X, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind * X_ind.T
        # remove the diagonal
#         co_counts = co_counts - scipy.sparse.dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        co_counts_shrink = co_counts.copy()
        co_counts_shrink.data += self.shrinkage
        co_counts.data /= co_counts_shrink.data
        dist.data *= co_counts.data
        return dist


# In[ ]:


distance = Cosine()
isim = distance.compute(ICM_tags)


# In[ ]:


class ContentBasedRecommender:
    def __init__(self, shrinkage=10, similarity='cosine'):
        self.shrinkage = shrinkage
        self.similarity_name = similarity
        if similarity == 'cosine':
            self.distance = Cosine(shrinkage=self.shrinkage)
        else:
            raise NotImplementedError('Distance {} not implemented'.format(similarity))
    
    def fit(self, training_set, ICM, target_tracks, tracks_with_title, playlist_titles, items_similarity):
        self.training_set = training_set
#         self.items_similarity = self.distance.compute(ICM)
        self.items_similarity = items_similarity
        self.tracks_with_title = tracks_with_title
        self.playlist_titles = playlist_titles
        
        self.target_tracks_mask = np.zeros(self.items_similarity.shape[0])
        for value in list(target_tracks['transformed_track_id']):
            self.target_tracks_mask[value] = 1
    
    def recommend(self, target_playlists):
        def make_recommendation(playlist):
            tracks_on_playlist = self.training_set.loc[self.training_set['playlist_id'] == playlist['playlist_id']]
            transformed_tracks_on_playlist = track_id_le.transform(list(tracks_on_playlist['track_id']))
            tracks_on_playlist_mask = np.ones(self.items_similarity.shape[0])
            for value in transformed_tracks_on_playlist:
                tracks_on_playlist_mask[value] = 0
            
#             titles = self.playlist_titles.loc[self.playlist_titles['playlist_id'] == playlist['playlist_id']]
#             titles_mask = np.squeeze(np.asarray(self.tracks_with_title[:, titles['title']].sum(axis=1)))
#             titles_mask = titles_mask / (np.amax(titles_mask)+ 1)
#             titles_mask = np.log(np.squeeze(np.asarray(titles_mask + 3)))
            tracks_tags_correlation = np.squeeze(np.asarray(recommender.items_similarity[:, transformed_tracks_on_playlist].sum(axis=1)))
            tracks_tags_correlation = tracks_tags_correlation * self.target_tracks_mask
            tracks_tags_correlation = tracks_tags_correlation * tracks_on_playlist_mask
#             tracks_tags_correlation = tracks_tags_correlation * titles_mask
            ind = np.argpartition(list(tracks_tags_correlation), -3)[-3:]
    
            recommended_tracks = track_id_le.inverse_transform(ind)
            playlist['recommendation'] = list(reversed(recommended_tracks))
            return playlist
        recommended_items = target_playlists.apply(lambda playlist: make_recommendation(playlist), axis=1)
        return recommended_items


# In[ ]:


print('Building model...')
begin = time.time()
recommender = ContentBasedRecommender()
recommender.fit(training_set, ICM_tags, test_target_tracks, tracks_with_title, playlist_titles, isim)
# recommender.fit(train_final, ICM_tags, target_tracks, tracks_with_title, playlist_titles)
print('Took {0:.{digits}f}s'.format(time.time() - begin, digits=5))


# In[1]:


print('Recommending...')
begin = time.time()
recommended_items = recommender.recommend(target_playlists.head(1))
print('Took {0:.{digits}f}s'.format(time.time() - begin, digits=5))

print('recommended_items {}'.format(recommended_items.shape))
print(recommended_items.head(3))


# In[40]:


if not SUBMISSION:
    evaluate(recommended_items, validation_set)


# In[76]:


print(recommended_items.shape)
print(recommended_items.sort_values('playlist_id').head(10))
print(validation_set.loc[validation_set['playlist_id'].isin(recommended_items['playlist_id'])])


# In[64]:


def print_results(recommended_items, filename):
    print('Printing...')
    with open('submissions/{}'.format(filename), 'w') as output_file:
        output_file.write('playlist_id,track_ids\n')
        for index, recommendation in recommended_items.iterrows():
            row = '{},'.format(recommendation['playlist_id'])
            for track_id in pd.Series(recommendation['recommendation']).values:
                row += ' {}'.format(track_id)
            row += '\n'
            output_file.write(row)
print_results(recommended_items, filename=SUBMISSION_FILENAME if SUBMISSION else TEST_FILENAME)

