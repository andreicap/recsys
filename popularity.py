

######################################################

playlist_rec = []
tracks_rec = []
x=0  
for i in wide_data.columns:
  if np.isin(wide_data.columns[i],(target_playlists.playlist_id)):
    query_index = i
    distances, indices = model_knn.kneighbors(wide_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
    for i in range(0, len(distances.flatten())):
      if i == 0:
          print(wide_data.columns[query_index], ',', sep = "", end = "")
          playlist_rec.append(wide_data.columns[query_index])
      else:
          print(wide_data.index[indices.flatten()[i]], end = " ")
      print()
      if x>10:                                                       
        break;  


######################################################
x=0                                                              
for i in wide_data.columns:                                          
  if np.isin(wide_data.columns[i],(target_playlists.playlist_id)):   
    print('---')                                                   
    print(i)                                                       
    print(wide_data.columns[i])                                      
    print(np.isin(wide_data.columns[i],(target_playlists.playlist_id)
))                                                                 
    print('---')
    x+=1;
    if x>10:                                                       
      break;  

##############################################################


train_data = combined.copy()
train_data.columns = ['user_id', 'song', 'artist_id', 'duration', 
     ...: 'listen_count', 'album', 'tags' ]




class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        #Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)
    
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(5)

    #Use the popularity based recommender system model to
    #make recommendations
    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        
        #Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id
    
        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        
        return user_recommendations

import string

recommend_final = []
for p in playlists:
   recs = pm.recommend(p);
   recommend_final.append(str(p)+","+" ".join(str(v) for v in recs['song'][0:5].values))

import csv

titles = ['playlist_id', 'track_ids']
wr = open("sample_submission.csv",'w')
for item in recommend_final:
    wr.write("%s\n" % item)