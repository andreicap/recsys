

#item-similarity-model


is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')


playlist_id = playlists[5]
user_items = is_model.get_user_items(playlist_id)
is_model.recommend(playlist_id)

recommend_final = []
for p in playlists:
   recs = is_model.recommend(p);
   recommend_final.append(str(p)+","+" ".join(str(v) for v in recs['song'][0:5].values))





class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.playlist_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    #Get unique items (songs) corresponding to a given playlist
    def get_playlist_items(self, playlist):
        playlist_data = self.train_data[self.train_data[self.playlist_id] == playlist]
        playlist_items = list(playlist_data[self.item_id].unique())
        
        return playlist_items
        
    #Get unique playlists for a given item (song)
    def get_item_playlists(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_playlists = set(item_data[self.playlist_id].unique())
            
        return item_playlists
        
    #Get unique items (songs) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items
        
    #Construct cooccurence matrix
    def construct_cooccurence_matrix(self, playlist_songs, all_songs):
            
        ####################################
        #Get playlists for all songs in playlist_songs.
        ####################################
        playlist_songs_playlists = []        
        for i in range(0, len(playlist_songs)):
            playlist_songs_playlists.append(self.get_item_playlists(playlist_songs[i]))
            
        ###############################################
        #Initialize the item cooccurence matrix of size 
        #len(playlist_songs) X len(songs)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(playlist_songs), len(all_songs))), float)
           
        #############################################################
        #Calculate similarity between playlist songs and all unique songs
        #in the training data
        #############################################################
        for i in range(0,len(all_songs)):
            #Calculate unique listeners (playlists) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            playlists_i = set(songs_i_data[self.playlist_id].unique())
            
            for j in range(0,len(playlist_songs)):       
                    
                #Get unique listeners (playlists) of song (item) j
                playlists_j = playlist_songs_playlists[j]
                    
                #Calculate intersection of listeners of songs i and j
                playlists_intersection = playlists_i.intersection(playlists_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(playlists_intersection) != 0:
                    #Calculate union of listeners of songs i and j
                    playlists_union = playlists_i.union(playlists_j)
                    
                    cooccurence_matrix[j,i] = float(len(playlists_intersection))/float(len(playlists_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix

    
    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, playlist, cooccurence_matrix, all_songs, playlist_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all playlist songs.
        playlist_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        playlist_sim_scores = np.array(playlist_sim_scores)[0].tolist()
 
        #Sort the indices of playlist_sim_scores based upon their value
        #Also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(playlist_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['playlist_id', 'song', 'score', 'rank']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pandas.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in playlist_songs and rank <= 10:
                df.loc[len(df)]=[playlist,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current playlist has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    #Create the item similarity based recommender system model
    def create(self, train_data, playlist_id, item_id):
        self.train_data = train_data
        self.playlist_id = playlist_id
        self.item_id = item_id

    #Use the item similarity based recommender system model to
    #make recommendations
    def recommend(self, playlist):
        
        ########################################
        #A. Get all unique songs for this playlist
        ########################################
        playlist_songs = self.get_playlist_items(playlist)    
            
        print("No. of unique songs for the playlist: %d" % len(playlist_songs))
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(playlist_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(playlist_songs, all_songs)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(playlist, cooccurence_matrix, all_songs, playlist_songs)
                
        return df_recommendations
    
    #Get similar items to given items
    def get_similar_items(self, item_list):
        
        playlist_songs = item_list
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(playlist_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(playlist_songs, all_songs)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        playlist = ""
        df_recommendations = self.generate_top_recommendations(playlist, cooccurence_matrix, all_songs, playlist_songs)
         
        return df_recommendations