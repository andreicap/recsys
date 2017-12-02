import time
import numpy as np
import scipy.sparse
from evaluation import evaluate_recommendations


class SLIM_BPR_Recommender_Cython_User:
    def __init__(self, URM_train, test_target_tracks, target_tracks, target_playlists, track_id_le, training_set, sgd_mode='adagrad'):
        super(SLIM_BPR_Recommender_Cython_User, self).__init__()
        self.URM_train = URM_train.T.copy().tocsr()
        self.n_tracks = self.URM_train.shape[0]
        self.n_playlists = self.URM_train.shape[1]

        self.target_tracks = target_tracks
        self.track_id_le = track_id_le
        self.training_set = training_set

        self.URM_mask = self.URM_train.copy().tocsr()
        self.URM_mask.eliminate_zeros()

        self.sgd_mode = sgd_mode
        self.target_tracks_set = set(target_tracks['transformed_track_id'])
        self.target_playlists_set = set(target_playlists['transformed_playlist_id'])

        self.target_tracks_mask = np.zeros(self.n_tracks)
        for value in list(test_target_tracks['transformed_track_id']):
            self.target_tracks_mask[value] = 1

        eligible_tracks = []
        for track_id in range(self.n_tracks):
            start_pos = self.URM_mask.indptr[track_id]
            end_pos = self.URM_mask.indptr[track_id + 1]

            if len(self.URM_mask.indices[start_pos:end_pos]) > 0:
                eligible_tracks.append(track_id)
        self.eligible_tracks = np.array(eligible_tracks, dtype=np.int64)

        from Cython.cython_epoch import SLIM_BPR_User_Cython_Epoch

        self.cython_epoch = SLIM_BPR_User_Cython_Epoch(self.URM_mask,
                                                       self.eligible_tracks,
                                                       self.target_tracks_set,
                                                       self.target_playlists_set,
                                                       sgd_mode=self.sgd_mode)

    def fit(self, epochs=30, batch_size=1000, lambda_i=0.0025, lambda_j=0.00025, learning_rate=0.05, topK=100,
            target_playlists=None, validation_set=None, validate_every_n_epochs=1, k=7):
        self.topK = topK
        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        mAP_max = 0.0
        train_start_time = time.time()

        for current_epoch in range(epochs):
            start_time_epoch = time.time()

            self.S = self.cython_epoch.epoch_iteration_cython(topK=self.topK,
                                                              learning_rate=self.learning_rate,
                                                              batch_size=self.batch_size,
                                                              lambda_i=self.lambda_i,
                                                              lambda_j=self.lambda_j)
            self.W_sparse = self.S

            if target_playlists is not None and ((current_epoch + 1) % validate_every_n_epochs == 0) and current_epoch != 0:
                print("Evaluation begins...")
                eval_begin = time.time()
                recommended_items = self.recommend(target_playlists, k=k)
                eval_results = evaluate_recommendations(recommended_items, validation_set)
                if eval_results[1] > mAP_max:
                    scipy.sparse.save_npz('prod_slim_user_20_{}'.format(current_epoch + 1), self.W_sparse)
                    mAP_max = eval_results[1]
                self.write_config(current_epoch + 1, eval_results, time.time() - eval_begin)
            print(
                'Epoch {} of {} complete in {:.2f} minutes'.format(current_epoch + 1, epochs,
                                                                   float(time.time() - start_time_epoch) / 60))
        print('Fit completed in {:.2f} minutes'.format(float(time.time() - train_start_time) / 60))

    def recommend(self, target_playlists, to_predict=3, k=7):
        def make_recommendation(playlist):
            tracks_on_playlist = self.training_set.loc[self.training_set['playlist_id'] == playlist['playlist_id']]
            transformed_tracks_on_playlist = self.track_id_le.transform(list(tracks_on_playlist['track_id']))
            tracks_on_playlist_mask = np.ones(self.n_tracks)
            for value in transformed_tracks_on_playlist:
                tracks_on_playlist_mask[value] = 0

            similar_playlists = np.squeeze(self.W_sparse.getrow(playlist['transformed_playlist_id']).toarray())
            indices = np.argpartition(similar_playlists, -k)[-k:]
            scores = np.take(similar_playlists, indices)

            correlation = np.zeros(self.n_tracks)
            for index, score in zip(indices, scores):
                correlation += np.squeeze(self.URM_mask.getcol(index).toarray()) * score

            correlation = correlation * self.target_tracks_mask
            correlation = correlation * tracks_on_playlist_mask
            track_indices = np.argpartition(list(correlation), -to_predict)[-to_predict:]
            track_scores = np.take(correlation, track_indices)

            recommended_tracks = self.track_id_le.inverse_transform(track_indices)
            playlist['recommendation'] = list(reversed(recommended_tracks))
            playlist['scores'] = sorted(list(track_scores), reverse=True)
            return playlist

        recommended_items = target_playlists.apply(lambda playlist: make_recommendation(playlist), axis=1)
        return recommended_items

    def write_config(self, current_epoch, eval_results, eval_time):
        precision_score, mAP_score = eval_results
        current_config = {'lambda_i': self.lambda_i,
                          'lambda_j': self.lambda_j,
                          'batch': self.batch_size,
                          'learn_rate': self.learning_rate,
                          'topK': self.topK,
                          'epoch': current_epoch}

        print('Test case: {}\n'.format(current_config))
        print('Precision: {0:.{digits}f}, mAP: {1:.{digits}f}, took {2:.{digits}f}s'
              .format(precision_score, mAP_score, eval_time, digits=5))
