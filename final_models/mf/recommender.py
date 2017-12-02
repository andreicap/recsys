import sys
import time
import numpy as np
from evaluation import evaluate_recommendations


class MF_BPR_Cython:
    def __init__(self, URM_train, target_tracks, target_playlists, track_id_le, playlist_id_le, training_set, num_factors=10,
                 sgd_mode='sgd'):
        super(MF_BPR_Cython, self).__init__()
        self.URM_train = URM_train.copy().tocsr()
        self.n_playlists = URM_train.shape[0]
        self.n_tracks = URM_train.shape[1]
        self.normalize = False
        self.num_factors = num_factors

        self.URM_mask = self.URM_train.copy().tocsr()
        self.URM_mask.eliminate_zeros()

        self.target_tracks_mask = np.zeros(self.n_tracks)
        for value in list(target_tracks['transformed_track_id']):
            self.target_tracks_mask[value] = 1

        self.target_tracks_set = set(target_tracks['transformed_track_id'])
        self.target_playlists_set = set(target_playlists['transformed_playlist_id'])

        self.track_id_le = track_id_le
        self.playlist_id_le = playlist_id_le
        self.training_set = training_set

        eligible_playlists = []
        for playlist_id in range(self.n_playlists):
            start_pos = self.URM_mask.indptr[playlist_id]
            end_pos = self.URM_mask.indptr[playlist_id + 1]

            if len(self.URM_mask.indices[start_pos:end_pos]) > 0:
                eligible_playlists.append(playlist_id)
        self.eligible_playlists = np.array(eligible_playlists, dtype=np.int64)

        from Cython.mf_bpr_cython_epoch import MF_BPR_Cython_Epoch

        self.sgd_mode = sgd_mode
        self.cython_epoch = MF_BPR_Cython_Epoch(self.URM_mask,
                                                self.eligible_playlists,
                                                self.num_factors,
                                                self.target_tracks_set,
                                                self.target_playlists_set,
                                                sgd_mode=self.sgd_mode)

    def fit(self, epochs=30, target_playlists=None, validation_set=None, validate_every_n_epochs=1, learning_rate=0.05,
            bias_reg=1.0, user_reg=0.0, positive_reg=0.0, negative_reg=0.0):
        self.learning_rate = learning_rate
        self.bias_reg = bias_reg
        self.user_reg = user_reg
        self.positive_reg = positive_reg
        self.negative_reg = negative_reg

        train_start_time = time.time()

        for current_epoch in range(epochs):
            start_time_epoch = time.time()

            self.cython_epoch.epoch_iteration_cython(learning_rate=self.learning_rate,
                                                     bias_reg=self.bias_reg,
                                                     user_reg=self.user_reg,
                                                     positive_reg=self.positive_reg,
                                                     negative_reg=self.negative_reg)

            if target_playlists is not None and ((current_epoch + 1) % validate_every_n_epochs == 0) and current_epoch != 0:
                print("Evaluation begins...")

                self.W = self.cython_epoch.get_W()
                self.H = self.cython_epoch.get_H()

                eval_begin = time.time()
                recommended_items = self.recommend(target_playlists, to_predict=10)
                eval_results = evaluate_recommendations(recommended_items, validation_set)
                self.write_config(current_epoch + 1, eval_results, time.time() - eval_begin)
                # print('Epoch {} of {} complete in {:.2f} minutes'.format(current_epoch + 1, epochs, float(time.time() - start_time_epoch) / 60))

        # Ensure W and H are up to date
        self.W = self.cython_epoch.get_W()
        self.H = self.cython_epoch.get_H()

        print('Fit completed in {:.2f} minutes'.format(float(time.time() - train_start_time) / 60))

        sys.stdout.flush()

    def write_config(self, current_epoch, eval_results, eval_time):
        precision_score, mAP_score = eval_results
        current_config = {'learn_rate': self.learning_rate,
                          'num_factors': self.num_factors,
                          'sgd_mode': self.sgd_mode,
                          'user_reg': self.user_reg,
                          'positive_reg': self.positive_reg,
                          'negative_reg': self.negative_reg,
                          'epoch': current_epoch}

        print('Test case: {}\n'.format(current_config))
        print('Precision: {0:.{digits}f}, mAP: {1:.{digits}f}, took {2:.{digits}f}s'
              .format(precision_score, mAP_score, eval_time, digits=5))
        sys.stdout.flush()

    def recommend(self, target_playlists, to_predict=3):
        def make_recommendation(playlist):
            tracks_on_playlist = self.training_set.loc[self.training_set['playlist_id'] == playlist['playlist_id']]
            transformed_tracks_on_playlist = self.track_id_le.transform(list(tracks_on_playlist['track_id']))
            tracks_on_playlist_mask = np.ones(self.n_tracks)
            for value in transformed_tracks_on_playlist:
                tracks_on_playlist_mask[value] = 0

            playlist_id = self.playlist_id_le.transform([playlist['playlist_id']])[0]

            correlation = np.dot(self.W[playlist_id], self.H.T)

            if self.normalize:
                # normalization will keep the scores in the same range
                # of value of the ratings in dataset
                rated = self.URM_train[playlist_id].copy()
                rated.data = np.ones_like(rated.data)
                print(rated.shape)
                print(self.W.shape)
                den = rated.dot(self.W).ravel()
                den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
                correlation /= den

            correlation = correlation * self.target_tracks_mask
            correlation = correlation * tracks_on_playlist_mask
            ind = np.argpartition(list(correlation), -to_predict)[-to_predict:]
            scores = np.take(correlation, ind)

            recommended_tracks = self.track_id_le.inverse_transform(ind)
            playlist['recommendation'] = list(reversed(recommended_tracks))
            playlist['scores'] = sorted(list(scores), reverse=True)
            return playlist

        recommended_items = target_playlists.apply(lambda playlist: make_recommendation(playlist), axis=1)
        return recommended_items
