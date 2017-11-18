import time
import numpy as np
import scipy.sparse
import scipy.special
from evaluation import evaluate_recommendations
from sampler import Sampler
from utils import similarity_matrix_topK


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SLIM_BPR_Recommender:
    def __init__(self, URM_train, test_target_tracks, target_tracks, track_id_le, training_set, sparse_weights=True):
        self.URM_train = URM_train
        self.target_tracks = target_tracks
        self.track_id_le = track_id_le
        self.training_set = training_set
        self.n_playlists = self.URM_train.shape[0]
        self.n_tracks = self.URM_train.shape[1]

        self.URM_mask = self.URM_train.copy().tocsr()
        self.URM_mask.eliminate_zeros()
        self.sampler = Sampler(self.target_tracks, self.URM_mask)
        self.sparse_weights = sparse_weights

        if self.sparse_weights:
            self.similarity_matrix = scipy.sparse.csr_matrix((self.n_tracks, self.n_tracks), dtype=np.float32)
        else:
            self.similarity_matrix = np.zeros((self.n_tracks, self.n_tracks)).astype('float32')

        self.target_tracks_mask = np.zeros(self.similarity_matrix.shape[0])
        for value in list(test_target_tracks['transformed_track_id']):
            self.target_tracks_mask[value] = 1

    def recommend(self, target_playlists, to_predict=3):
        def make_recommendation(playlist):
            tracks_on_playlist = self.training_set.loc[self.training_set['playlist_id'] == playlist['playlist_id']]
            transformed_tracks_on_playlist = self.track_id_le.transform(list(tracks_on_playlist['track_id']))
            tracks_on_playlist_mask = np.ones(self.similarity_matrix.shape[0])
            for value in transformed_tracks_on_playlist:
                tracks_on_playlist_mask[value] = 0

            correlation = np.squeeze(np.asarray(self.W_sparse[:, transformed_tracks_on_playlist].sum(axis=1)))
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

    def update_similarity_matrix(self):
        self.W_sparse = similarity_matrix_topK(self.similarity_matrix.T, k=self.topK)

    def fit(self, epochs=30, batch_size=1000, lambda_i=0.0025, lambda_j=0.00025, learning_rate=0.05, topK=100,
            target_playlists=None, validation_set=None, validate_every_n_epochs=1):
        self.topK = topK
        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate

        train_start_time = time.time()

        for current_epoch in range(epochs):
            start_time_epoch = time.time()

            # if current_epoch > 0:
            self.epoch_iteration()
            # else:
            #     self.update_similarity_matrix()

            if target_playlists is not None and ((current_epoch + 1) % validate_every_n_epochs == 0) and current_epoch != 0:
                print("Evaluation begins...")
                eval_begin = time.time()
                recommended_items = self.recommend(target_playlists)
                eval_results = evaluate_recommendations(recommended_items, validation_set)
                self.write_config(current_epoch + 1, eval_results, time.time() - eval_begin)
            print(
                'Epoch {} of {} complete in {:.2f} minutes'.format(current_epoch + 1, epochs, float(time.time() - start_time_epoch) / 60))
        print('Fit completed in {:.2f} minutes'.format(float(time.time() - train_start_time) / 60))

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

    def epoch_iteration(self):
        # Get number of available interactions
        num_positive_iteractions = int(self.URM_mask.nnz * 0.001)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        total_number_of_batch = int(num_positive_iteractions / self.batch_size) + 1

        # Uniform user sampling without replacement
        for current_batch in range(total_number_of_batch):
            sgd_playlists, sgd_positive_tracks, sgd_negative_tracks = self.sampler.sample_batch(batch_size=self.batch_size)
            self.update_weights_batch(sgd_playlists, sgd_positive_tracks, sgd_negative_tracks)

            # self.update_weights_loop(sgd_playlists, sgd_positive_tracks, sgd_negative_tracks)

            if time.time() - start_time_batch >= 30 or current_batch == total_number_of_batch - 1:
                print('Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}'.format(
                    (current_batch + 1) * self.batch_size,
                    100.0 * float((current_batch + 1) * self.batch_size) / num_positive_iteractions,
                    time.time() - start_time_batch,
                    float(current_batch + 1) * self.batch_size / (time.time() - start_time_epoch)))
                start_time_batch = time.time()

        # zero out diagonal
        self.similarity_matrix[np.arange(0, self.n_tracks), np.arange(0, self.n_tracks)] = 0.0

        self.update_similarity_matrix()

    def update_weights_batch(self, u, i, j):
        """
        Define the update rules to be used in the train phase and compile the train function
        :return:
        """
        x_ui = self.similarity_matrix[i]
        x_uj = self.similarity_matrix[j]

        # The difference is computed on the user_seen items
        x_uij = x_ui - x_uj
        x_uij = self.URM_mask[u, :].dot(x_uij.T).diagonal()

        gradient = np.sum(1 / (1 + np.exp(x_uij))) / self.batch_size

        itemsToUpdate = np.array(self.URM_mask[u, :].sum(axis=0) > 0).ravel()

        # Do not update items i, set all user-posItem to false
        # itemsToUpdate[i] = False

        self.similarity_matrix[i] += self.learning_rate * gradient * itemsToUpdate
        self.similarity_matrix[i, i] = 0

        # Now update i, setting all user-posItem to true
        # Do not update j

        # itemsToUpdate[i] = True
        # itemsToUpdate[j] = False

        self.similarity_matrix[j] -= self.learning_rate * gradient * itemsToUpdate
        self.similarity_matrix[j, j] = 0


class SLIM_BPR_Recommender_Cython(SLIM_BPR_Recommender):
    def __init__(self, URM_train, test_target_tracks, target_tracks, target_playlists, track_id_le, training_set, sparse_weights=True, sgd_mode='adagrad'):
        SLIM_BPR_Recommender.__init__(self, URM_train, test_target_tracks, target_tracks, track_id_le, training_set, sparse_weights)
        self.sgd_mode = sgd_mode
        self.target_tracks_set = set(target_tracks['transformed_track_id'])
        self.target_playlists_set = set(target_playlists['transformed_playlist_id'])

    def fit(self, epochs=30, batch_size=1000, lambda_i=0.0025, lambda_j=0.00025, learning_rate=0.05, topK=100,
            target_playlists=None, validation_set=None, validate_every_n_epochs=1):

        self.eligible_playlists = np.array(self.sampler.eligible_playlists, dtype=np.int64)

        from Cython.cython_epoch import SLIM_BPR_Cython_Epoch

        self.cython_epoch = SLIM_BPR_Cython_Epoch(self.URM_mask,
                                                  self.sparse_weights,
                                                  self.eligible_playlists,
                                                  self.target_tracks_set,
                                                  self.target_playlists_set,
                                                  topK=topK,
                                                  learning_rate=learning_rate,
                                                  batch_size=1,
                                                  sgd_mode=self.sgd_mode,
                                                  lambda_i=lambda_i,
                                                  lambda_j=lambda_j)

        # Cal super.fit to start training
        SLIM_BPR_Recommender.fit(self,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 lambda_i=lambda_i,
                                 lambda_j=lambda_j,
                                 learning_rate=learning_rate,
                                 topK=topK,
                                 target_playlists=target_playlists,
                                 validation_set=validation_set,
                                 validate_every_n_epochs=validate_every_n_epochs
                                 )

    def epoch_iteration(self):
        self.S = self.cython_epoch.epoch_iteration_cython()

        if self.sparse_weights:
            self.W_sparse = self.S
        else:
            self.W_sparse = self.S

    def write_config(self, current_epoch, eval_results, eval_time):
        precision_score, mAP_score = eval_results
        current_config = {'lambda_i': self.lambda_i,
                          'lambda_j': self.lambda_j,
                          'batch': self.batch_size,
                          'learn_rate': self.learning_rate,
                          'topK': self.topK,
                          'epoch': current_epoch,
                          'sgd_mode': self.sgd_mode}

        print('Test case: {}\n'.format(current_config))
        print('Precision: {0:.{digits}f}, mAP: {1:.{digits}f}, took {2:.{digits}f}s'
              .format(precision_score, mAP_score, eval_time, digits=5))
