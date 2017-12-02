#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from utils import check_matrix
import numpy as np
cimport numpy as np
import time
import sys

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX


cdef struct Sample:
    long playlist_id
    long positive_track_id
    long negative_track_id


cdef class MF_BPR_Cython_Epoch:
    cdef int n_playlists
    cdef int n_tracks, num_factors
    cdef int num_positive_iteractions

    cdef float learning_rate, bias_reg, user_reg, positive_reg, negative_reg

    cdef long[:] eligible_playlists
    cdef long n_eligible_playlists

    cdef set target_playlists
    cdef set target_tracks

    cdef int[:] sampled_playlist_tracks
    cdef int n_sampled_playlist_tracks

    cdef int[:] URM_mask_indices, URM_mask_indptr

    cdef double[:,:] W, H
    cdef double[:] item_bias


    def __init__(self, URM_mask, eligible_playlists, num_factors, target_tracks, target_playlists, sgd_mode='sgd'):
        super(MF_BPR_Cython_Epoch, self).__init__()

        URM_mask = check_matrix(URM_mask, 'csr')

        self.num_positive_iteractions = int(URM_mask.nnz * 1)
        self.n_playlists = URM_mask.shape[0]
        self.n_tracks = URM_mask.shape[1]
        self.num_factors = num_factors

        self.URM_mask_indices = URM_mask.indices
        self.URM_mask_indptr = URM_mask.indptr

        self.item_bias = np.zeros(self.n_tracks)
        # W and H cannot be initialized as zero, otherwise the gradient will always be zero
        self.W = np.random.random((self.n_playlists, self.num_factors))
        self.H = np.random.random((self.n_tracks, self.num_factors))

        if sgd_mode == 'sgd':
            pass
        else:
            raise ValueError("SGD_mode not valid. Acceptable values are: 'sgd'. Provided value was '{}'".format(sgd_mode))

        self.eligible_playlists = eligible_playlists
        self.n_eligible_playlists = len(eligible_playlists)

        self.target_tracks = target_tracks
        self.target_playlists = target_playlists

    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int[:] get_playlist_tracks(self, long index):
        return self.URM_mask_indices[self.URM_mask_indptr[index]: self.URM_mask_indptr[index + 1]]

    def epoch_iteration_cython(self, verbose=False, learning_rate=0.05, bias_reg=1.0, user_reg=0.0025, positive_reg=0.0025, negative_reg=0.00025):
        cdef long start_time_epoch = time.time()
        cdef long start_time_iter = time.time()

        cdef long total_number_of_iterations = self.num_positive_iteractions + 1

        cdef Sample sample
        cdef long u, i, j
        cdef long index, iteration
        cdef double x_uij, gradient

        cdef int numSeenItems
        cdef int print_step = 5000000

        cdef double H_i, H_j, W_u

        self.learning_rate = learning_rate
        self.bias_reg = bias_reg
        self.user_reg = user_reg
        self.positive_reg = positive_reg
        self.negative_reg = negative_reg

        for iteration in range(total_number_of_iterations):
            # Uniform user sampling with replacement
            sample = self.sample_cython()

            u = sample.playlist_id
            i = sample.positive_track_id
            j = sample.negative_track_id

            x_uij = 0.0

            for index in range(self.num_factors):
                x_uij += self.W[u, index] * (self.H[i, index] - self.H[j, index])

            x_uij += self.item_bias[i] - self.item_bias[j]

            gradient = 1 / (1 + exp(x_uij))


            d = gradient - self.bias_reg * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d

            d = -gradient - self.bias_reg * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

            for index in range(self.num_factors):

                # Copy original value to avoid messing up the updates
                H_i = self.H[i, index]
                H_j = self.H[j, index]
                W_u = self.W[u, index]

                self.W[u, index] += self.learning_rate * (gradient * ( H_i - H_j ) - self.user_reg * W_u)
                self.H[i, index] += self.learning_rate * (gradient * ( W_u ) - self.positive_reg * H_i)
                self.H[j, index] += self.learning_rate * (gradient * (-W_u ) - self.negative_reg * H_j)

            if verbose and (iteration % print_step == 0 and not iteration == 0 or iteration == total_number_of_iterations - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    iteration,
                    100.0 * float(iteration) / total_number_of_iterations,
                    time.time() - start_time_iter,
                    float(iteration + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_iter = time.time()

    def get_W(self):
        return np.array(self.W)

    def get_H(self):
        return np.array(self.H)

    cdef Sample sample_cython(self):
        cdef Sample sample = Sample()
        cdef long index
        cdef int negative_track_selected

        # Warning: rand() returns an integer

        cdef double RAND_MAX_DOUBLE = RAND_MAX

        index = int(rand() / RAND_MAX_DOUBLE * self.n_eligible_playlists)

        sample.playlist_id = self.eligible_playlists[index]

        self.sampled_playlist_tracks = self.get_playlist_tracks(sample.playlist_id)
        self.n_sampled_playlist_tracks = len(self.sampled_playlist_tracks)

        index = int(rand() / RAND_MAX_DOUBLE * self.n_sampled_playlist_tracks)
        # index = rand() % self.n_sampled_playlist_tracks

        sample.positive_track_id = self.sampled_playlist_tracks[index]

        negative_track_selected = False
        # It's faster to just try again then to build a mapping of the non-seen items for every user
        while not negative_track_selected:
            sample.negative_track_id = int(rand() / RAND_MAX_DOUBLE  * self.n_tracks)

            index = 0
            while index < self.n_sampled_playlist_tracks and self.sampled_playlist_tracks[index] != sample.negative_track_id:
                index += 1

            if index == self.n_sampled_playlist_tracks:# and (sample.playlist_id not in self.target_playlists or sample.negative_track_id not in self.target_tracks):
                negative_track_selected = True

        return sample
