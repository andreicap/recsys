import numpy as np


class Sampler:
    def __init__(self, target_tracks, URM_mask):
        self.target_tracks_set = set(target_tracks['transformed_track_id'])
        self.URM_mask = URM_mask

        self.n_playlists = self.URM_mask.shape[0]
        self.n_tracks = self.URM_mask.shape[1]

        # Extract playlists having at least one track to choose from
        self.eligible_playlists = []
        for playlist_id in range(self.n_playlists):
            start_pos = self.URM_mask.indptr[playlist_id]
            end_pos = self.URM_mask.indptr[playlist_id + 1]

            if len(self.URM_mask.indices[start_pos:end_pos]) > 0:
                self.eligible_playlists.append(playlist_id)

    def sample_playlist(self):
        playlist_id = np.random.choice(self.eligible_playlists)
        return playlist_id

    def sample_triplet(self, playlist_id=None):
        """Returns for the given playlist a random track that is on it and that is not"""
        if playlist_id is None:
            playlist_id = self.sample_playlist()
        # Get playlist tracks and choose one
        playlist_tracks = self.URM_mask[playlist_id, :].indices
        positive_track_id = np.random.choice(playlist_tracks)

        negative_track_selected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # Don't try again. Instead build mapping of non-seen items. Check if exluding URM_test helps!!!
        while not negative_track_selected:
            negative_track_id = np.random.randint(0, self.n_tracks)
            if negative_track_id not in playlist_tracks and negative_track_id not in self.target_tracks_set:
                negative_track_selected = True

        return playlist_id, positive_track_id, negative_track_id

    def sample_batch(self, batch_size=10):
        playlist_ids = np.random.choice(self.eligible_playlists, size=batch_size)
        positive_track_ids = [None] * batch_size
        negative_track_ids = [None] * batch_size

        for sample_index in range(batch_size):
            playlist_id = playlist_ids[sample_index]

            playlist_tracks = self.URM_mask[playlist_id, :].indices
            positive_track_ids[sample_index] = np.random.choice(playlist_tracks)

            negative_track_selected = False

            while not negative_track_selected:
                negative_track_id = np.random.randint(0, self.n_tracks)
                if negative_track_id not in playlist_tracks and negative_track_id not in self.target_tracks_set:
                    negative_track_selected = True
                    negative_track_ids[sample_index] = negative_track_id

        return list(playlist_ids), positive_track_ids, negative_track_ids

