from re import A
from bbq.archives.grid_archive import GridArchive
import numpy as np

class GridArchive(GridArchive):
    def __init__(self, p, seed=None, is_object=False):
        super().__init__(p, seed, is_object)

    def as_numpy(self, include_metadata=True):
        # Create array
        grid_res = [len(a)-1 for a in self.boundaries]
        n_obj     = len(self.get_random_elite().meta[0]['m_obj'])
        n_beh     = self._behavior_dim

        n_channels = sum([1, n_obj, n_beh, self.solution_dim])
        np_archive = np.full(np.r_[grid_res, n_channels], np.nan)

        # Fill array
        for elite in self:
            elite_stats = np.r_[elite.obj, elite.meta[0]['m_obj'], elite.beh, elite.sol]
            np_archive[elite.idx[0], elite.idx[1], :] = elite_stats
        if not include_metadata:
            return np_archive