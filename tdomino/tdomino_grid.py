import numpy as np
from bbq.archives import GridArchive
from ribs.archives._add_status import AddStatus


class TDominoGrid(GridArchive):
    def __init__(self, p, seed=None):
        super().__init__(p, seed=seed)
        self.n_obj = p['n_obj']
        self.hist_buffer = p['hist_length']
        self.reach = [4,4]# %p['reach']
        self.i_hist = np.zeros(np.r_[self.dims], dtype=np.int16)
        self.hist_array = np.full((np.r_[self.dims, self.hist_buffer, self.n_obj]), np.nan)
        self.neighbor_cache = self.get_neighbor_cache(reach=self.reach)

    def tdomino(self, objs, anchors):        
        return np.prod(np.sum(objs >= anchors,axis=0))  

    # -- Archive Competition Based on Tournmanent Dominance Objective ----- -- #
    def add(self, solution, objective_value, behavior_values, metadata=None):
        """Attempts to insert a new solution into the archive. New solutions are
        only inserted if their 'T-DominO' score is higher than the already
        existing solution.

        * Objectives are stored in the elite.meta dict under the key ['m_obj']
        """
        self._state["add"] += 1
        solution = np.asarray(solution)
        behavior_values = np.asarray(behavior_values)
        objective_value = self.dtype(objective_value)
        index = self.get_index(behavior_values)
        invader_obj = metadata[0]['m_obj']
        invader_obj = np.expand_dims(np.array(invader_obj),axis=0)

        # -- Compute T-Domino of both solutions --------------------------- -- #
        native = self.elite_with_behavior(behavior_values)
        anchor = self.anchor_neigh_hist(index)
        anchor = np.r_[invader_obj, anchor]
        if anchor is None: # no solutions added yet
            invader_fit = 1
        else:
            invader_fit = self.tdomino(invader_obj, anchor) 
            if native.sol is not None: # occupied bin
                native_obj  = native.meta[0]['m_obj']
                native_fit  = self.tdomino(native_obj, anchor)      
            else:
                native_fit = 0

            # Update objective values for insertion
            self._objective_values[index] = native_fit
            old_objective   = native_fit
            objective_value = invader_fit            

        # -- Attempt to Add Invader --------------------------------------- -- #
        was_inserted, already_occupied = self.insert(
            index, solution, objective_value, behavior_values, metadata)

        if was_inserted:
            self.add_to_history(index, invader_obj)

        # -- Update Archive Stats ----------------------------------------- -- #
        if was_inserted and not already_occupied:
            self._add_occupied_index(index)
            status = AddStatus.NEW
            value = objective_value
            self._stats_update(self.dtype(0.0), objective_value)
        elif was_inserted and already_occupied:
            status = AddStatus.IMPROVE_EXISTING
            value = objective_value - old_objective
        else:
            status = AddStatus.NOT_ADDED
            value = objective_value - old_objective
        return status, value 

    def add_to_history(self, index, obj_vals):
        buff_idx = self.i_hist[index]%(self.hist_buffer)
        all_indices = tuple((*index, buff_idx))
        self.hist_array[all_indices] = obj_vals
        self.i_hist[tuple(index)] += 1

    # ------ Neighborhood T-Domino ------------------------------------------- #    
    def anchor_neigh_hist(self, index):
        """"Returns anchor points as neighbor history"""
        neigh_idx = self.get_occ_neigh(index)
        A = self.hist_array[tuple(neigh_idx.T)]
        anchor = A.reshape(-1,A.shape[-1])
        return anchor
    
    def get_occ_neigh(self, idx):
        """Use Cache to get Occupied Neighbors"""
        #idx = self.get_index(desc)
        n_idx = self.neighbor_cache[idx[0]][idx[1]]
        occupied_neighbors = self._occupied[n_idx[:,0],n_idx[:,1]]>0
        n_idx = n_idx[occupied_neighbors]
        return n_idx 
 
    # ------ Neighborhood Caching -------------------------------------------- #
    def get_neighbor_cache(self, reach=None):
        """Precompute all neighbors"""
        neighbor_cache = []
        for i in range(self._dims[0]):
            neighbor_row = []
            for j in range(self._dims[1]):
                neighbor_row.append(self.get_neighbors((i,j), reach))
            neighbor_cache.append(neighbor_row)
        return neighbor_cache

    def get_neighbors(self, start, reach=None):
        """Determine neighbors of one cell"""
        grid = np.ones(self._dims)
        if reach is None:
            reach = np.ceil(np.array(grid.shape)/4).astype(int)
        x = np.arange(-reach[0],reach[0]+1)
        y = np.arange(-reach[1],reach[1]+1)
        xx, yy = np.meshgrid(x, y)
        offset = np.c_[xx.flatten(),yy.flatten()]
        n_idx = start+offset

        # Make sure offset is in bounds
        n_idx = n_idx[np.all(n_idx>=0,axis=1),:]
        n_idx = n_idx[n_idx[:,0]<grid.shape[0],:]
        n_idx = n_idx[n_idx[:,1]<grid.shape[1],:]
        return n_idx

    # ------ Export ---------------------------------------------------------- #
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
        else:
            return NotImplementedError
            return np.rollaxis(np_archive,2), np.rollaxis(meta_archive,2)
                