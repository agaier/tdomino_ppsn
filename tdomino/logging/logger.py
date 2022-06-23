from logging import raiseExceptions
from ribs.visualize import grid_archive_heatmap

import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil
import numpy as np
from bbq.logging import RibsLogger
from matplotlib import colors
from matplotlib.ticker import FixedLocator, FixedFormatter


class TDominoLogger(RibsLogger):
    def __init__(self, p, save_meta=False, copy_config=None, clear=True):
        super().__init__(p, save_meta, copy_config, clear)

    
    def update_metrics(self, archive, itr):
        ''' Adds current iterations metrics to running record '''   
        meta = archive._metadata.flatten()
        meta = meta[archive._occupied.flatten()]
        objs = np.vstack([m[0] for m in meta])

        self.metrics["Archive Size"]["itrs"].append(itr)
        self.metrics["Archive Size"]["vals"].append(archive.stats.num_elites)
        self.metrics["Mean Fitness"]["itrs"].append(itr)
        self.metrics["Mean Fitness"]["vals"].append(np.mean(objs,axis=0).tolist())
        self.metrics["QD Score"]["itrs"].append(itr)
        self.metrics["QD Score"]["vals"].append(np.sum(objs,axis=0).tolist())  


    def log_metrics(self, domain, archive, itr, time, save_all=False):
        ''' Calls all logging and visualization functions '''
        self.update_metrics(archive, itr)
        if (itr%self.p['print_rate'] == 0) or self.p['print_rate'] == 1:
            self.print_metrics(archive, itr, self.p['n_batch']*self.p['n_emitters'])
            with (self.log_dir / f"metrics.json").open("w") as file:
                json.dump(self.metrics, file, indent=2)        

        if (itr%self.p['plot_rate']==0) or save_all:
            self.plot_metrics()
            objs = objs_from_archive(archive)

            # -- Plot t-Dominance
            t_domino = archive._objective_values
            t_domino[np.logical_not(archive._occupied)] = np.nan
            fig, ax = plt.subplots(figsize=(4,4), dpi=100)
            self.plot_scalar(t_domino, ax = ax)
            ax.set_title('T_Domino')
            fig.savefig(str(self.log_dir / f"MAP_tdomino.png"))   
            plt.clf(); plt.close()

            # -- Plot Pareto Front
            fig, ax = plt.subplots(ncols=2, figsize=(8,4), dpi=100)
            # RGB Descriptor
            D = archive._behavior_values[archive._occupied]
            C = np.ones([D.shape[0], 3])*0.0
            C[:,1] = norm(D[:,0])
            C[:,2] = norm(D[:,1])
            ax[0].scatter(D[:,0], D[:,1], c=C)
            ax[0].set_title('Descriptor Space Colored By Descriptor')


            # Pareto front
            meta = archive._metadata.flatten()
            meta = meta[archive._occupied.flatten()]
            O = np.vstack([m[0] for m in meta]) 
            ax[1].scatter(O[:,0], O[:,1], c=C)
            ax[1].set_title('Objective Space Colored By Descriptor')

            fig.savefig(str(self.log_dir / f"Pareto_Front.png"))   
            plt.clf(); plt.close()


            # -- Plot Objectives
            fig, ax = plt.subplots(ncols=2, figsize=(8,4), dpi=100)
            self.plot_scalar(objs[0], ax = ax[0], colorbar=False)
            self.plot_scalar(objs[1], ax = ax[1], colorbar=False)
            ax[0].set_title('Objective 1')
            ax[1].set_title('Objective 2')
            fig.suptitle("Objective Values")
            fig.savefig(str(self.log_dir / f"MAP_obj.png"))   
            plt.clf(); plt.close()

        if (itr%self.p['save_rate']==0) or save_all:
            #self.save_archive(archive, self.log_dir, export_meta=self.save_meta)
            self.save_archive(archive, itr=itr, export_meta=self.save_meta)


    def plot_scalar(self, Z, ax=None, colorbar=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,8), dpi=100)
        Z = np.rollaxis(Z,1)
        img = ax.imshow(Z, cmap='YlGnBu')
        ax.invert_yaxis()
        if colorbar:
            cbar = plt.colorbar(img)
        ax = self.format_ticks(ax, Z)
        return ax

    def format_ticks(self, ax, grid, xlabels=None, ylabels=None):
        map_x, map_y = grid.shape
        xticks = -0.5+np.arange(map_x)
        yticks = -0.5+np.arange(map_y) 
        
        if xlabels is None:
            xlabels = ['']*map_x
            xlabels[1], xlabels[-1] = 'Small', 'Big'
            
        if ylabels is None:
            ylabels = ['']*map_y
            ylabels[1], ylabels[-1] = 'Small', 'Big'    

        x_formatter, x_locator = FixedFormatter(xlabels), FixedLocator(xticks)
        y_formatter, y_locator = FixedFormatter(ylabels), FixedLocator(yticks)

        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)

        ax.yaxis.set_major_locator(y_locator)
        ax.yaxis.set_major_formatter(y_formatter)

        ax.grid(color="silver", alpha=.8, linewidth=1)
        ax.tick_params(direction="in", width=1, length=5)

        # Set Axis Labels
        desc_labels = self.p['desc_labels']
        ax.set(xlabel= desc_labels[0][0],
                ylabel= desc_labels[1][0])

        return ax        




def objs_from_archive(archive):
    M = archive._metadata
    occ = archive._occupied
    objs = np.full([M.shape[0],M.shape[1],2],np.nan)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if occ[i,j]:
                objs[i,j,:] = M[i,j][0]                

    objs = np.rollaxis(objs,2)
    return objs


def objs_from_meta(M):
    objs = np.full([M.shape[0],M.shape[1],2],np.nan)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if type(M[i,j]) == np.ndarray:
                objs[i,j,:] = M[i,j][0]                

    objs = np.rollaxis(objs,2)
    return objs

def norm(a):
    """ Scale to 0/1"""
    b = (a - np.min(a))/np.ptp(a)
    return b


# def objs_from_meta(M):
#     objs = np.full([M.shape[0],M.shape[1],2],np.nan)
#     for i in range(M.shape[0]):
#         for j in range(M.shape[1]):
#             if type(M[i,j,0]) == np.ndarray:
#                 objs[i,j,:] = M[i,j,0][0]                

#     objs = np.rollaxis(objs,1)
#     objs = np.rollaxis(objs,2)
#     return objs