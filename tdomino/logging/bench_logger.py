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

from t_domino.t_domino_logger import TDomino_Logger
from pymoo.factory import get_performance_indicator


class MOOBenchLogger(TDomino_Logger):
    def __init__(self, p, problem, save_meta=False, copy_config=None, clear=True):
        super().__init__(p, save_meta, copy_config, clear) 
        self.problem = problem

    def create_plots(self, archive):
        self.plot_metrics() # Line Plots
        self.plot_obj(archive) # Heatmaps  
        self.plot_front(archive)

    def plot_front(self, archive):
    # -- Plot Pareto Front
        fig, ax = plt.subplots(ncols=2, figsize=(8,4), dpi=100)
        # RGB Descriptor
        D = archive._behavior_values[archive._occupied]
        C = np.ones([D.shape[0], 3])*0.0
        C[:,1] = norm(D[:,0])
        C[:,2] = norm(D[:,1])
        ax[0].scatter(D[:,0], D[:,1], c=C)
        ax[0].set_title('Descriptor Space Colored By Descriptor')
        ax[0].set_xlim(self.p['desc_bounds'][0])
        ax[0].set_ylim(self.p['desc_bounds'][1])
        ax[0].grid(color="silver", alpha=.8, linewidth=1)
        ax[0].set_xticks(archive._boundaries[0])
        ax[0].set_yticks(archive._boundaries[1])

        # Pareto front
        true_front = self.problem.pareto_front()
        ax[1].plot(true_front[:,0][::5], true_front[:,1][::5], 'r*')
        ax[0].set_xlim(self.p['desc_bounds'][0])
        ax[0].set_ylim(self.p['desc_bounds'][1])

        meta = archive._metadata.flatten()
        meta = meta[archive._occupied.flatten()]
        O = np.vstack([m[0] for m in meta]) 
        ax[1].scatter(-O[:,0], -O[:,1], c=C)
        ax[1].set_title('Objective Space Colored By Descriptor')

        # if self.problem.name()[-1] == '1':
        #     ax[1].set_xlim([0,1])
        #     ax[1].set_ylim([0,2])
        # if self.problem.name()[-1] == '2':
        #     ax[1].set_xlim([0,1])
        #     ax[1].set_ylim([0,2] )
        # if self.problem.name()[-1] == '3':
        #     ax[1].set_xlim([-0.05,1.05])
        #     ax[1].set_ylim([-0.95,3])
        # if self.problem.name()[-1] == '4':
        #     ax[1].set_xlim([-0.05,1.05])
        #     ax[1].set_ylim([-1,25])
        # if self.problem.name()[-1] == '6':
        #     ax[1].set_ylim([-0.3,6.5])                           
        
        fig.savefig(str(self.log_dir / f"Pareto_Front.png"))   
        plt.clf(); plt.close()


def norm(a):
    """ Scale to 0/1"""
    b = (a - np.min(a))/np.ptp(a)
    return b

