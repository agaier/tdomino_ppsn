import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from bbq.logging import RibsLogger
from humanfriendly import format_timespan
from tdomino.logging.metric_compare import get_stats, plot_comparison
from tdomino.plotting import map_to_image, plot_ys, set_map_grid
from tdomino.tdomino_grid import TDominoGrid

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

class TDomino_Logger(RibsLogger):
    def __init__(self, p, save_meta=False, copy_config=True, clear=True, rep=0):
        super().__init__(p, save_meta=save_meta, copy_config=copy_config, clear=clear, rep=rep)
        self.baseline = None #'/Users/gaiera/Desktop/t-Domino/tdomino/run/log/rast/me_sum'

    def create_plots(self, archive):
        self.plot_metrics() # Line Plots
        self.plot_obj(archive) # Heatmaps  

        # -- With baseline
        if self.baseline is not None:
            exp = [self.baseline, self.log_dir.parent]
            qd = get_stats(exp, 'QD Score')
            ax = plot_comparison(qd, obj_labels=None)
            fname = self.log_dir/'QD.png'
            plt.savefig(fname,bbox_inches='tight')
            print("Done")

    def plot_metrics(self):
        ''' Creates a plot for each matrix to visualize during and after run ''' 
        metric = 'QD Score'
        fig, ax = plt.subplots(figsize=(8,5),dpi=150)
        x = np.array(self.metrics[metric]['itrs'])
        ys = np.c_[np.array(self.metrics['Archive Size']['vals'][1:]),
                    np.array(self.metrics[metric]['vals'])]                  
        ax = plot_ys(x, ys, ["Archive Size"]+self.p['obj_labels'], ax=ax)

        fname = str(self.log_dir / f"{metric.lower().replace(' ', '_')}.png")
        plt.savefig(fname,bbox_inches='tight')
        plt.clf(); plt.close()


    def update_metrics(self, archive, itr):
        ''' Adds current iterations metrics to running record '''   
        meta = archive._metadata.flatten()
        meta = meta[archive._occupied.flatten()]
        objs = np.vstack([m[0]['m_obj'] for m in meta])

        self.metrics["Archive Size"]["itrs"].append(itr)
        self.metrics["Archive Size"]["vals"].append(archive.stats.num_elites)
        self.metrics["Mean Fitness"]["itrs"].append(itr)
        self.metrics["Mean Fitness"]["vals"].append(np.mean(objs,axis=0).tolist())
        self.metrics["QD Score"]["itrs"].append(itr)
        self.metrics["QD Score"]["vals"].append(np.sum(objs,axis=0).tolist()) 

    def print_metrics(self, archive, itr, eval_per_iter, time):
        ''' Print metrics to command line '''    
        moo_mean = np.array(self.metrics['Mean Fitness']['vals'][-1])
        moo_qd = np.array(self.metrics['QD Score']['vals'][-1])
        print(f"Iter: {itr}" \
            +f" | Eval: {itr*eval_per_iter}" \
            +f" | Size: {archive.stats.num_elites}" \
            +f" | QD: {np.round(moo_qd)}" \
            +f" | Time/Itr: {format_timespan(time)}")    


    def plot_obj(self, archive):
        if type(archive) == TDominoGrid:
            archive.refresh_tdomino()
            labels = ['T_Domino'] + self.p['obj_labels']
        else:
            labels = ['Fitness'] + self.p['obj_labels']
        A = np.rollaxis(archive.as_numpy(),-1)
        A = A[:(1+len(self.p['obj_labels']))]

        # Prep objectives            
        val = {}
        for i, label in enumerate(labels):
            val[label] = A[i]

        # Do the plotting
        #fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(8,2),dpi=150)
        fig,ax = plt.subplots(nrows=1,ncols=self.p['n_obj'],figsize=(8,2),dpi=150)
        ax = ax.flatten()

        for i, metric in enumerate(labels):
            ax[i] = map_to_image(val[metric], ax=ax[i])
            ax[i] = set_map_grid(ax[i], val[metric], **self.p) 
            ax[i].set(xlabel = self.p['desc_labels'][0], 
                        ylabel= self.p['desc_labels'][1], 
                        title=metric)
        plt.subplots_adjust(hspace=0.4)
        fig.savefig(str(self.log_dir / f"MAP_Objectives.png"))
        plt.clf(); plt.close()






    # def log_metrics(self, opt, d, itr, time, save_all=False):
    #     ''' Calls all logging and visualization functions '''
    #     archive = opt.archive
    #     emitter = opt.emitters
    #     self.update_metrics(archive, itr)
    #     if (itr%self.p['print_rate'] == 0) or self.p['print_rate'] == 1:
    #         n_evals = self.p['n_batch']*self.p['n_emitters']
    #         self.print_metrics(archive, itr, n_evals, time)           
    #         # Save metrics
    #         with (self.log_dir / f"metrics.json").open("w") as file:
    #             json.dump(self.metrics, file, indent=2)        

    #     if (itr%self.p['plot_rate']==0) or save_all:
    #         # -- Plot Objectives Line Plot
    #         self.create_plots(archive)
        
    #     if (itr%self.p['save_rate']==0) or save_all:
    #         self.save_archive(archive, itr=itr, export_meta=self.save_meta)