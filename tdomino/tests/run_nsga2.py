import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pathlib import Path
import shutil


def run_nsga2(d, p, rep=None):
    if rep is None:
        log_dir = Path(f'log/{p["task_name"]}/{p["alg_name"]}')
    else:
        log_dir = Path(f'log/{p["task_name"]}/{p["alg_name"]}/{rep}')

    if log_dir.exists() and log_dir.is_dir():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True,exist_ok=True)
    n_gens = p['n_gens']
    plot_mod = p['plot_rate']
    save_mod = p['save_rate']
    pop_size = np.prod(p['grid_res'])

    ### -- ###
    algorithm = NSGA2(pop_size=pop_size, n_offsprings=p['n_batch'])
    algorithm.setup(d, seed=1, termination=('n_gen', n_gens))

    for k in range(n_gens):
        algorithm.next()
        print(algorithm.n_gen)
        F = np.vstack([ind.F for ind in algorithm.pop])
        X = np.vstack([ind.X for ind in algorithm.pop])

        if k%plot_mod==0:
            plot_front(F)          

        if k%save_mod==0:
            np.savetxt(log_dir/"x.csv", X)
            np.savetxt(log_dir/"F.csv", F)
            np.savetxt(log_dir/"x_front.csv", algorithm.result().X)
            np.savetxt(log_dir/"F_front.csv", algorithm.result().F)            

    ### -- ###
    np.savetxt(log_dir/"x.csv", X)
    np.savetxt(log_dir/"F.csv", F)
    np.savetxt(log_dir/"x_front.csv", algorithm.result().X)
    np.savetxt(log_dir/"F_front.csv", algorithm.result().F)
    plot_front(F)

    print('\n[*] Done')

def plot_front(F):
    plt.figure(figsize=(7, 5))
    plt.scatter(F[:,0], -F[:,1], s=30, edgecolors='silver')
    plt.xlabel('Obj1'); plt.ylabel('Obj2')
    plt.title("Objective Space")
    #plt.show()cond
    plt.savefig('nsga_front.png')
    #plt.close()


