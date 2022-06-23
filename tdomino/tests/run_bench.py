#from ribs.emitters import GaussianEmitter, IsoLineEmitter
from bbq.emitters._standard_emitters import GaussianEmitter, IsoLineEmitter, ImprovementEmitter
from tdomino.logging.tdomino_logger import TDomino_Logger as Logger
from tdomino.tests.bench_algs import TDomino

from bbq.map_elites import map_elites
from tdomino.tests.run_nsga2 import run_nsga2

CONFIG_PATH = '../../config/'

def me_benchmark(Problem, task, p, name='tdomino', rep=0):
    d = Problem(task, p)
    p['exp_name'] = name
    logger = Logger(p, clear=False, rep=rep)
    archive = map_elites(d, p, logger, 
                         emitter_type=ImprovementEmitter, 
                         archive_type=d.archive_type)
    print("Done")

def moo_benchmark(Problem, task, p, name='nsga', rep=0):
    p['alg_name'] = name
    d = Problem(task, p)
    run_nsga2(d, p, rep=rep)
    print("Done: "+p['alg_name'])

if __name__ == '__main__':
    from bench_algs import NSGA2, ME_Single, ME_Sum, TDomino
    from bench_problems import get_obj_fcn
    from bbq.utils import create_config

    # -- Benchmark Problems -- #
    base_config = CONFIG_PATH+'rast.yaml'   
    #base_config = 'config/zdt3.yaml'

    # -- Experiment Settings -- #
    #exp_config = 'config/test.yaml'
    exp_config = CONFIG_PATH+'smoke.yaml'

    p = create_config(base_config, exp_config)
    task = get_obj_fcn(p['task_name'])

    me_benchmark(TDomino,   task, p, name='tdomino')  
    #me_benchmark(ME_Single, task, p, name='me_single')  
    #me_benchmark(ME_Sum,    task, p, name='me_sum') 
    #moo_benchmark(NSGA2,    task, p) 
