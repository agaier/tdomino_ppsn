from bbq.emitters._standard_emitters import ImprovementEmitter
from bbq.map_elites import map_elites
from tdomino.logging.tdomino_logger import TDomino_Logger as Logger
from tdomino.tests.bench_algs import TDomino


def me_benchmark(Problem, task, p, name='tdomino', rep=0):
    d = Problem(task, p)
    p['exp_name'] = name
    logger = Logger(p)
    archive = map_elites(d, p, logger, 
                         emitter_type=ImprovementEmitter, 
                         archive_type=d.archive_type)
    print("Done")

if __name__ == '__main__':
    from bbq.utils import create_config
    from tdomino.tests.bench_algs import TDomino
    from tdomino.tests.bench_problems import get_obj_fcn

    # -- Benchmark Problems -- #
    base_config = 'config/rast.yaml'   
    exp_config  = 'config/smoke.yaml'

    p = create_config(base_config, exp_config)
    task = get_obj_fcn(p['task_name'])

    me_benchmark(TDomino, task, p, name='tdomino')  
