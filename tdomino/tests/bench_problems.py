import numpy as np
from math import pi
from pymoo.factory import get_problem

def rast_moo(x):
    x1 = x
    x2 = x-2.2     
    f1 = 10 * x1.shape[0] + (x1 * x1 - 10 * np.cos(2 * pi * x1)).sum()
    f2 = 10 * x2.shape[0] + (x2 * x2 - 10 * np.cos(2 * pi * x2)).sum()
    return -f1 + 2*x.shape[0]**2, (-f2 + 2*x.shape[0]**2)

def get_obj_fcn(name):
    if name == 'rast':
        return rast_moo
    if name == 'dtlz3-5':
        problem = get_problem('dtlz3', 10, 5)
        return problem.evaluate        
    else:
        problem = get_problem(name)
        return problem.evaluate