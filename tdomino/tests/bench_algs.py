import numpy as np
# Test Domain
from bbq.domains._domain import RibsDomain
from bbq.utils import scale
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_problem
from tdomino.grid_archive import GridArchive
from tdomino.tdomino_grid import TDominoGrid


# -- T-Domino ------------------------------------------------------------- -- #
class TDomino(RibsDomain):
    def __init__(self, objective_fcn, p):
        self.obj_fcn = objective_fcn
        self.archive_type = TDominoGrid
        RibsDomain.__init__(self, p)
        self.param_bounds = p['param_bounds']                           
    
    def _fitness(self, x):   
        return self.obj_fcn(x)

    def _desc(self, x):
        # """ Use first parameters"""
        #return x[5:7] # for dtlz3
        return x[:2] # for others

    def express(self, x):
        return scale(x, self.param_bounds)

    def batch_eval(self, X, evaluator=None):
        "Evaluate NxM matrix of N solutions with M variables"
        pheno = self.express(X)
        obj = np.zeros(len(X))          # T-dominO fitness is calculated on add
        m_obj = [ {'m_obj':self._fitness(x)}  for x in pheno]
        desc  = [self._desc(x) for x in pheno]
        meta = list(zip(m_obj,pheno))
        return obj, desc, meta


# -- MAP-Elites on 1 Objective only --------------------------------------- -- #
class ME_Single(TDomino):
    def __init__(self, objective_fcn, p):
        super().__init__(objective_fcn, p)
        self.archive_type = GridArchive
    
    def batch_eval(self, X, evalator=None):
        "Evaluate NxM matrix of N solutions with M variables"
        pheno = self.express(X)
        m_obj = [ {'m_obj':self._fitness(x)}  for x in pheno]
        obj = np.vstack([m['m_obj'][0] for m in m_obj])
        desc  = [self._desc(x) for x in pheno]
        meta = list(zip(m_obj,pheno))
        return obj, desc, meta

# -- MAP-Elites with sum of objectives as fitness-------------------------- -- #
class ME_Sum(TDomino):
    def __init__(self, objective_fcn, p):
        super().__init__(objective_fcn, p)
        self.archive_type = GridArchive
    
    def batch_eval(self, X, evalator=None):
        "Evaluate NxM matrix of N solutions with M variables"
        pheno = self.express(X)
        m_obj = [ {'m_obj':self._fitness(x)}  for x in pheno]
        obj = [sum(list(moo)) for moo in self.weighted_obj(m_obj)] 
        desc  = [self._desc(x) for x in pheno]
        meta = list(zip(m_obj,pheno))
        return obj, desc, meta

    def weighted_obj(self, m_obj):
        m_obj = np.vstack([m['m_obj'] for m in m_obj])
        factor = 10**np.arange(m_obj.shape[1])
        return m_obj*factor


# -- NSGA-II -------------------------------------------------------------- -- #
class NSGA2(ElementwiseProblem):
    def __init__(self, objective_fcn, p):
        super().__init__(n_var=p['n_dof'],
                         n_obj=p['n_obj'],
                         n_constr=0,
                         xl=p['param_bounds'][0],
                         xu=p['param_bounds'][1]) 
        self.fitness = objective_fcn              

    def _evaluate(self, x, out, *args, **kwargs):
        f = self.fitness(x)
        #f1, f2 = self.fitness(x)
        out["F"] = [-obj for obj in f]
