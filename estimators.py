from collections import defaultdict
from typing import *
from typing import List

import numpy as np

from dataset import Dataset, GroupedDataset
from hdir import HierarchicalDirichletModel

class ProbabilityEstimator:
    def __init__(self, D: Dataset, s: float = 1):
        self.D = D 
        self.s = s

    def probability(self, variables: List[str]):
        counts = self.D.counts(variables)
        total = sum(counts.values())
        M = np.prod([self.D.r[v] for v in variables])
        prob = { value: (count + self.s)/(total + M*self.s) for value, count in counts.items() } 
        default = (self.s)/(total + M*self.s)

        return defaultdict(lambda: default, prob)

    def conditional(self, variable: str, parents: List[str]):
        numerator = self.probability([variable, *parents])
        denominator = self.probability(parents)

        parent_shape = [self.D.r[each] for each in parents]
        prob = np.zeros((np.prod(parent_shape), self.D.r[variable]))
        for i, parent_value in enumerate(np.ndindex(*parent_shape)):
            for j in range(self.D.r[variable]):
                prob[i, j] = numerator[(j, *parent_value)] / denominator[parent_value]
        return prob
    
class HDirProbabilityEstimator(ProbabilityEstimator):
    def __init__(self, D: GroupedDataset, s: float = 1, s0: float = 1):
        super().__init__(D, s)
        self.s0 = s0 
    
    def estimate_model(self, variables: List[str]):
        X_card = [self.D.r[v] for v in variables]
        n_groups = self.D.r[self.D.group_col]
        df = self.D.as_df(with_group=True)
        X, groups = df.drop([self.D.group_col], axis=1), df[self.D.group_col]
        X = X[variables].to_numpy().reshape((-1, len(variables)))
        groups = groups.to_numpy().reshape(-1)
        model = HierarchicalDirichletModel(X_card, n_groups, self.s, self.s0)
        model.fit(X, groups)
        return model

    def probability(self, variables: List[str]):
        model = self.estimate_model(variables) 
        X_card = [self.D.r[v] for v in variables]
        n_groups = self.D.r[self.D.group_col]      
        return [ { values: model.theta[values][group] for values in np.ndindex(*X_card) } 
                for group in range(n_groups) ] # (X_card, n_groups)
    
    def conditional(self, variable: str, parents: List[str]):
        numerator = self.probability([variable, *parents])
        denominator = self.probability(parents)
        
        parent_shape = [self.D.r[each] for each in parents]
        n_groups = self.D.r[self.D.group_col]  
        prob = np.zeros((n_groups, np.prod(parent_shape), self.D.r[variable]))
        for g in range(n_groups):
            for i, parent_value in enumerate(np.ndindex(*parent_shape)):
                for j in range(self.D.r[variable]):
                    prob[g, i, j] = numerator[g][(j, *parent_value)] / denominator[g][parent_value]
        return prob
    
