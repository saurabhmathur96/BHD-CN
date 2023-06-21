from collections import defaultdict
from typing import *

import numpy as np

from dataset import Dataset


class ProbabilityEstimator:
    def __init__(self, D: Dataset, s: float = 1):
        self.D = D 
        self.s = s

    def probability(self, variables: List[str]):
        counts = self.D.counts(variables)
        total = counts.values().sum()
        M = np.prod([self.D.r[v] for v in variables])
        prob = { value: (count + self.s)/(total + M*self.s) for value, count in counts.items() } 
        default = (self.s)/(total + M*self.s)

        return defaultdict(lambda: default, prob)
    
    def conditional(self, variable: str, parents: List[str]):
        numerator = self.probability([variable, *parents])
        denominator = self.probability(parents)

        parent_shape = [self.D.r[each] for each in parents]
        prob = np.zeros(np.prod(parent_shape), self.D.r[variable])
        for i, parent_value in enumerate(np.ndindex(*parent_shape)):
            for j in range(self.D.r[variable]):
                prob[i, j] = numerator[(j, *parent_value)] / denominator[parent_value]
        return prob


class HDirProbabilityEstimator(ProbabilityEstimator):
    pass