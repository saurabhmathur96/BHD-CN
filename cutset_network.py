from typing import *

import numpy as np
from pgmpy.inference.ExactInference import (BeliefPropagation,
                                            VariableElimination)
from pgmpy.metrics.metrics import log_likelihood_score

from dataset import Dataset


class CutsetNetworkNode:
    def __init__(self, scope, r):
        self.scope = scope
        self.r = r 

    def marginal(self, q):
        raise NotImplementedError
    
    def __call__(self, D: Dataset, log: bool):
        raise NotImplementedError
    
    @property
    def n_parameters(self):
        raise NotImplementedError

class LeafNode(CutsetNetworkNode):
    def __init__(self, scope, r, bn):
        super().__init__(scope, r)
        self.bn = bn 
        try:
            self.inference = BeliefPropagation(bn)
            self.inference.calibrate()
        except ValueError:
            self.inference = VariableElimination(bn)

    def marginal(self, q: Dict[str, int]):
        if not q: return 1
        variables = list(q.keys())
        result = self.inference.query(variables, show_progress = False)
        return result.get_value(**q)

    def __call__(self, D: Dataset, log: bool):
        score = log_likelihood_score(self.bn, D.as_df())
        return score if log else np.log(score)
    
    def __str__(self) -> str:
        return f"LeafNode({', '.join(self.scope)})"
    
    def n_parameters(self):
        return np.sum([np.prod(cpd.evidence_card)*(cpd.variable_card-1) 
                       for cpd in self.bn.get_cpds()])

class OrNode(CutsetNetworkNode):
    def __init__(self, scope, r, v, weights, children):
        super().__init__(scope, r)
        self.v = v
        self.weights = weights
        self.children = children
    
    def marginal(self, q: Dict[str, int]):
        if not q: return 1
        if self.v in q:
            value = q.pop(self.v)
            rest = self.children[value].marginal(q)
            return self.weights[value]*rest
        else:
            rest = np.array([child.marginal(q) for child in self.children])
            return np.dot(rest, self.weights)

    def __call__(self, D: Dataset, log: bool):
        score = np.sum(np.log(self.weights))
        for child, split in zip(self.children, D.split(self.v)):
            score += child(split, log = True)
        return score if log else np.log(score)
    
    def __str__(self) -> str:
        return f"OrNode({self.v})"
    
    def n_parameters(self):
        return len(self.weights)-1

def depth_first_order(node: CutsetNetworkNode):
    if isinstance(node, LeafNode):
        yield node 
    
    yield node
    for child in node.children:
        for descendence in depth_first_order(child):
            yield descendence

