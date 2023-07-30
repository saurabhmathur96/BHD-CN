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

    def __call__(self, D: Dataset, log: bool = True):
        score = log_likelihood_score(self.bn, D.as_df())
        return score if log else np.exp(score)
    
    def __str__(self) -> str:
        return f"LeafNode({', '.join(self.scope)})"
    
    @property
    def n_parameters(self):
        return np.sum([np.prod(cpd.cardinality[1:])*(cpd.variable_card-1) 
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

    def __call__(self, D: Dataset, log: bool = True):
        score = 0
        for weight, child, split in zip(self.weights, self.children, D.split(self.v)):
            if len(split) == 0: continue
            score += len(split)*np.log(weight)
            score += child(split, log = True)
        return score if log else np.exp(score)
    
    def __str__(self) -> str:
        return f"OrNode({self.v})"
    
    @property
    def n_parameters(self):
        return len(self.weights)-1

def depth_first_order(node: CutsetNetworkNode, D = None):
    yield (node, D) if D is not None else node 

    if isinstance(node, LeafNode):
        return

    if D is not None:    
        for child, split in zip(node.children, D.split(node.v)):
            for each in depth_first_order(child, split):
                yield each
    else:
        for child in node.children:
            for each in depth_first_order(child):
                yield each



from anytree import NodeMixin, RenderTree
from anytree.exporter import DotExporter
class WNode(NodeMixin):
    def __init__(self, name, text, parent=None, weight=None):
        super(WNode, self).__init__()
        self.name = name
        self.text = text
        self.parent = parent
        self.weight = weight 
    def _post_detach(self, parent):
        self.text = None
        self.weight = None

def build_tree(cnet):
    leaf_id = 1
    node_id = 1
    def _build_tree(cnet, parent=None, weight=None):
        nonlocal leaf_id, node_id
        if isinstance(cnet, OrNode):
            current = WNode(f"N{node_id}", cnet.v, parent, weight)
            node_id += 1
            for weight, child in zip(cnet.weights, cnet.children):
                _build_tree(child, current, weight)
        else:
            current = WNode(f"N{node_id}", f"T{leaf_id}", parent, weight)
            node_id += 1
            leaf_id += 1
        return current
    return _build_tree(cnet)

def to_text(cnet):
    lines = []
    tree = build_tree(cnet)
    for pre, _, node in RenderTree(tree):
        if node.weight is None:
            lines.append(f"{pre} {'[' + node.text + ']' if not node.is_leaf else node.text}")
        else:
            lines.append(f"{pre} ({node.weight:.4f}) {'[' + node.text + ']' if not node.is_leaf else node.text}")
    return "\n".join(lines)

def to_graphviz(cnet):
    lines = []
    tree = build_tree(cnet)
    exporter = DotExporter(tree,
                 nodenamefunc=lambda node: node.name,
                 nodeattrfunc=lambda node: f'label="{node.text}",shape=box' if node.is_leaf else f'label="{node.text}"',
                 edgeattrfunc=lambda parent, child: f"label={child.weight:.2f}"
    )
    for line in exporter:
        lines.append(line)
    
    return "\n".join(lines)
