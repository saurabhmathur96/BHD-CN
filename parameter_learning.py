from typing import List
import numpy as np
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from dataset import Dataset, GroupedDataset
from cutset_network import CutsetNetworkNode, LeafNode, depth_first_order
from estimators import HDirProbabilityEstimator, ProbabilityEstimator


def estimate_parameters_hdir(cnet: CutsetNetworkNode, D: GroupedDataset, s: float = 1, s0: float = 1):
    n_groups = D.r[D.group_col]
    parameters = [[] for _ in range(n_groups)]
    for node, Di in depth_first_order(cnet, D):
        est = HDirProbabilityEstimator(Di, s=s, s0=s0)
        if isinstance(node, LeafNode):
            cpds = [[] for _ in range(n_groups)]
            for each in node.bn.nodes:
                parents = node.bn.get_parents(each)
                if len(parents) == 0:
                    prob = est.probability([each])
                    for g in range(n_groups):
                        card = node.r[each]
                        values = [[prob[g][value]] for value in np.ndindex(card)]
                        cpd = TabularCPD(each, card, values)
                        cpd.normalize()
                        cpds[g].append(cpd)   
                else:
                    conditional = est.conditional(each, parents)
                    for g in range(n_groups):
                        shape = conditional[g].shape
                        axes = np.arange(len(shape))
                        values = np.transpose(conditional[g], (axes[-1], *axes[:-1]))
                        cpd = TabularCPD(each, shape[-1], values, evidence=parents,evidence_card=shape[:-1])
                        cpd.normalize()
                        cpds[g].append(cpd)
            for g in range(n_groups):
                parameters[g].append(cpds[g])
        else:
            prob = est.probability([node.v])
            for g in range(n_groups):
                weights = [prob[g][value] for value in np.ndindex(node.r[node.v])]
                parameters[g].append(weights)
    return parameters


def estimate_parameters(cnet: CutsetNetworkNode, D: Dataset, s: float = 1):
    parameters = []
    for node, Di in depth_first_order(cnet, D):
        if isinstance(node, LeafNode):
            est = BayesianEstimator(node.bn, D.as_df())
            cpds = est.get_parameters(prior_type="dirichlet", pseudo_counts=s)
            parameters.append(cpds)
        else:
            est = ProbabilityEstimator(D, s=s)
            prob = est.probability([node.v])
            weights = [prob[value] for value in np.ndindex(node.r[node.v])]
            parameters.append(weights)
    return parameters

def set_parameters(cnet: CutsetNetworkNode, parameters: List):
    # sets parameters inplace
    for node, param in zip(depth_first_order(cnet), parameters):
        if isinstance(node, LeafNode):
            node.bn.remove_cpds(*node.bn.get_cpds())
            node.bn.add_cpds(*param)
        else:
            node.weights = param
    return cnet