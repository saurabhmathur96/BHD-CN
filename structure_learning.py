import numpy as np
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
from sklearn.metrics import mutual_info_score

from cutset_network import CutsetNetworkNode, LeafNode, OrNode
from dataset import Dataset


def learn_chow_liu_leaf(D: Dataset) -> LeafNode:
    df = D.as_df()
    dag = TreeSearch(df).estimate(estimator_type="chow-liu")
    bn = BayesianNetwork(dag)
    bn.fit(df, prior_type = "dirichlet", equivalent_sample_size=1)
    return LeafNode(D.scope, D.r, bn)


from itertools import combinations


def select_mi_cut(D: Dataset):
    df = D.to_df()
    MI = np.zeros((df.shape[1], df.shape[1]))
    for (i, u), (j, v) in combinations(enumerate(df.columns), r=2):
        MI[i, j] = MI[j, i] = mutual_info_score(df[u], df[v])
    
    scores = np.sum(MI - np.diag(np.diag(MI)), axis=1)
    i = scores.argmax()
    return D.scope[i]

def learn_structure(D: Dataset, learn_leaf, select_best_cut, min_variables: int = 5) -> CutsetNetworkNode:
    if len(D.scope) < min_variables:
        leaf = learn_leaf(D)
        return leaf 
    else:
        best_cut = select_best_cut(D)
        if best_cut is None:
            leaf = learn_leaf(D)
            return leaf 
        else:
            children = [learn_structure(Di, learn_leaf) 
                        for Di in D.split(best_cut)]
            return OrNode(D.scope, D.r, best_cut, children)