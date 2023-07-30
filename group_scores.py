from structure_learning import *
from parameter_learning import estimate_parameters, set_parameters
class GroupBicScore(StructureScore):
    def __init__(self, D: GroupedDataset, **kwargs):
        state_names = { v: list(range(D.r[v])) for v in D.scope}
        super(GroupBicScore, self).__init__(D.as_df(with_group=True), state_names=state_names, **kwargs)
        scorers = []
        # Create a BicScore object for each group
        for dataset in D.datasets:
            if len(dataset) == 0:
                scorers.append(None)
            else:
                scorers.append(BicScore(dataset.as_df()))
        self.scorers = scorers

    def local_score(self, variable, parents):
        return np.sum([
            scorer.local_score(variable, parents)
            if scorer is not None
            else 0
            for scorer in self.scorers
        ])

def learn_group_chow_liu_leaf(D: GroupedDataset) -> LeafNode:
    df = D.as_df(with_group=True)
    state_names = { v: list(range(D.r[v])) for v in D.scope}
    if len(df) == 0:
        bn = BayesianNetwork()
        bn.add_nodes_from(D.scope)
        cpds = [
            TabularCPD(variable=v, variable_card=D.r[v], values=[[1/D.r[v]] for _ in range(D.r[v])])
            for v in D.scope
        ]
        bn.add_cpds(*cpds)
        return LeafNode(D.scope, D.r, bn)
    
    group_col = D.group_col
    n_groups = D.r[group_col]
    
    def edge_weights_fn(x1, x2):
        return sum(mutual_info_score(x1[df[group_col] == g], x2[df[group_col] == g])
                   if sum(df[group_col] == g) > 0 else 0
            for g in range(n_groups))
        

    
    dag = TreeSearch(df.drop([group_col], axis=1), state_names=state_names).estimate(estimator_type="chow-liu", 
                                                           show_progress=False,
                                                           edge_weights_fn=edge_weights_fn)
    bn = BayesianNetwork()
    bn.add_nodes_from(D.scope)
    bn.add_edges_from(dag.edges())
    
    bn.fit(df, BayesianEstimator, prior_type = "dirichlet", pseudo_counts=1, state_names=state_names)
    return LeafNode(D.scope, D.r, bn)




def group_bic_score(D: GroupedDataset, candidates: List[str]):
    scorer = GroupBicScore(D)
    score = np.zeros((len(candidates),)) 
    for i, u in enumerate(candidates):
        or_node = learn_cnet_stump(D, u, learn_group_chow_liu_leaf)
        
        children_scores = [
            GroupBicScore(split).score(child.bn)
            for split, child in zip(D.split(u), or_node.children)
        ]
        score[i] = scorer.local_score(u, []) + np.sum(children_scores)
    leaf = learn_group_chow_liu_leaf(D)
    base = scorer.score(leaf.bn)
    return score - base



class HDirScore(StructureScore):
    def __init__(self, dataset: GroupedDataset, s, s0, **kwargs):
        super(HDirScore, self).__init__(dataset.as_df(with_group=True), **kwargs)
        self.dataset = dataset
        self.est = HDirProbabilityEstimator(dataset, s, s0)
    
    def local_score(self, variable, parents):
        s = self.est.s
        n_groups = self.dataset.r[self.dataset.group_col]
        parents = list(parents)
        score = 0
        if len(parents) == 0:
          # base case
          model = self.est.estimate_model([variable])
          alpha = (model.s*model.kappa).reshape(-1)
          counts = model.counts.reshape((-1, n_groups)).T 
          """
          for g in range(n_groups):
            t1 = gammaln(s)-gammaln(s + np.sum(counts, axis = 0))
            score += np.sum(t1)
            t2 = gammaln(counts + alpha) - gammaln(alpha)
            
            score += np.sum(t2)
          """
          for g in range(n_groups):
            counts_sum = counts[g].sum(axis=-1)
            alpha_sum = alpha.sum(axis=-1,keepdims=True)

            t1 = gammaln(alpha_sum)-gammaln(alpha_sum + counts_sum)
            score += np.sum(t1)

            score += np.sum(gammaln(counts[g] + alpha))
            score -= np.sum(gammaln(alpha))
        else:
            combined_var = [*parents, variable]
            card = [self.dataset.r[v] for v in combined_var]
            # repeat for each parent
            model = self.est.estimate_model(combined_var)
            alpha = s*model.kappa[:, None]
            alpha = alpha.reshape(card)
            axes = np.arange(len(card)+1)
            counts = model.counts.reshape((*card, n_groups))
            counts = np.transpose(counts, (axes[-1], *axes[:-1])) # ((n_groups, *card))

            for g in range(n_groups):
                for parent_value in np.ndindex(*card[:-1]):
                    counts_sum = counts[g][parent_value].sum(axis=-1)
                    alpha_sum = alpha[parent_value].sum(axis=-1,keepdims=True)

                    t1 = gammaln(alpha_sum)-gammaln(alpha_sum + counts_sum)
                    score += np.sum(t1)

                    score += np.sum(gammaln(counts[g][parent_value] + alpha[parent_value]))
                    score -= np.sum(gammaln(alpha[parent_value]))
        return score
          


def group_hdir_score(D: GroupedDataset, candidates: List[str], s = 1, s0 = 1):
    scorer = HDirScore(D, s, s0)
    score = np.zeros((len(candidates),))
    for i, u in enumerate(candidates):
        or_node = learn_cnet_stump(D, u, learn_group_chow_liu_leaf)
        children_scores = [ HDirScore(Di, s, s0).score(child.bn) 
                                 for child, Di in zip(or_node.children, D.split(u))]
        score[i] = scorer.local_score(u, []) + np.sum(children_scores)
    leaf = learn_group_chow_liu_leaf(D)
    base = scorer.score(leaf.bn)
    return score - base



### Extra ###
def learn_mdl_leaf(D: Dataset) -> LeafNode:
    df = D.as_df()
    state_names = { v: list(range(D.r[v])) for v in D.scope}
    if len(df) == 0:
        return fully_factorized_leaf(D)
    search = HillClimbSearch(df)
    score = BicScore(df, state_names=state_names)
    dag = search.estimate(scoring_method=score, max_indegree=1,show_progress=False)
    bn = BayesianNetwork(dag)
    bn.fit(df, BayesianEstimator, prior_type = "dirichlet", pseudo_counts=1, state_names=state_names)
    return LeafNode(D.scope, D.r, bn)


def learn_hdir_leaf(D: GroupedDataset, s = 1, s0 = 1):
    scorer = HDirScore(D, s, s0)
    df = D.as_df()
    tree = HillClimbSearch(df).estimate(scoring_method=scorer, max_indegree=1, show_progress=False)
    bn = BayesianNetwork()
    bn.add_nodes_from(D.scope)
    bn.add_edges_from(tree.edges())
    state_names = { v: list(range(D.r[v])) for v in D.scope}
    bn.fit(df, BayesianEstimator, prior_type = "dirichlet", pseudo_counts=s, state_names=state_names)
    return LeafNode(D.scope, D.r, bn)


def learn_bdeu_leaf(D: Dataset, s = 1):
    df = D.as_df()
    state_names = { v: list(range(D.r[v])) for v in D.scope}
    scorer = BDeuScore(df, equivalent_sample_size=s, state_names=state_names)
    tree = HillClimbSearch(df).estimate(scoring_method=scorer, max_indegree=1, show_progress=False)
    bn = BayesianNetwork()
    bn.add_nodes_from(D.scope)
    bn.add_edges_from(tree.edges())
    state_names = { v: list(range(D.r[v])) for v in D.scope}
    bn.fit(df, BayesianEstimator, prior_type = "dirichlet", pseudo_counts=s, state_names=state_names)
    return LeafNode(D.scope, D.r, bn)

def bdeu_score(D: Dataset, candidates:List[str], s = 1):
    df = D.as_df()
    leaf = learn_bdeu_leaf(D)
    state_names = { v: list(range(D.r[v])) for v in D.scope}
    scorer = BDeuScore(df, equivalent_sample_size=s, state_names=state_names)
    base = scorer.score(leaf.bn)
    score = np.zeros((len(candidates),))  - base
    for i, u in enumerate(candidates):
        or_node = learn_cnet_stump(D, u, learn_bdeu_leaf)
        children_scores = [ BDeuScore(Di.as_df(), equivalent_sample_size=s, 
                                      state_names = { v: list(range(Di.r[v])) for v in Di.scope}).score(child.bn) 
                                 for child, Di in zip(or_node.children, D.split(u))]
        score[i] += scorer.local_score(u, []) + np.sum(children_scores)
    return score

def mdl_score(D: Dataset, candidates: List[str]):
    score = np.zeros((len(candidates),)) 
    for i, u in enumerate(candidates):
        or_node = learn_cnet_stump(D, u, learn_mdl_leaf)
        param_count = np.sum([node.n_parameters for node in depth_first_order(or_node)])
        score[i] += or_node(D, log=True) - 0.5*np.log(len(D))*param_count
    leaf = learn_mdl_leaf(D)
    base = leaf(D, log=True) - 0.5*np.log(len(D))*leaf.n_parameters
    return score - base


