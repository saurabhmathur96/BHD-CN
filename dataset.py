from typing import *
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFReader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class Dataset:
    def __init__(self, df, r = None):
        self.df = df
        self.scope = df.columns.tolist()
        if r is None:
            self.r = dict(zip(self.scope, (1+df.max(axis=0)).tolist()))
        else:
            self.r = r
        
    def split(self, v:str) -> List:
        r = { key: value for key, value in self.r.items() if key != v }
        datasets = [
            Dataset(self.df[self.df[v] == value].drop([v], axis=1), r)
            for value in range(self.r[v])
        ]
        return datasets

    def subset(self, V: List[str]):
        r = { key: value for key, value in self.r.items() if key not in V }
        return Dataset(self.df[V], r)

    def counts(self, V: List[str]) -> Dict:
        return self.df[V].value_counts().to_dict()
    
    def __len__(self):
        return len(self.df)
    
    def as_df(self):
        return self.df
    

class GroupedDataset(Dataset):
    def __init__(self, df, r=None, group_col="group"):
        super().__init__(df, r)
        self.group_col = group_col
        self.scope = [each for each in self.scope if each != group_col]
    

    def split(self, v:str) -> List:
        r = { key: value for key, value in self.r.items() if key != v }
        datasets = [
            GroupedDataset(self.df[self.df[v] == value].drop([v], axis=1), r, self.group_col)
            for value in range(self.r[v])
        ]
        return datasets
    
    def subset(self, V: List[str]):
        r = { key: value for key, value in self.r.items() if key not in V }
        return GroupedDataset(self.df[V+[self.group_col]], r, self.group_col)

    def counts(self, V: List[str], by_group=False):
        if not by_group:
            return self.df[V].value_counts().to_dict()
        else:
            return [self.df.query(f"{self.group_col} == {group}")[V].value_counts().to_dict()
                    for group in range(self.r[self.group_col]) ]

    def as_df(self, with_group=False):
        if not with_group:
            return self.df.drop([self.group_col], axis=1)
        else:
            return self.df
    
    @property
    def datasets(self):
        r = { key: value for key, value in self.r.items() if key != self.group_col }
        return [
            Dataset(self.df.query(f"{self.group_col} == {group}").drop([self.group_col], axis=1), r)
            for group in range(self.r[self.group_col])
        ]



def read_dataset(name: str):
  if name in ("random", "asia", "sachs", "alarm"):
    if name == "random":
        n_nodes = 10
        bn = BayesianNetwork.get_random(n_nodes = n_nodes, edge_prob = 0.75)
        n_states = None
        variable_names = [f"V{i}" for i in range(n_nodes)]
    else:
        bn = BIFReader(f"{name}.bif").get_model()
        n_states  = max(each.variable_card for each in bn.get_cpds())
        n_nodes = len(bn.nodes)
        variable_names = list(bn.nodes)
    n_groups = 5
    samples = []
    for i in range(n_groups):
      bn.get_random_cpds(n_states = n_states, inplace = True)
      sample = bn.simulate(n_samples = 200*(i+1), show_progress = False)
      sample.columns = variable_names
      sample["group"] = i
      samples.append(sample)
    df = pd.concat(samples)
    r = { column: (max(df[column].tolist())+1 if column != "group" else n_groups) for column in df.columns }
    train, test = train_test_split(df, train_size=0.8, stratify=df["group"])
    return GroupedDataset(train, r), GroupedDataset(test, r) 
  
  elif name == "hd":
    names = ["age", "sex", "cp", "trestbps", "chol", "fbs",
          "restecg", "thalach", "exang", "oldpeak", "slope",
          "ca", "thal", "num"]
    dfs = []
    prefix = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    for i, part_name in enumerate(["cleveland", "hungarian", "va", "switzerland"]):
      df = pd.read_csv(f"{prefix}/processed.{part_name}.data",names=names, na_values="?")
      df = df[["sex", "age", "chol", "trestbps", "fbs", "num"]]
      df["group"] = i
      dfs.append(df)
    df = pd.concat(dfs).dropna()
    df["num"] = df["num"] > 0
    df["age"] = pd.cut(df["age"], [-np.inf, 40, 60, np.inf], labels = np.arange(3))
    df["chol"] = pd.cut(df["chol"], [-np.inf, 200, 240, np.inf], labels = np.arange(3))
    df["trestbps"] = pd.cut(df["trestbps"], [-np.inf, 120, 140, np.inf], labels = np.arange(3))
    # df["fbs"] = pd.cut(df["fbs"], [-np.inf, 100, 125, np.inf], labels = np.arange(3))
    df = df.astype(int)
    df.columns = ["sex", "age", "chol", "bp", "fbs", "hd", "group"]
    r = { column: df[column].max()+1 for column in df.columns }
    
    train, test = train_test_split(df, train_size=0.8, stratify=df[["group"]], random_state=0)
    return GroupedDataset(train, r), GroupedDataset(test, r) 
  
  