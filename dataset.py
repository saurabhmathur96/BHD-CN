from typing import *

class Dataset:
    def __init__(self, df, r = None):
        self.df = df
        self.scope = df.columns.tolist()
        if r is None:
            self.r = dict(zip(self.scope, (1+df.max(axis=1)).tolist()))
        else:
            self.r = r
        
    def split(self, v:str) -> List:
        r = { key: value for key, value in self.r.items() if key != v }
        datasets = [
            Dataset(self.df[self.df[v] == value], r)
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
    def __init__(self, df, group_col):
        super().__init__(df)
        self.group_col = group_col
    
    def counts(self, V: List[str], by_group=False):
        pass

    def as_df(self):
        return self.df.drop([self.group_col], axis=1)
