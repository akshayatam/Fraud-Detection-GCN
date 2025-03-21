import torch 
from torch_geometric.data import Data 
import networkx as nx 
import networkit as nk 
import numpy as np 
import pandas as pd 

class FraudDetector(): 
    def __init__(
            self, 
            df_features, 
            df_edges, 
            is_directed=False, 
            train_mask=None, 
            val_mask=None, 
            test_mask=None, 
            fraud_dict=None, 
            name=None 
    ): 
        self.name = name 
        self.df_features = df_features 
        self.df_edges = df_edges 
        self.is_directed = is_directed 

        self.nodes, self.edges, self.map_id = self._set_up_network_info() 

        self.fraud_dict = dict(
            zip(
                df_features["txId"].map(self.map_id), 
                df_features["class"] 
            )
        ) 

        self.train_mask = train_mask 
        self.val_mask = val_mask 
        self.test_mask = test_mask 

    def _set_up_network_info(self): 
        nodes = self.df_features['txId'] 
        map_id = {j:i for i,j in enumerate(nodes)} 

        if self.is_directed: 
            edges = self.df_edges[['txId1', 'txId2']] 
        else: 
            edges_direct = self.df_edges[['txId1', 'txId2']] 
            edges_rev = edges_direct[['txId2', 'txId1']] 
            edges_rev.columns = ['txId1', 'txId2'] 
            edges = pd.concat([edges_direct, edges_rev]) 

        nodes = nodes.map(map_id) 
        edges.txId1 = edges.txId1.map(map_id)
        edges.txId2 = edges.txId2.map(map_id) 

        edges = edges.astype(int) 

        return (nodes, edges, map_id) 
    
    def get_network_nx(self): 
        edges_zipped = zip(self.edges['txId1'], self.edges['txId2']) 

        if self.is_directed: 
            G_nx = nx.DiGraph() 
        else: 
            G_nx = nx.Graph() 

        G_nx.add_nodes_from(self.nodes) 
        G_nx.add_edges_from(edges_zipped) 

        return (G_nx) 
    
    def get_network_nk(self): 
        edges_zipped = zip(self.edges['txId1'], self.edges['txId2']) 

        G_nk = nk.Graph(len(self.nodes), directed=self.is_directed) 

        for u,v in edges_zipped: 
            G_nk.addEdge(u,v) 

        return (G_nk) 
    
    def get_network_torch(self): 
        labels = self.df_features['class'] 
        features = self.df_features[self.df_features.columns.drop(['txId', 'class'])] 

        x = torch.tensor(np.array(features.values, dtype=float), dtype=torch.float) 

        if x.size()[1] == 0: 
            x = torch.ones(x.size()[0], 1) 
        
        y = torch.tensor(np.array(labels.values, dtype=np.int64), dtype=torch.int64) 

        # Reformat and convert to tensor 
        edge_index = np.array(self.edges.values).T 
        edge_index = torch.tensor(edge_index, dtype=torch.long) 

        #create weights tensor with same shape of edge_index
        weights = torch.tensor([1]* edge_index.shape[1] , dtype=torch.float) 
        
        # Create pyG dataset
        data = Data(x=x, y=y, edge_index=edge_index)

        if self.train_mask is not None:
            data.train_mask = torch.tensor(self.train_mask, dtype=torch.bool)
        if self.val_mask is not None:
            data.val_mask = torch.tensor(self.val_mask, dtype=torch.bool)
        if self.test_mask is not None:
            data.test_mask = torch.tensor(self.test_mask, dtype=torch.bool)
        
        return (data)
    
    def get_features(self, is_full=False):
        if self.name == 'elliptic':    
            if is_full:
                columns = [i for i in range(2, 184)]
            else:
                columns = [i for i in range(2, 95)]
        else:
            columns = self.df_features.columns.drop(['txId', 'class']) 
        X = self.df_features[columns]
        return (X)
    
    def get_features_torch(self, is_full=False):
        X = self.get_features(is_full)
        X = torch.tensor(X.values, dtype=torch.float32)
        return (X)

    def get_train_test_split_intrinsic(self, train_mask, test_mask, device = 'cpu'): 
        X = self.get_features()
        y = self.df_features['class']

        X_train = X[train_mask.numpy()]
        y_train = y[train_mask.numpy()]

        X_test = X[test_mask.numpy()]
        y_test = y[test_mask.numpy()]

        X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)

        X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test.values, dtype=torch.long).to(device)

        
        return (X_train, y_train, X_test, y_test)

    def get_fraud_dict(self):
        return (self.fraud_dict)
    
    def get_masks(self):
        return (self.train_mask, self.val_mask, self.test_mask)
