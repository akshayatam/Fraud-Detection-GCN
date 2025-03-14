import pandas as pd 
import torch 
from utils.network import FraudDetector 

# Load Elliptic++ Dataset 
def load_elliptic_pp(): 
    data_paths = [
        'data/txs_features.csv', 
        'data/txs_edgelist.csv', 
        'data/txs_classes.csv', 
    ]

    # Create DataFrames 
    df_features = pd.read_csv(data_paths[0]) 
    df_edges = pd.read_csv(data_paths[1]) 
    df_classes = pd.read_csv(data_paths[2]) 

    # Rename 'df_features' columns to be in line with the one used by the authors 
    columns = {
        'txId': 'txId', 
        'Time step': 'time_step' 
    }

    columns.update({f'Local_feature_{i}': i + 1 for i in range(1, 94)}) 
    columns.update({f'Aggregate_feature_{i}': i + 94 for i in range(1, 73)})

    other_features = [
        "in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
        "num_input_addresses", "num_output_addresses",
        "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
        "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total"
    ] 

    columns.update({feature: idx for idx, feature in enumerate(other_features, start=167)})
    
    # Apply the rename
    df_features = df_features.rename(columns=columns) 

    x = torch.from_numpy(df_features.loc[:, 'time_step':].values).to(torch.float) 

    # Rename 'df_classes' to be in line 
    # Elliptic++ has 3 classes 
    # 1: Illicit    2: Licit    3: UNK 
    mapping = {
        3: 2, 
        1: 1, 
        2: 0
    }   # Using int as key because dataset has int type 

    df_classes['class'] = df_classes['class'].map(mapping) 
    y = torch.from_numpy(df_classes['class'].values) 

    # Add the mappings to df_features 
    df_features['class'] = y 

    # Timestep based split 
    time_step = torch.from_numpy(df_features['time_step'].values) 

    train_mask = (time_step < 30) & (y != 2) 
    val_mask = (time_step >= 30) & (time_step < 40) & (y != 2) 
    test_mask = (time_step >= 40) * (y != 2) 

    fraud_detector_network = FraudDetector(df_features, df_edges, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, name='elliptic_pp') 

    return (fraud_detector_network) 