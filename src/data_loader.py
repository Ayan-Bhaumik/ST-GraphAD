"""
UNSW-NB15 Dataset Loader and Preprocessor
Handles downloading, loading, and converting network flows to graph structures.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
import networkx as nx
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# UNSW-NB15 feature names (49 features + label)
UNSW_FEATURES = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
    'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
]

CATEGORICAL_FEATURES = ['proto', 'state', 'service', 'attack_cat']
NUMERICAL_FEATURES = [f for f in UNSW_FEATURES if f not in CATEGORICAL_FEATURES and f not in ['srcip', 'dstip', 'sport', 'dsport', 'label']]

# UNSW-NB15 dataset URLs
UNSW_URLS = {
    'train': 'https://raw.githubusercontent.com/unsw-nb15/dataset/master/UNSW_NB15_training-set.csv',
    'test': 'https://raw.githubusercontent.com/unsw-nb15/dataset/master/UNSW_NB15_testing-set.csv'
}

# Alternative: Using the CSV files from the official source
UNSW_LOCAL_FILES = {
    'train': 'UNSW_NB15_training-set.csv',
    'test': 'UNSW_NB15_testing-set.csv'
}


class UNSWNB15Loader:
    """Load and preprocess UNSW-NB15 dataset."""

    def __init__(self, data_dir='data', download=True):
        self.data_dir = data_dir
        self.download = download
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.ip_encoder = LabelEncoder()
        self.port_encoder = LabelEncoder()

        os.makedirs(data_dir, exist_ok=True)

    def download_dataset(self):
        """Download UNSW-NB15 dataset if not present."""
        import urllib.request

        for split, url in UNSW_URLS.items():
            filepath = os.path.join(self.data_dir, UNSW_LOCAL_FILES[split])
            if not os.path.exists(filepath):
                print(f"Downloading {split} dataset...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    print(f"Downloaded to {filepath}")
                except Exception as e:
                    print(f"Failed to download {split}: {e}")
                    print("Please manually download the dataset from:")
                    print("https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/")

    def load_raw_data(self):
        """Load raw CSV files."""
        train_path = os.path.join(self.data_dir, UNSW_LOCAL_FILES['train'])
        test_path = os.path.join(self.data_dir, UNSW_LOCAL_FILES['test'])

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            if self.download:
                self.download_dataset()
            else:
                raise FileNotFoundError(f"Dataset files not found in {self.data_dir}")

        train_df = pd.read_csv(train_path, names=UNSW_FEATURES, header=0 if self._has_header(train_path) else None)
        test_df = pd.read_csv(test_path, names=UNSW_FEATURES, header=0 if self._has_header(test_path) else None)

        return train_df, test_df

    def _has_header(self, filepath):
        """Check if CSV has header."""
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        return not first_line.split(',')[0].replace('.', '').isdigit()

    def preprocess(self, train_df, test_df):
        """Preprocess data: encode categorical, scale numerical, encode IPs/ports."""
        # Combine for consistent encoding
        combined = pd.concat([train_df, test_df], ignore_index=True)

        # Encode IP addresses
        all_ips = pd.concat([combined['srcip'], combined['dstip']]).unique()
        self.ip_encoder.fit(all_ips)
        combined['srcip_enc'] = self.ip_encoder.transform(combined['srcip'])
        combined['dstip_enc'] = self.ip_encoder.transform(combined['dstip'])

        # Encode ports
        all_ports = pd.concat([combined['sport'], combined['dsport']]).unique()
        self.port_encoder.fit(all_ports)
        combined['sport_enc'] = self.port_encoder.transform(combined['sport'])
        combined['dsport_enc'] = self.port_encoder.transform(combined['dsport'])

        # Encode categorical features
        for cat_feat in CATEGORICAL_FEATURES:
            if cat_feat in combined.columns:
                le = LabelEncoder()
                combined[f'{cat_feat}_enc'] = le.fit_transform(combined[cat_feat].astype(str))
                self.label_encoders[cat_feat] = le

        # Scale numerical features
        num_cols = [c for c in NUMERICAL_FEATURES if c in combined.columns]
        combined[num_cols] = self.scaler.fit_transform(combined[num_cols])

        # Split back
        train_size = len(train_df)
        train_processed = combined.iloc[:train_size].copy()
        test_processed = combined.iloc[train_size:].copy()

        return train_processed, test_processed

    def create_node_features(self, df):
        """Create node feature matrix from flow data."""
        # Aggregate features per unique IP (node)
        src_features = df.groupby('srcip_enc').agg({
            'sbytes': 'mean', 'dbytes': 'mean', 'Spkts': 'mean', 'Dpkts': 'mean',
            'Sload': 'mean', 'Dload': 'mean', 'dur': 'mean', 'sttl': 'mean',
            'dttl': 'mean', 'sloss': 'mean', 'dloss': 'mean',
            'proto_enc': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'state_enc': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'service_enc': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'label': 'max'  # If any flow from this IP is attack, mark as attack
        }).reset_index()

        dst_features = df.groupby('dstip_enc').agg({
            'sbytes': 'mean', 'dbytes': 'mean', 'Spkts': 'mean', 'Dpkts': 'mean',
            'Sload': 'mean', 'Dload': 'mean', 'dur': 'mean', 'sttl': 'mean',
            'dttl': 'mean', 'sloss': 'mean', 'dloss': 'mean',
            'proto_enc': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'state_enc': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'service_enc': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'label': 'max'
        }).reset_index()

        # Merge src and dst features (nodes can be both src and dst)
        src_features.columns = ['node_id'] + [f'src_{c}' for c in src_features.columns[1:]]
        dst_features.columns = ['node_id'] + [f'dst_{c}' for c in dst_features.columns[1:]]

        node_features = pd.merge(src_features, dst_features, on='node_id', how='outer').fillna(0)

        # Create feature matrix
        feature_cols = [c for c in node_features.columns if c != 'node_id' and c != 'src_label' and c != 'dst_label']
        x = torch.tensor(node_features[feature_cols].values, dtype=torch.float)

        # Labels: 1 if attack, 0 if normal
        y = torch.tensor(np.maximum(node_features['src_label'].fillna(0), node_features['dst_label'].fillna(0)).values, dtype=torch.long)

        return x, y, node_features['node_id'].values

    def create_temporal_graphs(self, df, time_window='1H'):
        """Create a sequence of graphs over time windows."""
        df = df.copy()
        df['Stime'] = pd.to_datetime(df['Stime'], unit='s', errors='coerce')
        df = df.dropna(subset=['Stime'])
        df = df.sort_values('Stime')

        # Create time windows
        df['time_window'] = df['Stime'].dt.floor(time_window)

        graphs = []
        for window, window_df in df.groupby('time_window'):
            if len(window_df) < 10:  # Skip windows with too few flows
                continue

            x, y, node_ids = self.create_node_features(window_df)

            # Create edges from flows
            edge_index = self.create_edges(window_df, node_ids)

            if edge_index.shape[1] == 0:
                continue

            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data.time_window = window
            data.num_nodes = x.shape[0]

            graphs.append(data)

        return graphs

    def create_edges(self, df, node_ids):
        """Create edge index from flow data."""
        node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        edges = []
        for _, row in df.iterrows():
            src_idx = node_to_idx.get(row['srcip_enc'])
            dst_idx = node_to_idx.get(row['dstip_enc'])
            if src_idx is not None and dst_idx is not None:
                edges.append([src_idx, dst_idx])
                edges.append([dst_idx, src_idx])  # Undirected

        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def create_static_graph(self, df):
        """Create a single static graph from all flows."""
        x, y, node_ids = self.create_node_features(df)
        edge_index = self.create_edges(df, node_ids)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = x.shape[0]

        return data

    def get_data(self):
        """Main entry point to get processed data."""
        train_df, test_df = self.load_raw_data()
        train_processed, test_processed = self.preprocess(train_df, test_df)

        # Create static graphs for train and test
        train_graph = self.create_static_graph(train_processed)
        test_graph = self.create_static_graph(test_processed)

        # Create temporal graphs
        train_temporal = self.create_temporal_graphs(train_processed)
        test_temporal = self.create_temporal_graphs(test_processed)

        return {
            'train_static': train_graph,
            'test_static': test_graph,
            'train_temporal': train_temporal,
            'test_temporal': test_temporal,
            'train_df': train_processed,
            'test_df': test_processed,
            'num_node_features': train_graph.x.shape[1],
            'num_classes': 2,
            'ip_encoder': self.ip_encoder,
            'port_encoder': self.port_encoder,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }


def load_unsw_nb15(data_dir='data', download=True):
    """Convenience function to load UNSW-NB15 dataset."""
    loader = UNSWNB15Loader(data_dir=data_dir, download=download)
    return loader.get_data()


if __name__ == '__main__':
    # Test loading
    data = load_unsw_nb15(download=False)
    print(f"Train graph: {data['train_static']}")
    print(f"Test graph: {data['test_static']}")
    print(f"Train temporal graphs: {len(data['train_temporal'])}")
    print(f"Test temporal graphs: {len(data['test_temporal'])}")
    print(f"Node features: {data['num_node_features']}")