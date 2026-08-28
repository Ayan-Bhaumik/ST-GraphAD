"""
UNSW-NB15 Dataset Loader and Preprocessor
Handles loading, and converting network flows to graph structures with proper train/val/test splits.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import networkx as nx
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# UNSW-NB15 feature names (matching actual CSV columns)
UNSW_FEATURES = [
    'id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
    'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
    'is_sm_ips_ports', 'attack_cat', 'label'
]

CATEGORICAL_FEATURES = ['proto', 'state', 'service', 'attack_cat']
NUMERICAL_FEATURES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]

UNSW_LOCAL_FILES = {
    'train': 'UNSW_NB15_training-set.csv',
    'test': 'UNSW_NB15_testing-set.csv'
}


class UNSWNB15Loader:
    """Load and preprocess UNSW-NB15 dataset with proper train/val/test splits."""

    def __init__(self, data_dir='data', download=True, val_split=0.15, seed=42):
        self.data_dir = data_dir
        self.download = download
        self.val_split = val_split
        self.seed = seed

        # Encoders/scalers (fitted on TRAINING DATA ONLY)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.ip_encoder = LabelEncoder()
        self.port_encoder = LabelEncoder()

        os.makedirs(data_dir, exist_ok=True)

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
        """
        Preprocess data: encode categorical, scale numerical.
        IMPORTANT: Fit encoders/scalers on TRAINING data only, then apply to test.
        """
        # Encode categorical features on TRAINING data
        for cat_feat in CATEGORICAL_FEATURES:
            if cat_feat in train_df.columns:
                le = LabelEncoder()
                train_df[f'{cat_feat}_enc'] = le.fit_transform(train_df[cat_feat].astype(str))
                self.label_encoders[cat_feat] = le

        # Apply to test data
        for cat_feat in CATEGORICAL_FEATURES:
            if cat_feat in test_df.columns and cat_feat in self.label_encoders:
                le = self.label_encoders[cat_feat]
                # Handle unseen categories in test
                test_df[f'{cat_feat}_enc'] = test_df[cat_feat].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Create pseudo source/dest IPs from categorical combinations (on BOTH)
        for df in [train_df, test_df]:
            df['srcip'] = df['proto'].astype(str) + '_' + df['service'].astype(str) + '_src'
            df['dstip'] = df['state'].astype(str) + '_' + df['service'].astype(str) + '_dst'
            df['sport'] = df['proto'].astype(str) + '_' + df['spkts'].astype(str)
            df['dsport'] = df['state'].astype(str) + '_' + df['dpkts'].astype(str)

        # Encode IP addresses on TRAINING data
        all_ips = pd.concat([train_df['srcip'], train_df['dstip']]).unique()
        self.ip_encoder.fit(all_ips)
        train_df['srcip_enc'] = self.ip_encoder.transform(train_df['srcip'])
        train_df['dstip_enc'] = self.ip_encoder.transform(train_df['dstip'])

        # Apply to test data (handle unseen IPs)
        test_df['srcip_enc'] = test_df['srcip'].apply(
            lambda x: self.ip_encoder.transform([x])[0] if x in self.ip_encoder.classes_ else -1
        )
        test_df['dstip_enc'] = test_df['dstip'].apply(
            lambda x: self.ip_encoder.transform([x])[0] if x in self.ip_encoder.classes_ else -1
        )

        # Encode ports on TRAINING data
        all_ports = pd.concat([train_df['sport'], train_df['dsport']]).unique()
        self.port_encoder.fit(all_ports)
        train_df['sport_enc'] = self.port_encoder.transform(train_df['sport'])
        train_df['dsport_enc'] = self.port_encoder.transform(train_df['dsport'])

        # Apply to test data
        test_df['sport_enc'] = test_df['sport'].apply(
            lambda x: self.port_encoder.transform([x])[0] if x in self.port_encoder.classes_ else -1
        )
        test_df['dsport_enc'] = test_df['dsport'].apply(
            lambda x: self.port_encoder.transform([x])[0] if x in self.port_encoder.classes_ else -1
        )

        # Scale numerical features on TRAINING data
        num_cols = [c for c in NUMERICAL_FEATURES if c in train_df.columns]
        train_df[num_cols] = self.scaler.fit_transform(train_df[num_cols])

        # Apply to test data
        test_num_cols = [c for c in num_cols if c in test_df.columns]
        test_df[test_num_cols] = self.scaler.transform(test_df[test_num_cols])

        return train_df, test_df

    def split_train_val(self, train_df, val_split=None, seed=None):
        """Split training data into train/validation sets (stratified by label)."""
        if val_split is None:
            val_split = self.val_split
        if seed is None:
            seed = self.seed

        # Split by flows (not nodes) to maintain temporal structure
        train_flows, val_flows = train_test_split(
            train_df, test_size=val_split, random_state=seed, stratify=train_df['label']
        )

        train_flows = train_flows.reset_index(drop=True)
        val_flows = val_flows.reset_index(drop=True)

        print(f"Train/Val split: {len(train_flows)} train flows, {len(val_flows)} val flows")
        print(f"  Train label dist: {train_flows['label'].value_counts().to_dict()}")
        print(f"  Val label dist: {val_flows['label'].value_counts().to_dict()}")

        return train_flows, val_flows

    def create_node_features(self, df):
        """Create node feature matrix from flow data."""
        # Aggregate features per unique IP (node) - use actual CSV column names
        src_features = df.groupby('srcip_enc').agg({
            'sbytes': 'mean', 'dbytes': 'mean', 'spkts': 'mean', 'dpkts': 'mean',
            'sload': 'mean', 'dload': 'mean', 'dur': 'mean', 'sttl': 'mean',
            'dttl': 'mean', 'sloss': 'mean', 'dloss': 'mean',
            'proto_enc': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'state_enc': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'service_enc': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'label': 'max'
        }).reset_index()

        dst_features = df.groupby('dstip_enc').agg({
            'sbytes': 'mean', 'dbytes': 'mean', 'spkts': 'mean', 'dpkts': 'mean',
            'sload': 'mean', 'dload': 'mean', 'dur': 'mean', 'sttl': 'mean',
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

    def create_edges(self, df, node_ids):
        """Create edge index from flow data."""
        node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        edges = []
        for _, row in df.iterrows():
            src_idx = node_to_idx.get(row['srcip_enc'])
            dst_idx = node_to_idx.get(row['dstip_enc'])
            if src_idx is not None and dst_idx is not None and src_idx != -1 and dst_idx != -1:
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

    def create_temporal_graphs(self, df, window_size=1000):
        """Create a sequence of graphs over pseudo-temporal windows based on row order."""
        df = df.copy()
        # Create pseudo-temporal windows using row index
        df['pseudo_time'] = range(len(df))
        df['time_window'] = (df['pseudo_time'] // window_size).astype(str) + '_window'

        graphs = []
        for window, window_df in df.groupby('time_window'):
            if len(window_df) < 10:  # Skip windows with too few flows
                continue

            x, y, node_ids = self.create_node_features(window_df)
            edge_index = self.create_edges(window_df, node_ids)

            if edge_index.shape[1] == 0:
                continue

            # Also store attack_cat for per-category evaluation
            attack_cats = window_df['attack_cat'].values

            data = Data(x=x, edge_index=edge_index, y=y)
            data.time_window = window
            data.num_nodes = x.shape[0]
            data.attack_cats = attack_cats

            graphs.append(data)

        return graphs

    def get_data(self):
        """Main entry point to get processed data with train/val/test split."""
        train_df, test_df = self.load_raw_data()

        # Preprocess: fit on train, transform test
        train_processed, test_processed = self.preprocess(train_df, test_df)

        # Split train into train/val
        train_flows, val_flows = self.split_train_val(train_processed)

        # Create temporal graphs for each split
        window_size = 1000
        train_temporal = self.create_temporal_graphs(train_flows, window_size)
        val_temporal = self.create_temporal_graphs(val_flows, window_size)
        test_temporal = self.create_temporal_graphs(test_processed, window_size)

        # Also create static graphs (using all flows in each split)
        train_static = self.create_static_graph(train_flows)
        val_static = self.create_static_graph(val_flows)
        test_static = self.create_static_graph(test_processed)

        print(f"\n=== Graph Statistics ===")
        print(f"Train temporal graphs: {len(train_temporal)}")
        print(f"Val temporal graphs: {len(val_temporal)}")
        print(f"Test temporal graphs: {len(test_temporal)}")
        print(f"Train static nodes: {train_static.num_nodes}, edges: {train_static.num_edges}")
        print(f"Val static nodes: {val_static.num_nodes}, edges: {val_static.num_edges}")
        print(f"Test static nodes: {test_static.num_nodes}, edges: {test_static.num_edges}")
        print(f"Node features: {train_static.x.shape[1]}")

        return {
            'train_static': train_static,
            'val_static': val_static,
            'test_static': test_static,
            'train_temporal': train_temporal,
            'val_temporal': val_temporal,
            'test_temporal': test_temporal,
            'train_df': train_flows,
            'val_df': val_flows,
            'test_df': test_processed,
            'num_node_features': train_static.x.shape[1],
            'num_classes': 2,
            'ip_encoder': self.ip_encoder,
            'port_encoder': self.port_encoder,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }


def load_unsw_nb15(data_dir='data', download=True, val_split=0.15, seed=42):
    """Convenience function to load UNSW-NB15 dataset."""
    loader = UNSWNB15Loader(data_dir=data_dir, download=download, val_split=val_split, seed=seed)
    return loader.get_data()


if __name__ == '__main__':
    # Test loading
    data = load_unsw_nb15(download=False)
    print(f"Train temporal graphs: {len(data['train_temporal'])}")
    print(f"Val temporal graphs: {len(data['val_temporal'])}")
    print(f"Test temporal graphs: {len(data['test_temporal'])}")
    print(f"Node features: {data['num_node_features']}")