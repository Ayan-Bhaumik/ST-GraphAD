#!/usr/bin/env python3
"""
Inference script for trained ST-GNN models.
Use this to run anomaly detection on new network flow data.
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.models import create_model, GCNOnly, STGNN
from src.data_loader import UNSWNB15Loader


class IntrusionDetector:
    """Inference wrapper for trained models."""

    def __init__(self, model_path: str, model_type: str, data_dir: str = 'data',
                 device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load data loader to get encoders
        self.loader = UNSWNB15Loader(data_dir=data_dir, download=False)
        # We need to load the training data to fit encoders
        # In practice, you'd save the encoders with the model
        self._load_encoders()

        # Load model
        self.model_type = model_type
        self.model = self._load_model(model_path)

    def _load_encoders(self):
        """Load or re-fit encoders from training data."""
        # In production, you'd save these with the model
        # For now, we'll load a small sample to fit
        train_path = os.path.join(self.loader.data_dir, 'UNSW_NB15_training-set.csv')
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path, nrows=1000)
            _, _ = self.loader.preprocess(train_df, train_df)
        else:
            print("Warning: Training data not found for encoder fitting")

    def _load_model(self, model_path: str):
        """Load trained model."""
        # We need the input dimension - get from a sample
        # In practice, save this with the model
        in_channels = 50  # Default, should match training

        model = create_model(
            self.model_type,
            in_channels=in_channels,
            hidden_channels=128,
            gcn_layers=3,
            temporal_layers=2,
            num_heads=4,
            dropout=0.5,
            use_temporal=(self.model_type == 'stgnn')
        )

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def preprocess_flows(self, flows_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess new flow data using fitted encoders."""
        df = flows_df.copy()

        # Encode IPs
        all_ips = pd.concat([df['srcip'], df['dstip']]).unique()
        # Use existing encoder or fit new
        for ip in all_ips:
            if ip not in self.loader.ip_encoder.classes_:
                # Add unseen IPs
                self.loader.ip_encoder.classes_ = np.append(self.loader.ip_encoder.classes_, ip)

        df['srcip_enc'] = self.loader.ip_encoder.transform(df['srcip'])
        df['dstip_enc'] = self.loader.ip_encoder.transform(df['dstip'])

        # Encode ports
        all_ports = pd.concat([df['sport'], df['dsport']]).unique()
        for port in all_ports:
            if port not in self.loader.port_encoder.classes_:
                self.loader.port_encoder.classes_ = np.append(self.loader.port_encoder.classes_, port)

        df['sport_enc'] = self.loader.port_encoder.transform(df['sport'])
        df['dsport_enc'] = self.loader.port_encoder.transform(df['dsport'])

        # Encode categorical
        for cat_feat in ['proto', 'state', 'service']:
            if cat_feat in df.columns and cat_feat in self.loader.label_encoders:
                le = self.loader.label_encoders[cat_feat]
                df[f'{cat_feat}_enc'] = df[cat_feat].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        # Scale numerical
        num_cols = [c for c in self.loader.scaler.feature_names_in_ if c in df.columns]
        if len(num_cols) > 0:
            df[num_cols] = self.loader.scaler.transform(df[num_cols])

        return df

    def flows_to_graph(self, df: pd.DataFrame):
        """Convert flows to PyG graph."""
        x, y, node_ids = self.loader.create_node_features(df)
        edge_index = self.loader.create_edges(df, node_ids)

        data = torch_geometric.data.Data(x=x, edge_index=edge_index)
        data.num_nodes = x.shape[0]
        return data

    def predict(self, flows_df: pd.DataFrame, return_node_scores: bool = True) -> dict:
        """Run inference on new flows."""
        # Preprocess
        processed_df = self.preprocess_flows(flows_df)

        # Convert to graph
        graph = self.flows_to_graph(processed_df)
        graph = graph.to(self.device)

        # Inference
        with torch.no_grad():
            if self.model_type == 'stgnn':
                output = self.model([graph], return_node_scores=return_node_scores)
            else:
                output = self.model(graph, return_node_scores=return_node_scores)

        # Convert to numpy
        result = {
            'graph_anomaly_prob': output['graph_probs'][0, 1].item(),
            'graph_prediction': 'Attack' if output['graph_probs'][0, 1].item() > 0.5 else 'Normal'
        }

        if return_node_scores and 'anomaly_scores' in output:
            result['node_anomaly_scores'] = output['anomaly_scores'].cpu().numpy()
            result['node_predictions'] = output['node_probs'].argmax(dim=1).cpu().numpy()

        return result


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained ST-GNN')
    parser.add_argument('--model', type=str, choices=['gcn', 'stgnn'], required=True,
                        help='Model type')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with flows')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output predictions file')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    args = parser.parse_args()

    # Load input flows
    print(f"Loading flows from {args.input}...")
    flows = pd.read_csv(args.input)
    print(f"Loaded {len(flows)} flows")

    # Initialize detector
    print(f"Loading {args.model.upper()} model from {args.model_path}...")
    detector = IntrusionDetector(args.model_path, args.model, args.data_dir, args.device)

    # Run inference
    print("Running inference...")
    result = detector.predict(flows)

    # Print results
    print(f"\nGraph-level prediction: {result['graph_prediction']}")
    print(f"Graph-level anomaly probability: {result['graph_anomaly_prob']:.4f}")

    if 'node_anomaly_scores' in result:
        scores = result['node_anomaly_scores']
        preds = result['node_predictions']
        print(f"\nNode-level statistics:")
        print(f"  Mean anomaly score: {scores.mean():.4f}")
        print(f"  Max anomaly score: {scores.max():.4f}")
        print(f"  Nodes predicted as attack: {(preds == 1).sum()} / {len(preds)}")

        # Save node-level results
        node_results = pd.DataFrame({
            'node_id': range(len(scores)),
            'anomaly_score': scores,
            'prediction': ['Attack' if p == 1 else 'Normal' for p in preds]
        })
        node_results.to_csv(args.output.replace('.csv', '_nodes.csv'), index=False)
        print(f"\nNode predictions saved to {args.output.replace('.csv', '_nodes.csv')}")

    # Save graph-level result
    graph_result = pd.DataFrame([{
        'graph_anomaly_prob': result['graph_anomaly_prob'],
        'graph_prediction': result['graph_prediction']
    }])
    graph_result.to_csv(args.output, index=False)
    print(f"Graph prediction saved to {args.output}")


if __name__ == '__main__':
    main()