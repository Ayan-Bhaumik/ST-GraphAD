#!/usr/bin/env python3
"""
Test script to verify model forward pass works correctly.
"""

import sys
import torch
from torch_geometric.data import Data

sys.path.insert(0, 'src')

from src.models import GCNOnly, STGNN, create_model


def create_dummy_graph(num_nodes=50, num_features=30, num_edges=200):
    """Create a dummy graph for testing."""
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    y = torch.randint(0, 2, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


def test_gcn_only():
    """Test GCN-only model."""
    print("Testing GCN-only model...")

    graph = create_dummy_graph()
    in_channels = graph.x.shape[1]

    model = create_model('gcn', in_channels, hidden_channels=64, gcn_layers=2, dropout=0.1)

    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(graph, return_node_scores=True)

    print(f"  Graph logits shape: {output['graph_logits'].shape}")
    print(f"  Graph probs shape: {output['graph_probs'].shape}")
    print(f"  Node embeddings shape: {output['node_embeddings'].shape}")
    print(f"  Node logits shape: {output['node_logits'].shape}")
    print(f"  Node probs shape: {output['node_probs'].shape}")
    print(f"  Anomaly scores shape: {output['anomaly_scores'].shape}")

    assert output['graph_logits'].shape == (1, 2)
    assert output['graph_probs'].shape == (1, 2)
    assert output['node_embeddings'].shape == (graph.num_nodes, 64)
    assert output['node_logits'].shape == (graph.num_nodes, 2)
    assert output['anomaly_scores'].shape == (graph.num_nodes,)

    print("  ✓ GCN-only test passed")
    return True


def test_stgnn():
    """Test ST-GNN model."""
    print("\nTesting ST-GNN model...")

    # Create temporal sequence of graphs
    num_windows = 5
    graphs = [create_dummy_graph(num_nodes=30 + i*5, num_features=30, num_edges=100 + i*20)
              for i in range(num_windows)]

    in_channels = graphs[0].x.shape[1]

    model = create_model('stgnn', in_channels, hidden_channels=64, gcn_layers=2,
                        temporal_layers=1, num_heads=2, dropout=0.1)

    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(graphs, return_node_scores=True)

    print(f"  Graph logits shape: {output['graph_logits'].shape}")
    print(f"  Graph probs shape: {output['graph_probs'].shape}")
    print(f"  Node embeddings shape: {output['node_embeddings'].shape}")
    print(f"  Node logits shape: {output['node_logits'].shape}")
    print(f"  Anomaly scores shape: {output['anomaly_scores'].shape}")

    assert output['graph_logits'].shape == (1, 2)
    assert output['graph_probs'].shape == (1, 2)
    assert output['node_embeddings'].shape[0] == graphs[-1].num_nodes
    assert output['node_logits'].shape[0] == graphs[-1].num_nodes
    assert output['anomaly_scores'].shape[0] == graphs[-1].num_nodes

    print("  ✓ ST-GNN test passed")
    return True


def test_temporal_attention():
    """Test temporal attention module directly."""
    print("\nTesting Temporal Attention module...")

    from src.models import TemporalAttention

    batch_size = 2
    seq_len = 5
    embed_dim = 64
    num_heads = 4

    attn = TemporalAttention(embed_dim, num_heads, dropout=0.1)

    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    output, weights = attn(x, mask)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {weights.shape}")

    assert output.shape == (batch_size, seq_len, embed_dim)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    print("  ✓ Temporal Attention test passed")
    return True


def test_positional_encoding():
    """Test positional encoding."""
    print("\nTesting Positional Encoding...")

    from src.models import PositionalEncoding

    d_model = 64
    seq_len = 10
    batch_size = 3

    pe = PositionalEncoding(d_model, dropout=0.0)

    x = torch.zeros(batch_size, seq_len, d_model)
    output = pe(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == (batch_size, seq_len, d_model)
    # Check that positional encoding was added (output != input)
    assert not torch.allclose(output, x)

    print("  ✓ Positional Encoding test passed")
    return True


def main():
    print("="*60)
    print("Model Forward Pass Tests")
    print("="*60)

    tests = [
        test_gcn_only,
        test_stgnn,
        test_temporal_attention,
        test_positional_encoding,
    ]

    all_passed = True
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1


if __name__ == '__main__':
    sys.exit(main())