#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
Run this after installing dependencies.
"""

import sys
import traceback

def test_import(module_name, import_statement):
    """Test importing a module."""
    try:
        exec(import_statement)
        print(f"✓ {module_name}")
        return True
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        return False


def main():
    print("Testing imports...")
    print("-" * 40)

    tests = [
        ("PyTorch", "import torch; print(f'  Version: {torch.__version__}')"),
        ("PyTorch Geometric", "import torch_geometric; print(f'  Version: {torch_geometric.__version__}')"),
        ("NumPy", "import numpy; print(f'  Version: {numpy.__version__}')"),
        ("Pandas", "import pandas; print(f'  Version: {pandas.__version__}')"),
        ("Scikit-learn", "import sklearn; print(f'  Version: {sklearn.__version__}')"),
        ("Matplotlib", "import matplotlib; print(f'  Version: {matplotlib.__version__}')"),
        ("NetworkX", "import networkx; print(f'  Version: {networkx.__version__}')"),
        ("SciPy", "import scipy; print(f'  Version: {scipy.__version__}')"),
        ("TQDM", "import tqdm; print(f'  Version: {tqdm.__version__}')"),
    ]

    all_passed = True
    for name, stmt in tests:
        if not test_import(name, stmt):
            all_passed = False

    print("-" * 40)

    # Test local modules
    sys.path.insert(0, 'src')
    local_tests = [
        ("data_loader", "from src.data_loader import load_unsw_nb15, UNSWNB15Loader"),
        ("models", "from src.models import GCNOnly, STGNN, create_model, TemporalAttention"),
        ("train", "from src.train import Trainer, train_model, compare_models"),
        ("visualize", "from src.visualize import plot_graph_structure, plot_embeddings_tsne"),
    ]

    for name, stmt in local_tests:
        if not test_import(name, stmt):
            all_passed = False

    print("-" * 40)
    if all_passed:
        print("All imports successful! ✓")
        return 0
    else:
        print("Some imports failed! ✗")
        return 1


if __name__ == '__main__':
    sys.exit(main())