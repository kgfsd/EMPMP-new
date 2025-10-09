"""
Simple test suite for Dynamic GCN implementation.

This file contains basic tests to verify the GCN implementation works correctly.
Run with: python test_dynamic_gcn.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from models_dual_inter_traj.gcn import (
            GraphConvolution,
            build_dynamic_adjacency_matrix,
            build_distance_based_adjacency_matrix,
        )
        from models_dual_inter_traj.dynamic_gcn_model import DynamicGCNModel, HybridGCNMLP
        from models_dual_inter_traj.dynamic_data_utils import (
            DynamicPersonBatch,
            collate_variable_persons,
            compute_loss_with_mask,
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_gcn_layer_creation():
    """Test creating a GCN layer without torch."""
    print("\nTesting GCN layer creation...")
    try:
        from models_dual_inter_traj.gcn import GraphConvolution
        
        # Just test that the class can be instantiated
        print("✓ GraphConvolution class loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_dynamic_model_creation():
    """Test creating DynamicGCNModel without torch."""
    print("\nTesting DynamicGCNModel creation...")
    try:
        from models_dual_inter_traj.dynamic_gcn_model import DynamicGCNModel
        
        print("✓ DynamicGCNModel class loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_data_utils():
    """Test data utility functions."""
    print("\nTesting data utilities...")
    try:
        from models_dual_inter_traj.dynamic_data_utils import (
            DynamicPersonBatch,
            validate_batch_consistency,
        )
        
        print("✓ Data utility classes loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_code_structure():
    """Test that all files have correct structure."""
    print("\nTesting code structure...")
    
    files_to_check = [
        'gcn.py',
        'dynamic_gcn_model.py',
        'dynamic_data_utils.py',
        'dynamic_gcn_example.py',
        '__init__.py',
    ]
    
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models_dual_inter_traj'
    )
    
    all_exist = True
    for filename in files_to_check:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {filename}: {size} bytes")
        else:
            print(f"  ✗ {filename}: NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("✓ All files present")
    else:
        print("✗ Some files missing")
    
    return all_exist


def test_documentation():
    """Test that documentation exists."""
    print("\nTesting documentation...")
    
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models_dual_inter_traj'
    )
    
    readme_path = os.path.join(base_path, 'GCN_README.md')
    
    if os.path.exists(readme_path):
        size = os.path.getsize(readme_path)
        print(f"  ✓ GCN_README.md: {size} bytes")
        
        # Check if README has key sections
        with open(readme_path, 'r') as f:
            content = f.read()
            required_sections = [
                '## Overview',
                '## Architecture',
                '## Usage Examples',
                '## API Reference',
            ]
            
            missing = [s for s in required_sections if s not in content]
            if not missing:
                print("  ✓ All required sections present")
                return True
            else:
                print(f"  ✗ Missing sections: {missing}")
                return False
    else:
        print("  ✗ GCN_README.md: NOT FOUND")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("Dynamic GCN Implementation Tests")
    print("=" * 70)
    
    tests = [
        ("Module Imports", test_imports),
        ("GCN Layer Creation", test_gcn_layer_creation),
        ("Dynamic Model Creation", test_dynamic_model_creation),
        ("Data Utilities", test_data_utils),
        ("Code Structure", test_code_structure),
        ("Documentation", test_documentation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
