"""
Syntax validation test for Dynamic GCN implementation.

This tests that all Python files have valid syntax.
"""

import py_compile
import os
import sys


def test_file_syntax(filepath):
    """Test if a Python file has valid syntax."""
    try:
        py_compile.compile(filepath, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)


def main():
    """Test syntax of all new files."""
    print("=" * 70)
    print("Syntax Validation Tests")
    print("=" * 70)
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    files_to_test = [
        'gcn.py',
        'dynamic_gcn_model.py',
        'dynamic_data_utils.py',
        'dynamic_gcn_example.py',
        'test_dynamic_gcn.py',
        '__init__.py',
    ]
    
    all_passed = True
    
    for filename in files_to_test:
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ {filename}: FILE NOT FOUND")
            all_passed = False
            continue
        
        success, error = test_file_syntax(filepath)
        
        if success:
            size = os.path.getsize(filepath)
            lines = sum(1 for _ in open(filepath))
            print(f"✓ {filename}: Valid syntax ({lines} lines, {size} bytes)")
        else:
            print(f"✗ {filename}: SYNTAX ERROR")
            print(f"  Error: {error}")
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("✓ All files have valid Python syntax!")
        return 0
    else:
        print("✗ Some files have syntax errors")
        return 1


if __name__ == '__main__':
    sys.exit(main())
