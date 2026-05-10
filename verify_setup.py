"""
Verification script to ensure all components are properly set up.
Run this before starting training to catch any issues early.
"""

import sys
import os
import importlib.util


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80)
    
    dependencies = {
        'torch': 'PyTorch',
        'tensorboard': 'TensorBoard',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'monai': 'MONAI',
        'cv2': 'OpenCV',
        'tifffile': 'tifffile',
        'scipy': 'SciPy',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    missing = []
    
    for module_name, package_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name:20} - OK")
        except ImportError:
            print(f"✗ {package_name:20} - MISSING")
            missing.append(package_name)
    
    return len(missing) == 0, missing


def check_structure():
    """Check if project structure is correct."""
    print("\n" + "=" * 80)
    print("CHECKING PROJECT STRUCTURE")
    print("=" * 80)
    
    required_dirs = [
        'data',
        'losses/segmentation_losses',
        'losses/topological_losses',
        'metrics',
        'models',
        'configs',
        'Train',
        'VessMAP',
        'checkpoints'
    ]
    
    required_files = [
        'data/dataset.py',
        'data/transforms.py',
        'losses/segmentation_losses/losses.py',
        'losses/topological_losses/losses.py',
        'metrics/segmentation_metrics.py',
        'metrics/topological_metrcis.py',
        'models/__init__.py',
        'models/TransUnet.py',
        'Train/pipeline.py',
        'configs/default_config.yaml',
        'train.py',
        'evaluate.py'
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path:40} - OK")
        else:
            print(f"✗ {dir_path:40} - MISSING")
            all_ok = False
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ {file_path:40} - OK")
        else:
            print(f"✗ {file_path:40} - MISSING")
            all_ok = False
    
    return all_ok


def check_config():
    """Check if configuration file is valid."""
    print("\n" + "=" * 80)
    print("CHECKING CONFIGURATION")
    print("=" * 80)
    
    try:
        import yaml
        
        with open('configs/default_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['data', 'model', 'training', 'loss']
        
        all_ok = True
        for key in required_keys:
            if key in config:
                print(f"✓ Config section '{key}' - OK")
            else:
                print(f"✗ Config section '{key}' - MISSING")
                all_ok = False
        
        # Check dataset path exists
        data_path = config.get('data', {}).get('dataset_path')
        if data_path and os.path.isdir(data_path):
            print(f"✓ Dataset path exists - OK")
        else:
            print(f"✗ Dataset path not found: {data_path}")
            all_ok = False
        
        return all_ok
    
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def check_imports():
    """Check if all modules can be imported."""
    print("\n" + "=" * 80)
    print("CHECKING MODULE IMPORTS")
    print("=" * 80)
    
    modules = [
        ('Train.pipeline', 'Pipeline'),
        ('data.dataset', 'VessMapDataset'),
        ('models', 'TransUNet'),
        ('losses.segmentation_losses', 'get_segmentation_loss'),
        ('losses.topological_losses', 'get_topological_loss'),
        ('metrics.segmentation_metrics', 'SegmentationMetrics'),
    ]
    
    all_ok = True
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {module_name:50} - OK")
        except Exception as e:
            print(f"✗ {module_name:50} - ERROR: {e}")
            all_ok = False
    
    return all_ok


def check_gpu():
    """Check GPU availability."""
    print("\n" + "=" * 80)
    print("CHECKING GPU")
    print("=" * 80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available - OK")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Count: {torch.cuda.device_count()}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠ CUDA not available - Will use CPU (slower)")
            return True
    
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def main():
    """Run all checks."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "VESSMAP PIPELINE VERIFICATION" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    
    results = {}
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    results['Dependencies'] = deps_ok
    
    if not deps_ok:
        print(f"\n⚠  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
    
    # Check structure
    results['Structure'] = check_structure()
    
    # Check config
    results['Configuration'] = check_config()
    
    # Check imports
    results['Imports'] = check_imports()
    
    # Check GPU
    results['GPU'] = check_gpu()
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_ok = True
    for check_name, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{check_name:30} {status_str}")
        all_ok = all_ok and status
    
    print("\n" + "=" * 80)
    
    if all_ok:
        print("✓ All checks passed! Ready to start training.")
        print("\nQuick start:")
        print("  python train.py --config configs/default_config.yaml")
        print("\nFor more options:")
        print("  python train.py --help")
        print("  python evaluate.py --help")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
