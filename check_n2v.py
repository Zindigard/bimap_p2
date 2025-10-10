"""
Noise2Void Installation Verification Script
Run: python check_n2v.py
"""

import sys
import numpy as np

print("Noise2Void Installation Check")
print("=" * 50)

try:
    # Test basic imports
    from n2v.models import N2VConfig, N2V
    print("PASS: N2V imported successfully")
    
    # Test TensorFlow availability
    import tensorflow as tf
    print(f"PASS: TensorFlow version: {tf.__version__}")
    
    # Check GPU availability
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"PASS: TensorFlow GPU: Available ({len(gpu_devices)} device(s))")
        for device in gpu_devices:
            print(f"   - {device}")
    else:
        print("INFO: TensorFlow GPU: Not available (will use CPU)")
    
    # Test N2V configuration with minimal setup
    print("Testing N2V configuration...")
    
    # Create dummy training data
    X_train = np.random.random((8, 32, 32, 1)).astype(np.float32)
    
    # Create a simple configuration
    config = N2VConfig(
        X_train, 
        'YXC',
        train_steps_per_epoch=2,
        train_epochs=1,
        train_loss='mse',
        batch_norm=True,
        train_batch_size=2,
        n2v_perc_pix=0.198,
        n2v_patch_shape=(16, 16),
        blurpool=True,
        unet_kern_size=3,
        unet_n_first=8,
        unet_n_depth=1,
        train_learning_rate=1e-3
    )
    
    print("PASS: N2V configuration created successfully")
    
    # Test model creation
    model_name = 'test_n2v_model'
    model = N2V(config=config, name=model_name, basedir='./models_test')
    
    print("PASS: N2V model instantiated successfully")
    print("SUCCESS: N2V installation is working correctly!")
    
    # Clean up test directory
    import shutil
    shutil.rmtree('./models_test', ignore_errors=True)
    
except ImportError as e:
    print(f"FAIL: N2V import failed: {e}")
    print("Solution: Try: pip install n2v")
    sys.exit(1)
    
except Exception as e:
    print(f"FAIL: N2V test failed: {e}")
    print("Solution: Check TensorFlow and N2V installation")
    sys.exit(1)

print("\nN2V Installation Summary:")
print("   - Basic imports: PASS")
print("   - TensorFlow:    PASS")
print("   - GPU detection: PASS")
print("   - Model config:  PASS")
print("\nReady to use Noise2Void!")