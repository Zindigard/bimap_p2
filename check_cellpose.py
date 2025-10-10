"""
Cellpose Installation Verification Script
Run: python check_cellpose.py
"""

import sys
import numpy as np

print("Cellpose Installation Check")
print("=" * 50)

try:
    # Test basic import
    from cellpose import models, io
    print("PASS: Cellpose imported successfully")
    
    # Check GPU availability
    use_gpu = models.use_gpu()
    print(f"PASS: GPU available: {use_gpu}")
    
    # Test model loading
    model = models.CellposeModel(gpu=use_gpu)
    print("PASS: Cellpose model loaded successfully")
    
    # Test with a small sample image
    print("Testing with sample image...")
    sample_image = np.random.rand(64, 64, 3).astype(np.float32)
    
    masks, flows, styles = model.eval(
        sample_image, 
        diameter=30, 
        channels=[0,0],
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )
    
    print(f"PASS: Segmentation test successful - Found {masks.max()} cells")
    print("SUCCESS: Cellpose installation is working correctly!")
    
except ImportError as e:
    print(f"FAIL: Cellpose import failed: {e}")
    print("Solution: Try: pip install cellpose")
    sys.exit(1)
    
except Exception as e:
    print(f"FAIL: Cellpose test failed: {e}")
    print("Solution: Check your installation and GPU drivers")
    sys.exit(1)

print("\nCellpose Installation Summary:")
print("   - Basic imports: PASS")
print("   - GPU detection: PASS")  
print("   - Model loading: PASS")
print("   - Segmentation:  PASS")
print("\nReady to use Cellpose!")