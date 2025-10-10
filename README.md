# Microscopy Image Analysis Pipeline

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Scripts Overview](#script-descriptions)
- [Usage](#usage)
- [Citation](#related-citations)

  ## Overview

A comprehensive automated pipeline for processing, analyzing, and quantifying biological microscopy images with AI-powered cell segmentation, growth rate measurement, and multi-channel intensity analysis.
**Key Features:**
- **Automated Workflow**: From raw CZI files to quantitative insights
- **Deep Learning Segmentation**: Cellpose-based cell detection
- **Performance Metrics**: IoU, precision, and accuracy evaluation
- **Growth Analysis**: Cell dimensions and growth rate calculations
- **Multi-channel Analysis**: Intensity profiling across fluorescence channels

  ## Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU (optional, for faster processing)

### Setup
```bash
# 1. Clone repository
git clone https://github.com/Zindigard/bimap_p2.git
cd bimap_p2   # or the correct folder name after cloning

# 2. Create and activate conda environment
conda create -n Project python=3.9
conda activate Project

# 3. Install all dependencies
pip install numpy scipy matplotlib pandas scikit-image scikit-learn
pip install tifffile czifile tqdm pathlib Pillow
pip install cellpose
pip install n2v

# 4. (optional) Verify installation
python -c "import cellpose; print('Cellpose installed successfully')"
python -c "import n2v; print('N2V installed successfully')"

# 5. (Recommended) Run comprehensive verification scripts
python check_cellpose.py
python check_n2v.py
```
## Script Descriptions

### 1. `extractor.py`
**Purpose:**  
Processes raw CZI microscope files and converts them to standardized TIFF format.  

**Key Functions:**  
- Extracts comprehensive metadata (microscope specs, laser settings, pixel dimensions).  
- Converts CZI to TIFF while preserving image data.  

---

### 2. `truth_extractor.py`
**Purpose:**  
Generates ground truth masks from ImageJ ROI annotations.  

**Key Functions:**  
- Reads ImageJ ROI zip files (`*_ROISET.zip`).  
- Converts manual annotations to binary masks.  
- *(Only needed if you plan to fine-tune models; otherwise can be skipped.)*  

---

### 3. `denoising.py`
**Purpose:**  
Enhances image quality using the BM3D denoising algorithm.  

**Key Functions:**  
- Applies Block-Matching 3D denoising.  
- Calculates quality metrics (SSIM, PSNR, MSE).  
- Generates performance reports and comparisons.  

**Adjustable Parameters:**  
- `sigma`: Noise level (0.1–1.0).  

---

### 4. `segmentation.py`
**Purpose:**  
Performs cell segmentation using Cellpose.  

**Key Functions:**  
- Uses pre-trained Cellpose models for cell detection.  
- Generates binary masks and outline visualizations.  
- Includes post-processing to clean segmentation results.  

**Adjustable Parameters:**  
- `flow_threshold`: Detection sensitivity (0.4–1.0).  
- `cellprob_threshold`: Cell probability (-6 to 6).  
- `min_size`: Minimum cell area in pixels.  

---

### 5. `iou.py`
**Purpose:**  
Evaluates segmentation performance against ground truth.  

**Key Functions:**  
- Calculates Intersection over Union (IoU) metrics.  
- Generates visual comparisons (green = GT, red = prediction, purple = overlap).  
- Produces performance reports.  

---

### 6. `grow_rate_and_intensity.py`
**Purpose:**  
Measures cell growth rates and analyzes multi-channel intensities.  

**Key Functions:**  
- Enables interactive cell selection and measurement.  
- Calculates growth rates from major/minor axes.  
- Analyzes intensity profiles across fluorescence channels.  
- Generates spatial intensity maps.  
- Supports both interactive and pipeline modes.  

---

## Additional Modules

### `N2V_Denoising.py`
**Purpose:**  
Advanced self-supervised denoising using Noise2Void.  

**Use Case:**  
When BM3D denoising is insufficient for very noisy images.  

---

### `train_model.py`
**Purpose:**  
Trains custom Cellpose models on your specific data.  

**Key Features:**  
- Data augmentation (rotation, flipping, elastic deformation).  
- Transfer learning from pre-trained models.  
- Performance validation and visualization.  
- Model checkpointing and saving.  

**Use Case:**  
When pre-trained Cellpose models don’t perform well on your specific cell types.  

---

### `run_all.py`
**Purpose:**  
Executes the complete pipeline automatically.  

 ## Usage
 ### Method 1: Automated Full Pipeline (Recommended)
```bash
python "run all.py"
```
### Method 2: Step-by-Step Execution

```bash
# 1. Process raw microscopy data (CZI → TIFF + metadata extraction)
python extractor.py

# 2. Generate ground truth from ImageJ annotations
python truth_extractor.py

# 3. Enhance image quality with BM3D denoising
python denoising.py

# 4. Segment cells using Cellpose
python segmentation.py

# 5. Evaluate segmentation performance against ground truth
python iou.py

# 6. Analyze cell growth and multi-channel intensities
python grow_rate_and_intensity.py --pipeline

```
 ## Related Citations
Cellpose:
@article{stringer2021cellpose,
  title={Cellpose: a generalist algorithm for cellular segmentation},
  author={Stringer, Carsen and Wang, Tim and Michaelos, Michalis and Pachitariu, Marius},
  journal={Nature Methods},
  volume={18},
  number={1},
  pages={100--106},
  year={2021},
  publisher={Nature Publishing Group}
}

Noise2Void:
@inproceedings{krull2019noise2void,
  title={Noise2void-learning denoising from single noisy images},
  author={Krull, Alexander and Buchholz, Tim-Oliver and Jug, Florian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2129--2137},
  year={2019}
}

