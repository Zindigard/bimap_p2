import numpy as np
from skimage import io, color, measure
from scipy.ndimage import binary_fill_holes
from PIL import Image, ImageDraw
import os
from read_roi import read_roi_zip

def get_project_root():
    """
    Get the root directory where scripts are located.
    
    Returns:
        str: Absolute path to the directory containing this script
    """
    return os.path.dirname(os.path.abspath(__file__))

def setup_folders():
    """
    Create necessary folder structure for ground truth extraction.
    
    Returns:
        dict: Dictionary with paths to input and output folders
    """
    root_dir = get_project_root()
    
    folders = {
        'input_root': os.path.join(root_dir, "unpacked images", "True"),
        'processed_output_root': os.path.join(root_dir, "True processed"),
        'mask_output_root': os.path.join(root_dir, "Evaluations", "Ground")
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
        print(f"Ensured folder exists: {folder}")
    
    return folders

def visualize_rois_white_on_rgb(
        roi_zip_path: str,
        image_path: str,
) -> np.ndarray:
    """
    Read ROIs from ZIP file and draw them as white outlines on RGB image.
    
    Args:
        roi_zip_path (str): Path to ImageJ ROI zip file
        image_path (str): Path to corresponding image file
        
    Returns:
        np.ndarray: Annotated image with ROI outlines as numpy array
    """
    rois = read_roi_zip(roi_zip_path)
    pil_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    white = (255, 255, 255)
    line_thickness = 2

    for name, roi in rois.items():
        try:
            roi_type = roi.get('type', 'unknown')

            if roi_type in ['polygon', 'freehand', 'traced']:
                x_coords = list(map(int, roi['x']))
                y_coords = list(map(int, roi['y']))
                points = list(zip(x_coords, y_coords))
                # Draw closed polygon
                if len(points) > 1:
                    for i in range(len(points)):
                        start = points[i]
                        end = points[(i + 1) % len(points)]
                        draw.line([start, end], fill=white, width=line_thickness)

            elif roi_type == 'rectangle':
                left, top = int(roi['left']), int(roi['top'])
                width, height = int(roi['width']), int(roi['height'])
                coords = [left, top, left + width, top + height]
                draw.rectangle(coords, outline=white, width=line_thickness)

            elif roi_type == 'oval':
                left, top = int(roi['left']), int(roi['top'])
                width, height = int(roi['width']), int(roi['height'])
                coords = [left, top, left + width, top + height]
                draw.ellipse(coords, outline=white, width=line_thickness)

            elif roi_type == 'line':
                x1 = int(float(roi['x1']))
                y1 = int(float(roi['y1']))
                x2 = int(float(roi['x2']))
                y2 = int(float(roi['y2']))
                draw.line([(x1, y1), (x2, y2)], fill=white, width=line_thickness)

            else:
                print(f"Unsupported ROI type: {roi_type}")

        except Exception as e:
            print(f"Error drawing ROI {name}: {str(e)}")

    return np.array(pil_image)

def create_interior_mask_from_roi_image(
        roi_image: np.ndarray,
        line_threshold: int = 150,
        min_contour_area: int = 5,
) -> np.ndarray:
    """
    Generate binary mask from image with drawn ROIs.
    
    Args:
        roi_image (np.ndarray): Image with ROI outlines
        line_threshold (int): Grayscale threshold for detecting drawn lines (0-255)
        min_contour_area (int): Minimum area in pixels for valid contours
        
    Returns:
        np.ndarray: Binary mask with filled interior regions
    """
    gray = color.rgb2gray(roi_image)
    binary_line = gray >= (line_threshold / 255.0)
    
    filled = binary_fill_holes(binary_line)
    interior = filled & ~binary_line 
    
    label_img = measure.label(interior)
    mask = np.zeros_like(interior, dtype=np.uint8)
    
    for region in measure.regionprops(label_img):
        if region.area >= min_contour_area:
            mask[label_img == region.label] = 1
    
    return mask.astype(np.uint8)

def process_all_folders():
    """
    Main processing function that handles all subfolders automatically.
    Processes ROI files and generates ground truth masks.
    """
    folders = setup_folders()
    roi_suffix = "_ROISET.zip" # Change if using different ROI file naming
    img_extensions = (".tif", ".tiff", ".png", ".jpg") # Add other formats if needed
    min_contour_area = 5 # Increase to filter out small noise artifacts
    line_threshold = 200  # Increase if lines not detected, decrease if too sensitive
    
    for folder_name in os.listdir(folders['input_root']):
        folder_path = os.path.join(folders['input_root'], folder_name)

        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing: {folder_name}")

        roi_zip = None
        for f in os.listdir(folder_path):
            if f.endswith(roi_suffix):
                roi_zip = os.path.join(folder_path, f)
                break

        if not roi_zip:
            print(f"   No ROI zip found")
            continue

        base_name = os.path.splitext(roi_zip)[0][:-len("_ROISET")] 
        img_file = None

        for ext in img_extensions:
            test_path = base_name + ext
            if os.path.exists(test_path):
                img_file = test_path
                break

        if not img_file:
            print(f"   No matching image found")
            continue

        try:
            # Generate annotated image with ROI outlines
            roi_drawn_image = visualize_rois_white_on_rgb(
                roi_zip_path=roi_zip,
                image_path=img_file,
            )

            processed_img_name = f"{os.path.basename(base_name)}_processed.tif"
            processed_img_path = os.path.join(folders['processed_output_root'], processed_img_name)
            io.imsave(processed_img_path, roi_drawn_image)

            mask = create_interior_mask_from_roi_image(
                roi_image=roi_drawn_image,
                line_threshold=line_threshold,
                min_contour_area=min_contour_area
            )

            mask_name = f"{os.path.basename(base_name)}_mask.tif"
            mask_path = os.path.join(folders['mask_output_root'], mask_name)
            io.imsave(mask_path, mask * 255)

            print(f"   Processed image saved to {processed_img_path}")
            print(f"   Mask saved to {mask_path}")

        except Exception as e:
            print(f"   Processing failed: {str(e)}")

if __name__ == "__main__":
    process_all_folders()
    