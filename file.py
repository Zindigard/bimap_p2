import cv2
import numpy as np
from skimage import io
import os
from read_roi import read_roi_zip

def create_cellpose_mask(
        roi_zip_path: str,
        image_shape: tuple,
        min_contour_area: int = 5,
) -> np.ndarray:
    """
    Creates CellPose-compatible mask where each ROI has unique integer label.
    (0=background, 1,2,3...=individual objects)
    Now using uint8 type (limited to 255 objects + background)
    """
    rois = read_roi_zip(roi_zip_path)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Changed to uint8
    
    # Check if we have too many objects for uint8
    if len(rois) > 255:
        raise ValueError(f"Too many ROIs ({len(rois)}) for uint8 mask (max 255)")
    
    for idx, (name, roi) in enumerate(rois.items(), start=1):
        try:
            roi_type = roi.get('type', 'unknown')
            points = []
            
            # Polygon/Freehand
            if roi_type in ['polygon', 'freehand', 'traced']:
                x_coords = list(map(int, roi['x']))
                y_coords = list(map(int, roi['y']))
                points = np.array([x_coords, y_coords]).T.reshape((-1, 1, 2))
            
            # Rectangle
            elif roi_type == 'rectangle':
                left, top = int(roi['left']), int(roi['top'])
                width, height = int(roi['width']), int(roi['height'])
                points = np.array([
                    [left, top],
                    [left + width, top],
                    [left + width, top + height],
                    [left, top + height]
                ])
            
            # Oval → Polygon approximation
            elif roi_type == 'oval':
                left, top = int(roi['left']), int(roi['top'])
                width, height = int(roi['width']), int(roi['height'])
                center = (left + width//2, top + height//2)
                axes = (width//2, height//2)
                points = cv2.ellipse2Poly(center, axes, 0, 0, 360, 10)
            
            # Line → Thin rectangle
            elif roi_type == 'line':
                x1, y1 = int(float(roi['x1'])), int(float(roi['y1']))
                x2, y2 = int(float(roi['x2'])), int(float(roi['y2']))
                thickness = 2
                angle = np.arctan2(y2-y1, x2-x1)
                dx = thickness * np.sin(angle)
                dy = thickness * np.cos(angle)
                points = np.array([
                    [x1-dx, y1+dy],
                    [x1+dx, y1-dy],
                    [x2+dx, y2-dy],
                    [x2-dx, y2+dy]
                ])
            
            # Draw filled contour with unique label
            if len(points) > 0 and cv2.contourArea(points) >= min_contour_area:
                cv2.drawContours(mask, [points], -1, idx, thickness=cv2.FILLED)
                
        except Exception as e:
            print(f"Error processing ROI {name}: {str(e)}")
    
    return mask

def process_for_cellpose(
        input_folder: str,
        output_folder: str,
        roi_suffix: str = "_ROISET.zip",
        img_extensions: tuple = (".tif", ".tiff", ".png", ".jpg")
):
    """Process all images to create CellPose-compatible masks with '_masks' extension"""
    os.makedirs(output_folder, exist_ok=True)
    
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        if not os.path.isdir(item_path):
            continue
            
        print(f"Processing: {item}")
        
        # Find ROI zip and matching image
        roi_zip, img_path = None, None
        for f in os.listdir(item_path):
            if f.endswith(roi_suffix):
                base_name = f[:-len(roi_suffix)]
                roi_zip = os.path.join(item_path, f)
                
                # Find matching image
                for ext in img_extensions:
                    if os.path.exists(os.path.join(item_path, base_name + ext)):
                        img_path = os.path.join(item_path, base_name + ext)
                        break
                break
        
        if not roi_zip or not img_path:
            print(f"  ! Missing files in {item}")
            continue
            
        try:
            # Create CellPose-compatible mask
            image = io.imread(img_path)
            mask = create_cellpose_mask(roi_zip, image.shape)
            
            # Save with original name + "_masks.tif"
            mask_name = f"{base_name}_masks.tif"  # Changed to _masks.tif
            io.imsave(
                os.path.join(output_folder, mask_name),
                mask,
                check_contrast=False  # Disable low-contrast warning
            )
            print(f"  ✓ Saved masks: {mask_name}")
            
        except Exception as e:
            print(f"  ! Failed: {str(e)}")

def verify_cellpose_mask(mask_path: str):
    """Check if a mask meets CellPose requirements"""
    mask = io.imread(mask_path)
    unique = np.unique(mask)
    
    print(f"\nVerifying: {os.path.basename(mask_path)}")
    print(f"Unique labels: {unique}")
    print(f"Number of objects: {len(unique)-1}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Value distribution:")
    
    for val in unique:
        print(f"- {val}: {(mask == val).sum():,} pixels")

if __name__ == "__main__":
    # Configuration
    input_dir = r"C:\Users\zindi\PycharmProjects\P2\train_data\unziped data"
    output_dir = r"C:\Users\zindi\PycharmProjects\P2\train_data\cellpose_masks"
    
    # Process all images
    process_for_cellpose(input_dir, output_dir)
    
    # Verify first output mask
    if os.listdir(output_dir):  # Check if output directory is not empty
        sample_mask = os.path.join(output_dir, os.listdir(output_dir)[0])
        verify_cellpose_mask(sample_mask)