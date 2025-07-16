import numpy as np
from skimage import io, color, measure
from scipy.ndimage import binary_fill_holes
from PIL import Image, ImageDraw
import os
from read_roi import read_roi_zip

def visualize_rois_white_on_rgb(
        roi_zip_path: str,
        image_path: str,
) -> np.ndarray:
    """
    Visualizes ROIs in white on original RGB images using Pillow
    Returns the image with ROIs drawn
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
    Creates a binary mask using scikit-image and scipy
    Regions inside white ROIs are 1.
    """
    # Convert to grayscale and threshold
    gray = color.rgb2gray(roi_image)
    binary_line = gray >= (line_threshold / 255.0)
    
    # Fill enclosed regions
    filled = binary_fill_holes(binary_line)
    interior = filled & ~binary_line  # Exclude the lines themselves
    
    # Label regions and filter by size
    label_img = measure.label(interior)
    mask = np.zeros_like(interior, dtype=np.uint8)
    
    for region in measure.regionprops(label_img):
        if region.area >= min_contour_area:
            mask[label_img == region.label] = 1
    
    return mask.astype(np.uint8)

def process_all_folders(
        input_root: str = r"C:\Users\zindi\PycharmProjects\P2\unpacked images\True",
        processed_output_root: str = r"C:\Users\zindi\PycharmProjects\P2\True processed",
        mask_output_root: str = r"C:\Users\zindi\PycharmProjects\P2\Evaluations\Ground",
        roi_suffix: str = "_ROISET.zip",
        img_extensions: tuple = (".tif", ".tiff", ".png", ".jpg"),
):
    """Process all folders using the new implementations"""
    os.makedirs(processed_output_root, exist_ok=True)
    os.makedirs(mask_output_root, exist_ok=True)

    for folder_name in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder_name)

        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing: {folder_name}")

        roi_zip = None
        for f in os.listdir(folder_path):
            if f.endswith(roi_suffix):
                roi_zip = os.path.join(folder_path, f)
                break

        if not roi_zip:
            print(f"  ! No ROI zip found")
            continue

        base_name = os.path.splitext(roi_zip)[0][:-len("_ROISET")] 
        img_file = None

        for ext in img_extensions:
            test_path = base_name + ext
            if os.path.exists(test_path):
                img_file = test_path
                break

        if not img_file:
            print(f"  ! No matching image found")
            continue

        try:
            roi_drawn_image = visualize_rois_white_on_rgb(
                roi_zip_path=roi_zip,
                image_path=img_file,
            )

            processed_img_name = f"{os.path.basename(base_name)}_processed.tif"
            processed_img_path = os.path.join(processed_output_root, processed_img_name)
            io.imsave(processed_img_path, roi_drawn_image)

            mask = create_interior_mask_from_roi_image(
                roi_image=roi_drawn_image,
                line_threshold=200,
            )

            mask_name = f"{os.path.basename(base_name)}_mask.tif"
            mask_path = os.path.join(mask_output_root, mask_name)
            io.imsave(mask_path, mask * 255)

            print(f"  ✓ Processed image saved to {processed_img_path}")
            print(f"  ✓ Mask saved to {mask_path}")

        except Exception as e:
            print(f"  ! Processing failed: {str(e)}")

if __name__ == "__main__":
    process_all_folders()
    print("\nBatch processing complete!")