import cv2
import numpy as np
from skimage import io, morphology
import os
import matplotlib.pyplot as plt
from read_roi import read_roi_zip

def visualize_rois_white_on_rgb(
        roi_zip_path: str,
        image_path: str,
        output_path: str = None,
        show_plot: bool = False,
) -> np.ndarray:
    """
    Visualizes ROIs in white on original RGB images
    """
    rois = read_roi_zip(roi_zip_path)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    white = (255, 255, 255)
    line_thickness = 2

    for name, roi in rois.items():
        try:
            roi_type = roi.get('type', 'unknown')

            if roi_type in ['polygon', 'freehand', 'traced']:
                x_coords = list(map(int, roi['x']))
                y_coords = list(map(int, roi['y']))
                points = np.array([x_coords, y_coords]).T.reshape((-1, 1, 2))
                cv2.polylines(image, [points], isClosed=True, color=white, thickness=line_thickness)

            elif roi_type == 'rectangle':
                left, top = int(roi['left']), int(roi['top'])
                width, height = int(roi['width']), int(roi['height'])
                cv2.rectangle(image, (left, top), (left + width, top + height), white, line_thickness)

            elif roi_type == 'oval':
                left, top = int(roi['left']), int(roi['top'])
                width, height = int(roi['width']), int(roi['height'])
                cv2.ellipse(image,
                            (left + width // 2, top + height // 2),
                            (width // 2, height // 2),
                            0, 0, 360, white, line_thickness)

            elif roi_type == 'line':
                x1, y1 = int(float(roi['x1'])), int(float(roi['y1']))
                x2, y2 = int(float(roi['x2'])), int(float(roi['y2']))
                cv2.line(image, (x1, y1), (x2, y2), white, line_thickness)

            else:
                print(f"Unsupported ROI type: {roi_type}")

        except Exception as e:
            print(f"Error drawing ROI {name}: {str(e)}")

    # Save output
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return image


def create_interior_mask_from_roi_image(
        roi_image_path: str,
        output_mask_path: str = None,
        line_threshold: int = 150,
        min_contour_area: int = 5,
) -> np.ndarray:
    """
    Creates a binary mask where regions inside white ROIs are 1.
    Assumes input ROIs are already closed contours (from visualize_rois_white_on_rgb).

    Args:
        roi_image_path: Path to the image with white ROIs (from visualize_rois_white_on_rgb).
        output_mask_path: Where to save the mask (optional).
        line_threshold: Brightness threshold for detecting white lines (0-255).
        min_contour_area: Minimum area to consider a valid ROI (removes noise).

    Returns:
        Binary mask (np.uint8: 0 or 1).
    """

    roi_image = cv2.imread(roi_image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    #  thresholding
    _, binary = cv2.threshold(gray, line_threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask
    mask = np.zeros_like(gray, dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_contour_area:
            cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)

    if output_mask_path:
        io.imsave(output_mask_path, mask * 255)
    return mask


def process_all_folders(
        input_root: str = r"C:\Users\zindi\PycharmProjects\P2\unpacked images\True",
        output_root: str = r"C:\Users\zindi\PycharmProjects\P2\True processed",
        roi_suffix: str = "_ROISET.zip",
        img_extensions: tuple = (".tif", ".tiff", ".png", ".jpg"),
        mask_output_root: str = r"C:\Users\zindi\PycharmProjects\P2\Evaluations\Ground",
):

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

        base_name = roi_zip[:-len(roi_suffix)]
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
                output_path=None,
                show_plot=False,
            )


            temp_roi_drawn_path = os.path.join(output_root, f"temp_roi_drawn_{folder_name}.tif")
            cv2.imwrite(temp_roi_drawn_path, cv2.cvtColor(roi_drawn_image, cv2.COLOR_RGB2BGR))

            mask_output_name = f"{os.path.basename(base_name)}_mask.tif"
            mask_output_path = os.path.join(mask_output_root, mask_output_name)

            mask = create_interior_mask_from_roi_image(
                roi_image_path=temp_roi_drawn_path,
                output_mask_path=mask_output_path,
                line_threshold=200,  # Adjust as needed
            )

            print(f"  ✓ Mask saved to {mask_output_path}")

            os.remove(temp_roi_drawn_path)

        except Exception as e:
            print(f"  ! Processing failed: {str(e)}")


if __name__ == "__main__":
    process_all_folders()
    print("\nBatch processing complete!")