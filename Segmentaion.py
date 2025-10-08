import numpy as np
import os
from cellpose import models, core, io, utils
from pathlib import Path
import matplotlib.pyplot as plt
from natsort import natsorted
from skimage.io import imsave
from skimage.morphology import remove_small_objects, binary_closing
from skimage.segmentation import clear_border

def post_process_masks(masks):
    """Clean up segmentation masks"""
    masks = masks.astype(np.int32)
    cleaned_masks = np.zeros_like(masks)
    max_label = masks.max()

    if max_label > 0:
        for i in range(1, max_label + 1):
            mask = masks == i
            mask = remove_small_objects(mask, min_size=50)
            mask = binary_closing(mask, footprint=np.ones((3, 3)))
            mask = clear_border(mask)
            if mask.sum() > 0:
                cleaned_masks[mask] = i

    return cleaned_masks

def list_image_files(directory: Path, extension=".tif"):
    """Get image files"""
    files = natsorted([f for f in directory.glob(f"*{extension}")
                       if "_masks" not in f.name and "_flows" not in f.name])
    if not files:
        raise FileNotFoundError("No image files found, did you specify the correct folder and extension?")
    else:
        print(f"{len(files)} images in folder:")
    for f in files:
        print(f.name)
    return files

def prepare_image(file: Path):
    """Normalize and transpose image"""
    img = io.imread(file)
    img_rgb = np.transpose(img, (1, 2, 0)) if img.ndim == 3 else img
    if img_rgb.dtype == np.uint16:
        img_rg1 = (img_rgb / 256).astype(np.uint8)
    else:
        img_rg1 = img_rgb.astype(np.uint8)
    return img_rg1

def select_channels(img_rg1, channels=['0','1','2']):
    """Prepare image for segmentation"""
    selected_indices = []
    for c in channels:
        if c == 'None':
            continue
        c_int = int(c)
        if c_int >= img_rg1.shape[-1]:  
            raise ValueError(f'Invalid channel index {c_int}. Image has {img_rg1.shape[-1]} channels.')
        selected_indices.append(c_int)
    
    print('Selected channels for composite:', selected_indices)
    
    composite = np.zeros(img_rg1.shape[:2], dtype=img_rg1.dtype)
    for idx in selected_indices:
        composite += img_rg1[:, :, idx]
    
    return composite[..., np.newaxis]

def run_segmentation(model, img_selected_channels):
    """Run model inference"""
    print("Running Cellpose segmentation...")
    masks, flows, styles = model.eval( 
        img_selected_channels,
        batch_size=8,
        diameter=None,
        flow_threshold=0.7,
        cellprob_threshold=-0.6,
        min_size=15,
        resample=True,
    )
    return masks, flows, styles

def save_binary_mask(masks, path):
    """Save mask for later processing"""
    binary_mask = (masks > 0).astype(np.uint8) * 255
    io.imsave(path, binary_mask)
    print(f"Saved binary mask to: {path}")
    return binary_mask

def save_outlined_image(img_rg1, masks, path):
    """Save outlines for visualization"""
    outlines = utils.outlines_list(masks)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rg1)
    for outline in outlines:
        plt.plot(outline[:, 0], outline[:, 1], 'y', linewidth=1)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved outlined mask to: {path}")
    return outlines

def visualize_results(img_selected_channels, masks, flows, binary_mask, outlines, vis_img):
    """Visualize segmentation results"""
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Binary Mask')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(vis_img)
    for outline in outlines:
        plt.plot(outline[:, 0], outline[:, 1], 'y', linewidth=1)
    plt.title('Outlines on Original Image')
    plt.axis('off')
    plt.show()

def load_visualization_image(img_rg1, brightness_dir, file):
    """Load image for visualization"""
    brightness_file = brightness_dir / file.name
    vis_img = None
    if brightness_file.exists():
        vis_img = io.imread(brightness_file)
        if len(vis_img.shape) == 3:
            vis_img = vis_img.transpose(1, 2, 0)
        vis_img = (vis_img / 256).astype(np.uint8) if vis_img.dtype == np.uint16 else vis_img.astype(np.uint8)
    else:
        vis_img = img_rg1
    return vis_img

def process_image(file, model, output_dir, brightness_dir):
    """Process a single image"""
    original_name = file.stem
    outlines_file_path = output_dir / f"{original_name}_outlined.tif"
    mask_file_path = output_dir / f"{original_name}_binary.tif"
    npy_save_path = output_dir / f"{original_name}_masks.npy"

    if outlines_file_path.exists() and mask_file_path.exists():
        print(f"\nSegmentation file already exists for {file.name}, skipping...")
        return

    print(f"\nProcessing {file.name}...")

    img_rg1 = prepare_image(file)
    print(f"Image shape: {img_rg1.shape}")

    img_selected_channels = select_channels(img_rg1)

    masks, flows, styles = run_segmentation(model, img_selected_channels)

    masks = post_process_masks(masks)

    vis_img = load_visualization_image(img_rg1, brightness_dir, file)

    binary_mask = save_binary_mask(masks, mask_file_path)
    outlines = save_outlined_image(img_rg1, masks, outlines_file_path)

    visualize_results(img_selected_channels, masks, flows, binary_mask, outlines, vis_img)

    np.save(npy_save_path, masks)
    print(f"Masks saved to: {npy_save_path}")

def main():
    """Main function with all path configurations"""
    base_dir = Path(r"C:\Users\zindi\PycharmProjects\P2")
    image_dir = base_dir / "test_data"
    output_dir = base_dir / "Evaluations" / "SAM"
    brightness_dir = base_dir / "train_brightness"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    use_gpu = core.use_gpu()
    print(f"Using GPU: {use_gpu}")
    model = models.CellposeModel(gpu=use_gpu)
    
    files = list_image_files(image_dir)
    for file in files:
        process_image(file, model, output_dir, brightness_dir)

if __name__ == "__main__":
    main()



