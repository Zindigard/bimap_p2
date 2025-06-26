import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from cellpose import models, core, io, plot
from pathlib import Path
from cellpose import utils
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

    # Only proceed if there are any labels
    if max_label > 0:
        for i in range(1, max_label + 1):
            mask = masks == i
            mask = remove_small_objects(mask, min_size=50)
            mask = binary_closing(mask, footprint=np.ones((3, 3)))
            mask = clear_border(mask)
            if mask.sum() > 0:
                cleaned_masks[mask] = i

    return cleaned_masks


def setup_paths():
    image_dir = Path(r"C:\Users\zindi\PycharmProjects\P2\test_data")
    output_dir = Path(r"C:\Users\zindi\PycharmProjects\P2\Evaluations\SAM")
    brightness_dir = Path(r"C:\Users\zindi\PycharmProjects\P2\train_brightness")
    output_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, output_dir, brightness_dir


def list_image_files(directory: Path, extension=".tif"):
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
    img = io.imread(file)
    img_rgb = np.transpose(img, (1, 2, 0)) if img.ndim == 3 else img
    if img_rgb.dtype == np.uint16:
        img_rg1 = (img_rgb / 256).astype(np.uint8)
    else:
        img_rg1 = img_rgb.astype(np.uint8)
    return img_rg1


def select_channels(img_rg1, channels=['0', '1', '2']):
    selected_channels = []
    for i, c in enumerate(channels):
        if c == 'None':
            continue
        if int(c) > img_rg1.ndim:
            assert False, 'invalid channel index, must have index greater or equal to the number of channels'
        if c != 'None':
            selected_channels.append(int(c))

    img_selected_channels = np.zeros_like(img_rg1)
    print('Selected channels:', selected_channels)
    img_selected_channels[:, :, :len(selected_channels)] = img_rg1[:, :, selected_channels]
    return img_selected_channels


def run_segmentation(model, img_selected_channels):
    print("Running Cellpose segmentation...")
    masks, flows, styles = model.eval(
        img_selected_channels,
        batch_size=8,
        diameter=None,
        flow_threshold=0.8,
        cellprob_threshold=0.0,
        min_size=15,
        stitch_threshold=0.0,  # Don't stitch cells
    )
    return masks, flows, styles


def save_binary_mask(masks, path):
    binary_mask = (masks > 0).astype(np.uint8) * 255
    io.imsave(path, binary_mask)
    print(f"Saved binary mask to: {path}")
    return binary_mask


def save_outlined_image(img_rg1, masks, path):
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


def image_segmentation():
    io.logger_setup()
    model = models.CellposeModel(gpu=False)
    dir_path, output_dir, brightness_path = setup_paths()
    files = list_image_files(dir_path)

    for file in files:
        process_image(file, model, output_dir, brightness_path)


if __name__ == "__main__":
    image_segmentation()




