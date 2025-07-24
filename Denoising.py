import numpy as np
from skimage import io
from skimage.util import img_as_float32
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import bm3d
from pathlib import Path
from natsort import natsorted
import matplotlib.pyplot as plt
import warnings
from scipy import ndimage
from typing import Tuple, Dict, List


def load_image(file_path: Path) -> np.ndarray:
    """Load and normalize image to float32 [0,1] range."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = io.imread(file_path)

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    print(f"Loaded: {file_path.name} | Shape: {img.shape} | Type: {img.dtype} | "
          f"Original Range: [{img.min()}, {img.max()}]")

    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img_as_float32(img)

    img = np.clip(img, 0.0, 1.0)
    return img


def adaptive_denoise_channel(channel: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Enhanced denoising with adaptive parameters for single channel."""
    channel_norm = (channel - channel.mean()) / (channel.std() + 1e-8)

    # adaptive parameters
    denoised = bm3d.bm3d(
        channel_norm,
        sigma_psd=max(sigma, 0.05),
        stage_arg=bm3d.BM3DStages.ALL_STAGES
    )

    denoised = (denoised * channel.std()) + channel.mean()
    denoised = ndimage.gaussian_filter(denoised, sigma=0.5)

    return np.clip(denoised, 0.0, 1.0)


def enhanced_denoise_rgb(img: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Improved multi-channel denoising with color preservation."""
    denoised_channels = []

    for c in range(3):
        channel = img[:, :, c]
        denoised = adaptive_denoise_channel(channel, sigma)
        denoised_channels.append(denoised)

    denoised_img = np.stack(denoised_channels, axis=-1)
    denoised_img = (denoised_img - denoised_img.min()) / \
                   (denoised_img.max() - denoised_img.min() + 1e-10)

    # Blend with original
    diff = np.abs(img - denoised_img)
    print(f"Max diff: {diff.max():.4f}")
    print(f"Mean diff: {diff.mean():.4f}")
    denoised_img = 0.1 * img + 0.9 * denoised_img

    return np.clip(denoised_img, 0.0, 1.0)


def calculate_quality_metrics(original: np.ndarray, denoised: np.ndarray) -> Dict:
    """Calculate comprehensive quality metrics."""
    metrics = {}
    data_range = 1.0

    metrics['ssim'] = ssim(
        original, denoised,
        win_size=7,
        data_range=data_range,
        channel_axis=2,
        gaussian_weights=True
    )

    metrics['ssim_channels'] = [
        ssim(original[:, :, c], denoised[:, :, c],
             data_range=data_range,
             win_size=7,
             gaussian_weights=True) for c in range(3)
    ]

    metrics['mse'] = np.mean((original - denoised) ** 2)
    metrics['psnr'] = psnr(original, denoised, data_range=data_range)
    metrics['ncc'] = np.corrcoef(original.ravel(), denoised.ravel())[0, 1]

    return metrics


def visualize_comparison(original: np.ndarray, denoised: np.ndarray, metrics: Dict):
    """Simplified visualization focusing on absolute difference. """
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(original)
    ax1.set_title(f"Original Image\nRange: [{original.min():.2f}, {original.max():.2f}]")
    ax1.axis('off')
    
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(denoised)
    ax2.set_title(
        f"Denoised Image\n"
        f"SSIM: {metrics['ssim']:.4f}\n"
        f"PSNR: {metrics['psnr']:.2f} dB\n"
        f"MSE: {metrics['mse']:.4f}"
    )
    ax2.axis('off')
    
    difference = np.abs(original - denoised)
    ax3 = plt.subplot(1, 3, 3)
    im = ax3.imshow(difference, cmap='inferno', vmin=0, vmax=0.3)  
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title(f"Absolute Difference\nMax: {difference.max():.4f}, Mean: {difference.mean():.4f}")
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


def save_metrics_to_txt(metrics: Dict, file_path: Path, mode: str = 'a'):
    """Save metrics dictionary to a text file in a readable format."""
    with open(file_path, mode) as f:
        f.write("\n=== Image Metrics ===\n")
        for key, value in metrics.items():
            if isinstance(value, list):
                f.write(f"{key}: {[f'{x:.4f}' for x in value]}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
        f.write("\n")


def save_summary_to_txt(results: List[Dict], file_path: Path):
    """Save aggregated statistics to a text file."""
    if not results:
        return

    ssim_values = [x['ssim'] for x in results]
    psnr_values = [x['psnr'] for x in results]
    mse_values = [x['mse'] for x in results]

    with open(file_path, 'w') as f:
        f.write("=== Final Summary Statistics ===\n\n")
        f.write(f"Total Images Processed: {len(results)}\n\n")

        f.write("SSIM Statistics:\n")
        f.write(f"- Average: {np.mean(ssim_values):.4f}\n")
        f.write(f"- Median: {np.median(ssim_values):.4f}\n")
        f.write(f"- Range: [{np.min(ssim_values):.4f}, {np.max(ssim_values):.4f}]\n")
        f.write(f"- Std Dev: {np.std(ssim_values):.4f}\n\n")

        f.write("PSNR Statistics:\n")
        f.write(f"- Average: {np.mean(psnr_values):.2f} dB\n")
        f.write(f"- Median: {np.median(psnr_values):.2f} dB\n")
        f.write(f"- Range: [{np.min(psnr_values):.2f}, {np.max(psnr_values):.2f}] dB\n")
        f.write(f"- Std Dev: {np.std(psnr_values):.2f}\n\n")

        f.write("MSE Statistics:\n")
        f.write(f"- Average: {np.mean(mse_values):.6f}\n")
        f.write(f"- Median: {np.median(mse_values):.6f}\n")
        f.write(f"- Range: [{np.min(mse_values):.6f}, {np.max(mse_values):.6f}]\n")
        f.write(f"- Std Dev: {np.std(mse_values):.6f}\n")


def process_single_image(file_path: Path, output_dir: Path, sigma: float = 0.1,
                         overwrite: bool = False) -> Tuple[bool, Dict]:
    """Complete processing pipeline with enhanced metrics."""
    output_path = output_dir / f"{file_path.stem}_denoised.tif"
    metrics_file = output_dir / "denoising_metrics.txt"

    if output_path.exists() and not overwrite:
        print(f"Skipped (exists): {output_path.name}")
        return True, {}

    try:
        print(f"\nProcessing: {file_path.name}")
        img = load_image(file_path)
        denoised = enhanced_denoise_rgb(img, sigma)
        metrics = calculate_quality_metrics(img, denoised)

        print("\nQuality Metrics:")
        print(f"- SSIM: {metrics['ssim']:.4f} (Channels: {[f'{x:.4f}' for x in metrics['ssim_channels']]})")
        print(f"- PSNR: {metrics['psnr']:.2f} dB")
        print(f"- MSE: {metrics['mse']:.6f}")
        print(f"- NCC: {metrics['ncc']:.4f}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if img.dtype == np.uint16:
                io.imsave(output_path, (denoised * 65535).astype(np.uint16))
            else:
                io.imsave(output_path, (denoised * 255).astype(np.uint8))

        print(f"Saved: {output_path}")
        save_metrics_to_txt(metrics, metrics_file)  # Save  to TXT

        visualize_comparison(img, denoised, metrics)

        return True, metrics

    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        return False, {}


def batch_process(input_dir: str, output_dir: str, sigma: float = 0.1, overwrite: bool = False):
    """Batch process with  reporting and metrics saving. """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = natsorted(
        [f for f in input_dir.glob("*.[tT][iI][fF]*") if "_masks" not in f.name and "_flows" not in f.name])

    if not files:
        raise FileNotFoundError("No compatible TIFF images found!")

    print(f"\nFound {len(files)} images in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: sigma={sigma}, overwrite={overwrite}")

    results = []
    for i, file in enumerate(files, 1):
        print(f"\nImage {i}/{len(files)}:")
        success, metrics = process_single_image(file, output_dir, sigma, overwrite)
        if success and metrics:
            results.append(metrics)

    if results:
        save_summary_to_txt(results, output_dir / "denoising_metrics.txt") 
        print(f"\nAll metrics saved to: {output_dir / 'denoising_metrics.txt'}")

        print("\n\n=== Final Summary ===")
        print(f"Processed {len(results)} images successfully")

        ssim_values = [x['ssim'] for x in results]
        psnr_values = [x['psnr'] for x in results]
        mse_values = [x['mse'] for x in results]

        print("\nSSIM Statistics:")
        print(f"- Average: {np.mean(ssim_values):.4f}")
        print(f"- Median: {np.median(ssim_values):.4f}")
        print(f"- Range: [{np.min(ssim_values):.4f}, {np.max(ssim_values):.4f}]")
        print(f"- Std Dev: {np.std(ssim_values):.4f}")

        print("\nPSNR Statistics:")
        print(f"- Average: {np.mean(psnr_values):.2f} dB")
        print(f"- Median: {np.median(psnr_values):.2f} dB")
        print(f"- Range: [{np.min(psnr_values):.2f}, {np.max(psnr_values):.2f}] dB")

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.hist(ssim_values, bins=20, color='skyblue', edgecolor='black')
        plt.title('SSIM Distribution')
        plt.xlabel('SSIM Value')

        plt.subplot(132)
        plt.hist(psnr_values, bins=20, color='lightgreen', edgecolor='black')
        plt.title('PSNR Distribution')
        plt.xlabel('PSNR (dB)')

        plt.subplot(133)
        plt.scatter(ssim_values, psnr_values, alpha=0.6)
        plt.title('SSIM vs PSNR')
        plt.xlabel('SSIM')
        plt.ylabel('PSNR (dB)')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    input_dir = r"C:\Users\zindi\PycharmProjects\P2\test_data"
    output_dir = r"C:\Users\zindi\PycharmProjects\P2\denoised"
    sigma = 0.4  
    overwrite = False

    batch_process(input_dir, output_dir, sigma, overwrite)