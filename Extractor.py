import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from czifile import CziFile
from skimage import exposure
import os
import tifffile
from datetime import datetime
import shutil


def parse_czi_metadata(metadata_xml: str, filename: str) -> dict:
    """Extract detailed metadata from CZI XML."""
    root = ET.fromstring(metadata_xml)
    metadata = {
        'filename': filename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Extract detailed pixel scaling information
    pixel_sizes = {}
    scaling = root.find(".//Scaling")
    if scaling is not None:
        for distance in scaling.findall(".//Distance"):
            dim = distance.get('Id')
            value = distance.find('Value').text
            pixel_sizes[dim] = f"{float(value) * 1e6:.4f} µm"  # Convert to µm

    metadata.update({
        'pixel_size_x': pixel_sizes.get('X', 'N/A'),
        'pixel_size_y': pixel_sizes.get('Y', 'N/A'),
        'pixel_size_z': pixel_sizes.get('Z', 'N/A'),
        'pixel_size_unit': 'µm'
    })

    # Extract detailed microscope information
    microscope_info = {
        'microscope_model': root.findtext(".//Microscope/System", default="N/A"),
        'objective_model': root.findtext(".//Objective/Manufacturer/Model", default="N/A"),
        'objective_na': root.findtext(".//Objective/LensNA", default="N/A"),
        'objective_magnification': root.findtext(".//Objective/NominalMagnification", default="N/A"),
        'objective_immersion': root.findtext(".//Objective/Immersion", default="N/A"),
        'detector_model': root.findtext(".//Detectors/Detector/Manufacturer/Model", default="N/A"),
        'illumination_type': root.findtext(".//LightSources/LightSource/Type", default="N/A")
    }
    metadata.update(microscope_info)

    # Extract acquisition date and time
    acquisition_date = root.findtext(".//AcquisitionDateAndTime", default="N/A")
    if acquisition_date != "N/A":
        try:
            # Try to format the date in a more readable way
            dt = datetime.strptime(acquisition_date, "%Y-%m-%dT%H:%M:%S")
            acquisition_date = dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass
    metadata['acquisition_date'] = acquisition_date

    # Extract channel information if available
    channels = []
    for channel in root.findall(".//Channel"):
        channel_info = {
            'name': channel.get('Name', 'N/A'),
            'excitation_wavelength': channel.findtext(".//ExcitationWavelength", default="N/A"),
            'emission_wavelength': channel.findtext(".//EmissionWavelength", default="N/A")
        }
        channels.append(channel_info)

    if channels:
        metadata['channels'] = channels

    return metadata


def display_czi_images(input_path: str):
    if os.path.isdir(input_path):
        czi_files = [f for f in os.listdir(input_path) if f.lower().endswith('.czi')]
        if not czi_files:
            print(f"No CZI files found in directory: {input_path}")
            return
    else:
        czi_files = [os.path.basename(input_path)]
        input_path = os.path.dirname(input_path) or '.'

    for czi_filename in czi_files:
        full_path = os.path.join(input_path, czi_filename)
        try:
            with CziFile(full_path) as czi:
                image = czi.asarray()
                metadata_xml = czi.metadata() if callable(czi.metadata) else czi.metadata

                metadata = parse_czi_metadata(metadata_xml, czi_filename) if isinstance(metadata_xml,
                                                                                        (str, bytes)) else {}

                print(f"\n{' METADATA ':=^40}")
                print(f"File: {czi_filename}")
                for k, v in metadata.items():
                    print(f"{k.replace('_', ' ').title():>20}: {v}")

                print(f"\n{' IMAGE DATA ':=^40}")
                print(f"Original shape: {image.shape}")
                image_squeezed = np.squeeze(image)
                print(f"Processed shape: {image_squeezed.shape}")

                if image_squeezed.ndim == 3 and image_squeezed.shape[0] == 3:
                    print("\nProcessing 3-channel image with correct color mapping...")

                    ch0 = exposure.rescale_intensity(image_squeezed[0], out_range=(0, 1))  # Blue
                    ch1 = exposure.rescale_intensity(image_squeezed[1], out_range=(0, 1))  # Green
                    ch2 = exposure.rescale_intensity(image_squeezed[2], out_range=(0, 1))  # Red

                    rgb_composite = np.stack([ch2, ch1, ch0], axis=-1)  # RGB order
                    print('shape', rgb_composite.shape)

                    # Create colored individual channels
                    ch0_blue = np.stack([ch0, np.zeros_like(ch0), np.zeros_like(ch0)], axis=-1)  # Blue
                    ch1_green = np.stack([np.zeros_like(ch1), ch1, np.zeros_like(ch1)], axis=-1)  # Green
                    ch2_red = np.stack([np.zeros_like(ch2), np.zeros_like(ch2), ch2], axis=-1)  # Red

                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

                    axes[0, 0].imshow(rgb_composite)
                    axes[0, 0].set_title("Composite (R=Ch2, G=Ch1, B=Ch0)")
                    axes[0, 0].axis('off')

                    axes[0, 1].imshow(ch2_red)
                    axes[0, 1].set_title("Channel 0 (Blue)")
                    axes[0, 1].axis('off')

                    axes[1, 0].imshow(ch1_green)
                    axes[1, 0].set_title("Channel 1 (Green)")
                    axes[1, 0].axis('off')

                    axes[1, 1].imshow(ch0_blue)
                    axes[1, 1].set_title("Channel 2 (Red)")
                    axes[1, 1].axis('off')

                    plt.suptitle(f"CZI Image - {czi_filename}", y=1.02)
                    plt.tight_layout()
                    plt.show()

                elif image_squeezed.ndim == 3:
                    fig, axes = plt.subplots(1, image_squeezed.shape[0], figsize=(15, 5))
                    for c in range(image_squeezed.shape[0]):
                        channel = image_squeezed[c]
                        channel_norm = exposure.rescale_intensity(
                            channel.astype(np.float32),
                            out_range=(0, 1)
                        )
                        axes[c].imshow(channel_norm, cmap='gray')
                        axes[c].set_title(f"Channel {c}\n{channel.shape}")
                        axes[c].axis('off')

                    plt.suptitle(f"CZI Image - {czi_filename}", y=1.05)
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"\nWarning: Unexpected shape {image_squeezed.shape} - cannot display")

        except Exception as e:
            print(f"\nError processing file {czi_filename}: {str(e)}")


def save_metadata_to_file(metadata_list: list, output_folder: str):
    """Save all metadata to a single text file."""
    output_path = os.path.join(output_folder, "metadata_summary.txt")
    with open(output_path, 'w') as f:
        f.write("CZI Image Metadata Summary\n")
        f.write("=" * 40 + "\n\n")

        for metadata in metadata_list:
            f.write(f"File: {metadata['filename']}\n")
            f.write(f"- Microscope Model: {metadata.get('microscope_model', 'N/A')}\n")
            f.write(f"- Pixel Size: {metadata.get('pixel_size_x', 'N/A')} (X), "
                    f"{metadata.get('pixel_size_y', 'N/A')} (Y)\n")
            f.write(f"- Processing Timestamp: {metadata['timestamp']}\n")
            
            # Add channel information if available
            if 'channels' in metadata:
                f.write("- Channels:\n")
                for i, channel in enumerate(metadata['channels']):
                    f.write(f"  Channel {i}:\n")
                    f.write(f"    Name: {channel.get('name', 'N/A')}\n")
                    f.write(f"    Excitation: {channel.get('excitation_wavelength', 'N/A')}\n")
                    f.write(f"    Emission: {channel.get('emission_wavelength', 'N/A')}\n")
            
            f.write("\n")


def convert_czi_to_tiff(czi_path: str, output_folder: str, metadata: dict):
    """Convert CZI file to TIFF and save with metadata."""
    with CziFile(czi_path) as czi:
        image = czi.asarray()
        image_squeezed = np.squeeze(image)

        if image_squeezed.ndim == 3 and image_squeezed.shape[0] == 3:
            image_squeezed = image_squeezed[::-1]

        base_name = os.path.splitext(os.path.basename(czi_path))[0]
        tiff_path = os.path.join(output_folder, f"{base_name}.tif")

        tifffile.imwrite(
            tiff_path,
            image_squeezed,
            metadata=metadata
        )

    return tiff_path


def plot_tiff_image(tiff_path: str):
    """Display TIFF image to verify conversion."""
    with tifffile.TiffFile(tiff_path) as tif:
        image = tif.asarray()
        metadata = tif.pages[0].tags

        print(f"\n{' TIFF METADATA ':=^40}")
        for tag in metadata.values():
            print(f"{tag.name:>20}: {tag.value}")

        plt.figure(figsize=(10, 6))

        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        if image.ndim == 3 and image.shape[0] == 3:
            disp_image = np.transpose(image, (1, 2, 0))
            plt.imshow(disp_image)
            plt.title(f"TIFF Image (RGB)\n{os.path.basename(tiff_path)}")
        elif image.ndim == 3:
            plt.imshow(image[0], cmap='gray')
            plt.title(f"TIFF Image (Channel 0)\n{os.path.basename(tiff_path)}")
        else:
            plt.imshow(image, cmap='gray')
            plt.title(f"TIFF Image\n{os.path.basename(tiff_path)}")

            plt.axis('off')
            plt.show()


def find_matching_true_folder(czi_filename: str, true_root_folder: str) -> str:
    """
    Find matching folder in the True directory based on filename patterns.
    Returns the full path if found, None otherwise.
    """
    base_pattern = czi_filename.split('.')[0]  #  extension
    base_pattern = '_'.join(base_pattern.split('_')[:-1])  #

    for root, dirs, files in os.walk(true_root_folder):
        for dir_name in dirs:
            if base_pattern in dir_name:
                return os.path.join(root, dir_name)
    return None


def process_czi_file(czi_path: str, output_folder: str, metadata_list: list, true_root_folder: str = None):
    """Process a single CZI file automatically without user confirmation."""
    base_name = os.path.basename(czi_path)
    print(f"Processing {base_name}...")

    with CziFile(czi_path) as czi:
        metadata_xml = czi.metadata() if callable(czi.metadata) else czi.metadata
        metadata = parse_czi_metadata(metadata_xml, base_name) if isinstance(metadata_xml, (str, bytes)) else {}

        # Convert and save to main output folder (train data)
        tiff_path = convert_czi_to_tiff(czi_path, output_folder, metadata)
        print(f"Saved TIFF: {tiff_path}")

        # If true_root_folder is provided, look for matching folder and save there
        if true_root_folder:
            matching_folder = find_matching_true_folder(base_name, true_root_folder)
            if matching_folder:
                true_tiff_path = os.path.join(matching_folder, os.path.basename(tiff_path))
                shutil.copy2(tiff_path, true_tiff_path)
                print(f"Also saved to True folder: {true_tiff_path}")

        metadata_list.append(metadata)

    return True


def process_all_czi_files(raw_folder: str, output_folder: str, true_root_folder: str = None):
    """Process all CZI files in the folder automatically."""
    os.makedirs(output_folder, exist_ok=True)

    metadata_list = []
    processed_files = []

    # Get all CZI files
    czi_files = [f for f in os.listdir(raw_folder) if f.lower().endswith('.czi')]

    if not czi_files:
        print("No CZI files found in the raw folder.")
        return

    print(f"Found {len(czi_files)} CZI files to process.")

    for czi_file in czi_files:
        czi_path = os.path.join(raw_folder, czi_file)
        try:
            process_czi_file(czi_path, output_folder, metadata_list, true_root_folder)
            processed_files.append(czi_file)
        except Exception as e:
            print(f"Error processing {czi_file}: {str(e)}")

    if metadata_list:
        save_metadata_to_file(metadata_list, output_folder)
        print(f"\nMetadata summary saved to {os.path.join(output_folder, 'metadata_summary.txt')}")

    print("\nProcessing Summary:")
    print(f"- Processed files: {len(processed_files)}")


def enhance_brightness(image: np.ndarray, brightness_factor: float = 1.2) -> np.ndarray:
    """
    Enhance brightness of an image by scaling pixel values.

    Args:
        image: Input image as numpy array
        brightness_factor: Factor to multiply pixel values by (>1 = brighter, <1 = darker)

    Returns:
        Brightness-enhanced image
    """
    original_dtype = image.dtype

    if original_dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif original_dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    enhanced = np.clip(image * brightness_factor, 0, 1)

    if original_dtype == np.uint16:
        enhanced = (enhanced * 65535).astype(np.uint16)
    elif original_dtype == np.uint8:
        enhanced = (enhanced * 255).astype(np.uint8)
    else:
        enhanced = enhanced.astype(original_dtype)

    return enhanced


def process_and_save_brightness_tiff(input_path: str, output_path: str, brightness_factor: float = 1.2):
    """
    Load a TIFF image, enhance brightness, and save to new location.

    Args:
        input_path: Path to input TIFF file
        output_path: Path to save enhanced TIFF file
        brightness_factor: Brightness multiplier (>1 = brighter, <1 = darker)
    """
    with tifffile.TiffFile(input_path) as tif:
        image = tif.asarray()

        # List of TIFF tags that should NOT be included in metadata
        exclude_tags = {
            'ImageWidth', 'ImageLength', 'BitsPerSample', 'Compression',
            'PhotometricInterpretation', 'StripOffsets', 'SamplesPerPixel',
            'RowsPerStrip', 'StripByteCounts', 'XResolution', 'YResolution',
            'PlanarConfiguration', 'ResolutionUnit', 'TileWidth', 'TileLength',
            'TileOffsets', 'TileByteCounts'
        }

        metadata = {}
        for tag in tif.pages[0].tags.values():
            if tag.name not in exclude_tags:
                try:
                    if isinstance(tag.value, (str, int, float)):
                        metadata[tag.name] = tag.value
                    elif hasattr(tag.value, '__str__'):
                        metadata[tag.name] = str(tag.value)
                except:
                    continue

        enhanced_image = enhance_brightness(image, brightness_factor)

        tifffile.imwrite(
            output_path,
            enhanced_image,
            metadata=metadata
        )


def process_all_tiff_brightness(input_folder: str, output_folder: str, brightness_factor: float = 1.2):
    """
    Process all TIFF files in input folder, enhance brightness, and save to output folder.

    Args:
        input_folder: Folder containing original TIFF files
        output_folder: Folder to save enhanced TIFF files
        brightness_factor: Brightness multiplier (>1 = brighter, <1 = darker)
    """
    os.makedirs(output_folder, exist_ok=True)

    tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

    if not tiff_files:
        print(f"No TIFF files found in {input_folder}")
        return

    print(f"Found {len(tiff_files)} TIFF files to process for brightness enhancement.")

    for tiff_file in tiff_files:
        input_path = os.path.join(input_folder, tiff_file)
        output_path = os.path.join(output_folder, tiff_file)

        try:
            process_and_save_brightness_tiff(input_path, output_path, brightness_factor)
            print(f"Brightness-enhanced and saved: {tiff_file}")
        except Exception as e:
            print(f"Error processing {tiff_file}: {str(e)}")

    print(f"\nFinished brightness enhancement. Images saved to {output_folder}")


if __name__ == "__main__":
    # Define paths
    raw_folder = r"C:\Users\zindi\PycharmProjects\P2\unpacked images\Raw"
    output_folder = r"C:\Users\zindi\PycharmProjects\P2\test_data"
    brightness_folder = r"C:\Users\zindi\PycharmProjects\P2\train_brightness"
    true_root_folder = r"C:\Users\zindi\PycharmProjects\P2\unpacked images\True"

    print(f"Processing {raw_folder}...")
    # Uncomment to display images before processing
    # display_czi_images(raw_folder)

    # Process all CZI files (saves to both train data and matching True folders)
    process_all_czi_files(raw_folder, output_folder, true_root_folder)

    # Create brightness folder if it doesn't exist
    os.makedirs(brightness_folder, exist_ok=True)

    # Process for brightness enhancement
    print("\nEnhancing brightness of TIFF files...")
    process_all_tiff_brightness(output_folder, brightness_folder, brightness_factor=4.5)