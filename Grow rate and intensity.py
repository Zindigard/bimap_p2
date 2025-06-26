import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from skimage.measure import regionprops, find_contours
import tifffile
import argparse
import re
from matplotlib.patches import Polygon


def read_pixel_size_from_metadata(metadata_folder, image_filename):
    """Read pixel size from metadata file for a specific image"""
    # Try different possible metadata filenames
    possible_metadata_files = [
        metadata_folder / "metadata_summary.txt",
        metadata_folder / "metadata.txt",
        metadata_folder / "summary.txt"
    ]
    
    metadata_path = None
    for file in possible_metadata_files:
        if file.exists():
            metadata_path = file
            break
    
    if metadata_path is None:
        print(f"Warning: No metadata file found in {metadata_folder}")
        return None

    try:
        with open(metadata_path, 'r') as f:
            content = f.read()

        # Find the section for our image (handle different extensions)
        base_filename = Path(image_filename).stem  # Remove extension if present
        pattern = re.compile(r"File: (.*" + re.escape(base_filename) + r")")
        match = pattern.search(content)
        if not match:
            print(f"Warning: No metadata found for image {base_filename}")
            return None

        # Extract the pixel size line (looking in the matched section)
        section_start = match.start()
        section_end = content.find("\n\n", section_start)  # Look for next blank line
        if section_end == -1:
            section_end = len(content)
            
        section_content = content[section_start:section_end]
        
        # Look for pixel size in various possible formats
        pixel_line_match = re.search(r"Pixel Size[:\s]+([0-9.]+)\s*µm", section_content)
        if not pixel_line_match:
            pixel_line_match = re.search(r"Pixel[:\s]+([0-9.]+)\s*µm", section_content)
        if not pixel_line_match:
            pixel_line_match = re.search(r"Size[:\s]+([0-9.]+)\s*µm", section_content)
            
        if not pixel_line_match:
            print(f"Warning: Pixel size not found for image {base_filename}")
            return None

        pixel_size_x = float(pixel_line_match.group(1))
        print(f"Found pixel size: {pixel_size_x} µm for {base_filename}")
        return pixel_size_x

    except Exception as e:
        print(f"Error reading metadata from {metadata_path}: {e}")
        return None


def find_matching_files(sam_folder, brightness_folder):
    """Find matching mask and image file pairs"""
    mask_files = list(sam_folder.glob("*_masks.npy"))
    file_pairs = []

    for mask_file in mask_files:
        original_stem = mask_file.stem.replace('_masks', '')
        # Try different possible image extensions
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            tif_file = brightness_folder / f"{original_stem}{ext}"
            if tif_file.exists():
                file_pairs.append((mask_file, tif_file))
                break
        else:
            print(f"No matching image file found for {mask_file}")

    return file_pairs


def load_data(mask_path, image_path):
    """Load mask and image data with proper 16-bit handling"""
    masks = np.load(mask_path)
    image = tifffile.imread(image_path)

    if image.ndim == 3:
        if image.shape[0] <= 4:  # If channels first
            image = np.transpose(image, (1, 2, 0))

    return masks, image


def display_image_with_mask_borders(ax, image, masks):
    """Display image with only mask borders"""
    if image.dtype == np.uint16:
        display_img = image.astype(np.float32) / 65535.0
    elif image.dtype in [np.float32, np.float64]:
        display_img = np.clip(image, 0, 1)
    else:
        display_img = image

    # Display the image
    ax.imshow(display_img)
    
    # Find and plot contours for each mask
    for i in range(1, masks.max() + 1):
        mask = (masks == i).astype(np.uint8)
        contours = find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='yellow')
    
    return display_img


def calculate_axis_endpoints(centroid, length, angle, is_major=True):
    """Calculate endpoints for axis lines"""
    y0, x0 = centroid
    if is_major:
        x1 = x0 + np.cos(angle) * 0.5 * length
        y1 = y0 - np.sin(angle) * 0.5 * length
        x2 = x0 - np.cos(angle) * 0.5 * length
        y2 = y0 + np.sin(angle) * 0.5 * length
    else:
        x1 = x0 - np.sin(angle) * 0.5 * length
        y1 = y0 - np.cos(angle) * 0.5 * length
        x2 = x0 + np.sin(angle) * 0.5 * length
        y2 = y0 + np.cos(angle) * 0.5 * length

    return (x1, y1), (x2, y2)


def plot_cell_info(ax, region, pixel_size_um, time_interval):
    """Plot cell information including axes and growth rate"""
    centroid = region.centroid
    minor_len = region.minor_axis_length
    major_len = region.major_axis_length
    orientation = region.orientation

    # Calculate endpoints
    major_p1, major_p2 = calculate_axis_endpoints(centroid, major_len, orientation, True)
    minor_p1, minor_p2 = calculate_axis_endpoints(centroid, minor_len, orientation, False)

    # Plot axes and centroid and return the line objects
    major_line = ax.plot([major_p1[0], major_p2[0]], [major_p1[1], major_p2[1]],
                        color='red', linewidth=0.8, label='Major Axis')[0]
    minor_line = ax.plot([minor_p1[0], minor_p2[0]], [minor_p1[1], minor_p2[1]],
                        color='blue', linewidth=0.8, label='Minor Axis')[0]
    centroid_point = ax.plot(centroid[1], centroid[0], 'yo', markersize=2)[0]

    # Calculate growth metrics
    half_major_px = major_len / 2
    half_major_um = half_major_px * pixel_size_um
    growth_rate = half_major_um / time_interval  # µm/min

    info_text = (
        f"Major Axis: {major_len * pixel_size_um:.1f}µm\n"
        f"Growth Rate: {growth_rate:.3f}µm/min"
    )

    return info_text, growth_rate, major_line, minor_line, centroid_point


def create_intensity_plots(image, mask, cell_num, axs):
    """Create intensity distribution plots for a cell"""
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Image must be RGB (3 channels)")

    cell_mask = mask == cell_num
    channels = ['Red', 'Green', 'Blue']

    for i, (channel, ax) in enumerate(zip(channels, axs)):
        channel_data = image[:, :, i]
        masked_data = channel_data[cell_mask]
        
        # Normalize based on dtype
        if channel_data.dtype == np.uint16:
            masked_data = masked_data / 65535.0
        elif channel_data.dtype == np.uint8:
            masked_data = masked_data / 255.0
        
        ax.cla()
        ax.hist(masked_data.flatten(), bins=50, color=channel.lower(), alpha=0.7)
        ax.set_title(f'{channel} Channel Intensity')
        ax.set_ylabel('Pixel Count')
        ax.grid(True)
        ax.set_xlim(0, 1)  # Consistent scale for comparison


def analyze_image_pair_interactive(mask_path, image_path, pixel_size_um, time_interval=40):
    """Interactive analysis function with mouse hover functionality"""
    masks, image = load_data(mask_path, image_path)
    regions = regionprops(masks)

    # Create main figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, width_ratios=[2, 0.05, 1])  # 3 rows, 3 columns
    
    # Main image takes left 2 columns and all rows
    ax_img = fig.add_subplot(gs[:, 0])

    # Create a column for the info box between image and plots
    info_ax = fig.add_subplot(gs[:, 1])
    info_ax.axis('off')  # Hide the axes
    
    # Create separate axes for each intensity plot
    ax_red = fig.add_subplot(gs[0, 2])    # Top right
    ax_green = fig.add_subplot(gs[1, 2])  # Middle right
    ax_blue = fig.add_subplot(gs[2, 2])   # Bottom right
    
    intensity_axs = [ax_red, ax_green, ax_blue]

    # Display the image with mask borders
    display_img = display_image_with_mask_borders(ax_img, image, masks)

    # Create a dictionary to map coordinates to cell numbers
    coord_to_cell = {}
    for i, region in enumerate(regions, 1):
        for coord in region.coords:
            coord_to_cell[(coord[0], coord[1])] = i

    # Create text box for cell info (position adjusted)
    info_box = fig.text(0.72, 0.95, "", 
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
                       fontsize=9,
                       linespacing=1.5)

    # Create a cursor
    cursor = Cursor(ax_img, useblit=True, color='red', linewidth=1)

    # Initialize variables to store plot elements
    last_contour = None
    major_axis_line = None
    minor_axis_line = None
    centroid_point = None

    def on_mouse_move(event):
        nonlocal last_contour, major_axis_line, minor_axis_line, centroid_point

        if event.inaxes != ax_img:
            return

        try:
            x, y = int(event.xdata), int(event.ydata)
        except (TypeError, ValueError):
            # Handle cases where coordinates are None or invalid
            return

        # Remove previous elements if they exist
        for element in [last_contour, major_axis_line, minor_axis_line, centroid_point]:
            if element is not None and element in ax_img.lines or element in ax_img.collections:
                try:
                    element.remove()
                except ValueError:
                    pass  # Element was already removed
            element = None

        # Find which cell the cursor is on
        cell_num = coord_to_cell.get((y, x), None)

        if cell_num is not None and 1 <= cell_num <= len(regions):
            try:
                region = regions[cell_num - 1]

                # Highlight the current cell with a thicker border
                mask = (masks == cell_num).astype(np.uint8)
                contours = find_contours(mask, 0.5)
                for contour in contours:
                    last_contour = ax_img.plot(contour[:, 1], contour[:, 0], 
                                             linewidth=2, color='cyan')[0]

                # Update cell info
                info_text, _, major_axis_line, minor_axis_line, centroid_point = plot_cell_info(
                    ax_img, region, pixel_size_um, time_interval
                )
                info_box.set_text(info_text)

                # Update intensity plots
                try:
                    create_intensity_plots(image, masks, cell_num, intensity_axs)
                except ValueError as e:
                    print(f"Error creating intensity plots: {e}")
                    for ax in intensity_axs:
                        ax.cla()
                        ax.set_title('')
                        ax.set_ylabel('')
                        ax.grid(False)

            except IndexError:
                # Handle case where cell_num is out of bounds
                pass
        else:
            # Clear the info box when not hovering over a cell
            info_box.set_text("")
            
            # Clear intensity plots
            for ax in intensity_axs:
                ax.cla()
                ax.set_title('')
                ax.set_ylabel('')
                ax.grid(False)

        try:
            fig.canvas.draw_idle()
        except:
            pass

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    plt.suptitle(f"Interactive Analysis - {image_path.stem}\n"
                f"Pixel size: {pixel_size_um} µm | Time interval: {time_interval} min")
    plt.tight_layout()
    plt.show()


def main():
    """Main program entry point"""
    parser = argparse.ArgumentParser(description='Interactive cell growth analysis')
    parser.add_argument('--pipeline', action='store_true',
                        help='Run in pipeline mode (processes only first image)')
    args = parser.parse_args()

    sam_folder = Path(r"C:\Users\zindi\PycharmProjects\P2\Evaluations\SAM")
    brightness_folder = Path(r"C:\Users\zindi\PycharmProjects\P2\train_brightness")
    metadata_folder = Path(r"C:\Users\zindi\PycharmProjects\P2\test_data")

    file_pairs = find_matching_files(sam_folder, brightness_folder)

    if not file_pairs:
        print("No matching file pairs found!")
        return

    if args.pipeline:
        # Process only the first image in pipeline mode
        mask_path, image_path = file_pairs[0]
        image_filename = image_path.name
        pixel_size_um = read_pixel_size_from_metadata(metadata_folder, image_filename) or 0.0322
        analyze_image_pair_interactive(mask_path, image_path, pixel_size_um)
    else:
        print("\nFound the following file pairs:")
        for i, (mask_path, image_path) in enumerate(file_pairs, 1):
            print(f"{i}: {mask_path.name} with {image_path.name}")

        while True:
            try:
                selection = input(f"\nEnter which file to process (1-{len(file_pairs)}): ")
                selected_idx = int(selection) - 1
                if 0 <= selected_idx < len(file_pairs):
                    mask_path, image_path = file_pairs[selected_idx]
                    image_filename = image_path.name
                    pixel_size_um = read_pixel_size_from_metadata(metadata_folder, image_filename) or 0.0322
                    analyze_image_pair_interactive(mask_path, image_path, pixel_size_um)
                    break
                print(f"Please enter a number between 1 and {len(file_pairs)}")
            except ValueError:
                print("Please enter a valid number")


if __name__ == "__main__":
    main()