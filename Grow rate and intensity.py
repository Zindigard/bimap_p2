import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from skimage.measure import regionprops, find_contours
import tifffile
import argparse
import re
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap


def read_pixel_size_from_metadata(metadata_folder, image_filename):
    """Extract the X pixel size (µm) from metadata."""
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

        base_filename = Path(image_filename).stem
        pattern = re.compile(r"File: (.*" + re.escape(base_filename) + r")", re.IGNORECASE)
        match = pattern.search(content)
        if not match:
            print(f"Warning: No metadata found for {base_filename}")
            return None

        pixel_match = re.search(r"X:\s*([0-9.]+)\s*µm", content)
        if not pixel_match:
            print(f"Warning: X pixel size not found for {base_filename}")
            return None

        return float(pixel_match.group(1))

    except Exception as e:
        print(f"Error reading metadata: {e}")
        return None


def find_matching_files(sam_folder, brightness_folder):
    """Find matching mask and image file pairs"""
    mask_files = list(sam_folder.glob("*_masks.npy"))
    file_pairs = []

    for mask_file in mask_files:
        original_stem = mask_file.stem.replace('_masks', '')
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            tif_file = brightness_folder / f"{original_stem}{ext}"
            if tif_file.exists():
                file_pairs.append((mask_file, tif_file))
                break
    return file_pairs


def load_data(mask_path, image_path):
    """Load mask and image data with proper 16-bit handling and channel order"""
    masks = np.load(mask_path)
    image = tifffile.imread(image_path)
    
    # Fix channel order for matplotlib (convert from channels-first to channels-last)
    if image.ndim == 3:
        # Check if channel dimension is first
        if image.shape[0] in [1, 3, 4]:  # Typical channel sizes
            # Move channel dimension to last position
            image = np.transpose(image, (1, 2, 0))
    
    return masks, image


def display_image_with_mask_borders(ax, image, masks):
    """Display image with cell outlines"""
    if image.dtype == np.uint16:
        display_img = image.astype(np.float32) / 65535.0
    elif image.dtype in [np.float32, np.float64]:
        display_img = np.clip(image, 0, 1)
    else:
        display_img = image
    
    ax.imshow(display_img)
    
    # Draw all cell outlines in yellow
    for i in range(1, masks.max() + 1):
        mask = (masks == i).astype(np.uint8)
        contours = find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='yellow')
    
    return display_img


def calculate_growth_rate(major_length_px, pixel_size_um, time_interval=40):
    """Calculate growth rate based on major axis length and time interval"""
    half_major_um = (major_length_px / 2) * pixel_size_um
    return half_major_um / time_interval  # µm/min


def show_cell_details(image, masks, cell_num, pixel_size_um, time_interval=40):
    """Show detailed view with optimized layout and consistent plot sizes"""
    # Determine number of channels (assume 3 channels)
    num_channels = 3
    
    # Create figure with dark theme
    fig = plt.figure(figsize=(18, 15), facecolor='black')  # Increased height for better proportions
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], width_ratios=[1, 1])
    
    # Top-left: Cell image with axes
    ax_cell = fig.add_subplot(gs[0, 0])
    
    # Display original image
    if image.dtype == np.uint16:
        display_img = image.astype(np.float32) / 65535.0
    else:
        display_img = image
    ax_cell.imshow(display_img)
    
    # Get cell mask and calculate properties
    cell_mask = (masks == cell_num).astype(np.uint8)
    region = regionprops(cell_mask)[0]
    
    # Extract key properties
    centroid_y, centroid_x = region.centroid
    major_length = region.major_axis_length
    minor_length = region.minor_axis_length
    
    # Calculate growth rate (added from first code)
    growth_rate = calculate_growth_rate(major_length, pixel_size_um, time_interval)
    
    # ROTATION: Use orientation + 90° (π/2 radians) for axes
    orientation = region.orientation + np.pi/2
    
    # Calculate major axis endpoints
    major_x1 = centroid_x + (major_length/2) * np.cos(orientation)
    major_y1 = centroid_y - (major_length/2) * np.sin(orientation)
    major_x2 = centroid_x - (major_length/2) * np.cos(orientation)
    major_y2 = centroid_y + (major_length/2) * np.sin(orientation)
    
    # Calculate minor axis endpoints
    minor_orientation = orientation + np.pi/2
    minor_x1 = centroid_x + (minor_length/2) * np.cos(minor_orientation)
    minor_y1 = centroid_y - (minor_length/2) * np.sin(minor_orientation)
    minor_x2 = centroid_x - (minor_length/2) * np.cos(minor_orientation)
    minor_y2 = centroid_y + (minor_length/2) * np.sin(minor_orientation)
    
    # Highlight cell border
    contours = find_contours(cell_mask, 0.5)
    for contour in contours:
        ax_cell.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='cyan')
    
    # Plot axes and centroid
    ax_cell.plot([major_x1, major_x2], [major_y1, major_y2], 'w-', linewidth=2)
    ax_cell.plot([minor_x1, minor_x2], [minor_y1, minor_y2], 'w-', linewidth=2)
    ax_cell.plot(centroid_x, centroid_y, 'yo', markersize=8)
    
    # Add legend to top right
    ax_cell.text(0.98, 0.98, "Major Axis\nMinor Axis\nCentroid", 
                transform=ax_cell.transAxes, 
                color='white', fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Add title
    ax_cell.set_title(f"Cell {cell_num}", fontsize=14, color='white')
    
    # Zoom in on the cell with 10% margin
    minr, minc, maxr, maxc = region.bbox
    margin = max(maxr - minr, maxc - minc) * 0.1
    ax_cell.set_xlim(minc - margin, maxc + margin)
    ax_cell.set_ylim(maxr + margin, minr - margin)
    
    # Remove grid and ticks
    ax_cell.grid(False)
    ax_cell.set_xticks([])
    ax_cell.set_yticks([])
    
    # Top-right: Green channel intensity plot
    ax_green = fig.add_subplot(gs[0, 1])
    
    # Middle-left: Measurement information
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')  # Turn off axis
    
    # Create measurement text with growth rate (added from first code)
    info_text = (
        f"Cell Measurements:\n\n"
        f"Length: {major_length * pixel_size_um:.2f} µm\n"
        f"Width: {minor_length * pixel_size_um:.2f} µm\n"
        f"Growth Rate: {growth_rate:.3f} µm/min\n"
        f"Pixel Size: {pixel_size_um:.4f} µm\n"
        f"Time Interval: {time_interval} min"
    )
    
    # Add text to plot centered
    ax_info.text(0.5, 0.5, info_text, fontsize=14, color='white',
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', transform=ax_info.transAxes)
    
    # Middle-right: Red channel intensity plot
    ax_red = fig.add_subplot(gs[1, 1])
    
    # Bottom-right: Blue channel intensity plot
    ax_blue = fig.add_subplot(gs[2, 1])
    
    # Create coordinate system with major axis as x-axis
    A = np.array([major_x1, major_y1])
    B = np.array([major_x2, major_y2])
    L = np.linalg.norm(B - A)
    direction = (B - A) / L if L > 0 else np.array([1, 0])
    perp = np.array([-direction[1], direction[0]])
    
    # Get cell points for spatial intensity calculations
    cell_points = np.where(cell_mask)
    
    # Define channel plots
    channel_axes = [ax_green, ax_red, ax_blue]
    channel_indices = [1, 0, 2]  # Green, Red, Blue
    channel_names = ["Green", "Red", "Blue"]
    
    # Channel-specific colormaps from black to pure color
    colormaps = [
        LinearSegmentedColormap.from_list('green', ['#000000', '#00ff00']),
        LinearSegmentedColormap.from_list('red', ['#000000', '#ff0000']),
        LinearSegmentedColormap.from_list('blue', ['#000000', '#0000ff'])
    ]
    
    # Process each channel
    for ax, channel_idx, channel_name, colormap in zip(channel_axes, channel_indices, channel_names, colormaps):
        # Set axis background to black
        ax.set_facecolor('black')
        
        # Get channel data
        if image.ndim == 2:
            channel_data = image
        else:
            channel_data = image[:, :, channel_idx]
        
        # Create grid for spatial intensity mapping
        num_x_bins = 100
        num_y_bins = 50
        spatial_grid = np.zeros((num_y_bins, num_x_bins))
        count_grid = np.zeros((num_y_bins, num_x_bins))
        
        # Transform cell points to new coordinate system
        for y, x in zip(*cell_points):
            P = np.array([x, y])
            vec = P - A
            x_pos = np.dot(vec, direction)  # Position along major axis
            y_pos = np.dot(vec, perp)       # Distance from major axis
            
            # Normalize positions to grid coordinates
            x_idx = int(np.clip(x_pos / L * num_x_bins, 0, num_x_bins-1))
            y_idx = int(np.clip((y_pos + minor_length/2) / minor_length * num_y_bins, 0, num_y_bins-1))
            
            # Add intensity to grid
            intensity = channel_data[y, x]
            spatial_grid[y_idx, x_idx] += intensity
            count_grid[y_idx, x_idx] += 1
        
        # Calculate average intensity
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_intensity = np.divide(spatial_grid, count_grid)
            avg_intensity[count_grid == 0] = 0
        
        # Plot spatial intensity with channel-specific colormap
        im = ax.imshow(avg_intensity, cmap=colormap, aspect='auto', 
                      extent=[0, major_length * pixel_size_um, 
                              -minor_length/2 * pixel_size_um, 
                              minor_length/2 * pixel_size_um],
                      origin='lower')
        
        # Add colorbar with white label
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Add labels and title with white text
        ax.set_xlabel('Position along cell (µm)', fontsize=10, color='white')
        ax.set_ylabel('Distance from axis (µm)', fontsize=10, color='white')
        ax.set_title(f'{channel_name} Channel Intensity', fontsize=12, color='white')
        
        # Add center line
        ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        
        # Add grid for better readability
        ax.grid(True, linestyle=':', alpha=0.3, color='white')
        
        # Set tick colors to white
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Set figure text color to white
    for text in fig.texts:
        text.set_color('white')
    
    plt.show()
    
    return {
        'centroid': (centroid_x, centroid_y),
        'major_axis': ((major_x1, major_y1), (major_x2, major_y2)),
        'minor_axis': ((minor_x1, minor_y1), (minor_x2, minor_y2)),
        'length_um': major_length * pixel_size_um,
        'width_um': minor_length * pixel_size_um,
        'growth_rate': growth_rate
    }


def analyze_image_pair_interactive(mask_path, image_path, pixel_size_um):
    """Interactive analysis function with cell selection"""
    masks, image = load_data(mask_path, image_path)
    regions = regionprops(masks)

    # Create main figure
    fig, ax = plt.subplots(figsize=(10, 10))
    display_img = display_image_with_mask_borders(ax, image, masks)
    
    # Create coordinate to cell number mapping
    coord_to_cell = {}
    for i, region in enumerate(regions, 1):
        for coord in region.coords:
            coord_to_cell[(coord[0], coord[1])] = i  # row (y), column (x)

    # Add cursor
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    
    # Variable to store highlighted contour artist
    highlighted_contour = None
    
    def on_click(event):
        nonlocal highlighted_contour
        
        if event.inaxes != ax:
            return
        
        try:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
        except (TypeError, ValueError):
            return
        
        # Remove previous highlight safely
        if highlighted_contour is not None:
            try:
                highlighted_contour.remove()
            except ValueError:
                pass  # Already removed by other means
            finally:
                highlighted_contour = None
            fig.canvas.draw_idle()
        
        # Check bounds and get cell number
        height, width = image.shape[:2]
        if 0 <= x < width and 0 <= y < height:
            cell_num = coord_to_cell.get((y, x), None)
        else:
            cell_num = None
        
        if cell_num:
            print(f"Selected Cell: {cell_num}")
            # Highlight selected cell in cyan
            mask = (masks == cell_num).astype(np.uint8)
            contours = find_contours(mask, 0.5)
            if contours:
                # Find the longest contour
                contour = max(contours, key=len)
                highlighted_contour = ax.plot(
                    contour[:, 1], contour[:, 0],
                    linewidth=2, color='cyan'
                )[0]
                fig.canvas.draw_idle()
            
            # Show cell details
            show_cell_details(image, masks, cell_num, pixel_size_um)
        else:
            # Clicked on empty space - just redraw to remove any highlights
            fig.canvas.draw_idle()

    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.title(f"Click on a cell to analyze\n{image_path.stem} | Pixel size: {pixel_size_um} µm")
    plt.tight_layout()
    plt.show()


def main():
    """Main program entry point"""
    parser = argparse.ArgumentParser(description='Interactive cell analysis')
    parser.add_argument('--pipeline', action='store_true',
                        help='Run in pipeline mode (processes only first image)')
    args = parser.parse_args()

    sam_folder = Path(r"C:\Users\zindi\PycharmProjects\P2\Evaluations\SAM")
    brightness_folder = Path(r"C:\Users\zindi\PycharmProjects\P2\test_brightness")
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