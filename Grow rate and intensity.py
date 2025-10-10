import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from skimage.measure import regionprops, find_contours
from skimage.transform import rotate
import tifffile
import argparse
import re
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def get_project_root():
    """
    Get the root directory where scripts are located.
    
    Returns:
        str: Absolute path to the directory containing this script
    """
    return os.path.dirname(os.path.abspath(__file__))

def read_pixel_size_from_metadata(metadata_folder, image_filename):
    """
    Extract the X pixel size (µm) from metadata file.
    
    Args:
        metadata_folder (Path): Directory containing metadata files
        image_filename (str): Name of the image file
        
    Returns:
        float: Pixel size in micrometers or None if not found
    """
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
    """
    Find matching mask and image file pairs for analysis.
    
    Args:
        sam_folder (Path): Directory containing segmentation masks
        brightness_folder (Path): Directory containing brightness-enhanced images
        
    Returns:
        list: List of (mask_path, image_path) tuples
    """
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
    """
    Load mask and image data with proper 16-bit handling and channel order.
    
    Args:
        mask_path (Path): Path to segmentation mask file
        image_path (Path): Path to corresponding image file
        
    Returns:
        tuple: (masks, image) arrays
    """
    masks = np.load(mask_path)
    image = tifffile.imread(image_path)
    
    if image.ndim == 3:
        if image.shape[0] in [1, 3, 4]:  
            image = np.transpose(image, (1, 2, 0))
    
    return masks, image


def display_image_with_mask_borders(ax, image, masks):
    """
    Display image with cell outlines for interactive selection.
    
    Args:
        ax: Matplotlib axes object
        image (np.ndarray): Input image
        masks (np.ndarray): Segmentation masks
        
    Returns:
        np.ndarray: Display-ready image
    """
    if image.dtype == np.uint16:
        display_img = image.astype(np.float32) / 65535.0
    elif image.dtype in [np.float32, np.float64]:
        display_img = np.clip(image, 0, 1)
    else:
        display_img = image
    
    ax.imshow(display_img)
    
    # outlines  
    for i in range(1, masks.max() + 1):
        mask = (masks == i).astype(np.uint8)
        contours = find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='yellow')
    
    return display_img


def calculate_growth_rate(major_length_px, pixel_size_um, time_interval=40):
    """
    Calculate growth rate based on major axis length and time interval.
    
    Args:
        major_length_px (float): Major axis length in pixels
        pixel_size_um (float): Pixel size in micrometers
        time_interval (float): Time between frames in minutes
        
    Returns:
        float: Growth rate in µm/min
    """
    half_major_um = (major_length_px / 2) * pixel_size_um
    return half_major_um / time_interval  # µm/min


def show_cell_details(image, masks, cell_num, pixel_size_um, time_interval=40):
    """
    Display detailed analysis for a single cell including measurements and intensity profiles.
    
    Args:
        image (np.ndarray): Multi-channel image
        masks (np.ndarray): Segmentation masks
        cell_num (int): Cell ID to analyze
        pixel_size_um (float): Pixel size in micrometers
        time_interval (float): Time interval for growth rate calculation
        
    Returns:
        dict: Cell measurements and analysis results
    """
    num_channels = 3
    
    fig = plt.figure(figsize=(18, 15), facecolor='black')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], width_ratios=[1, 1])
    
    ax_cell = fig.add_subplot(gs[0, 0])
    
    if image.dtype == np.uint16:
        display_img = image.astype(np.float32) / 65535.0
    else:
        display_img = image
    ax_cell.imshow(display_img)
    
    cell_mask = (masks == cell_num).astype(np.uint8)
    region = regionprops(cell_mask)[0]
    
    centroid_y, centroid_x = region.centroid
    major_length = region.major_axis_length
    minor_length = region.minor_axis_length
    
    growth_rate = calculate_growth_rate(major_length, pixel_size_um, time_interval)
    
    orientation = region.orientation + np.pi/2
    
    major_x1 = centroid_x + (major_length/2) * np.cos(orientation)
    major_y1 = centroid_y - (major_length/2) * np.sin(orientation)
    major_x2 = centroid_x - (major_length/2) * np.cos(orientation)
    major_y2 = centroid_y + (major_length/2) * np.sin(orientation)
    
    minor_orientation = orientation + np.pi/2
    minor_x1 = centroid_x + (minor_length/2) * np.cos(minor_orientation)
    minor_y1 = centroid_y - (minor_length/2) * np.sin(minor_orientation)
    minor_x2 = centroid_x - (minor_length/2) * np.cos(minor_orientation)
    minor_y2 = centroid_y + (minor_length/2) * np.sin(minor_orientation)
    
    contours = find_contours(cell_mask, 0.5)
    for contour in contours:
        ax_cell.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='cyan')
    
    ax_cell.plot([major_x1, major_x2], [major_y1, major_y2], 'w-', linewidth=2)
    ax_cell.plot([minor_x1, minor_x2], [minor_y1, minor_y2], 'w-', linewidth=2)
    ax_cell.plot(centroid_x, centroid_y, 'yo', markersize=8)
    
    ax_cell.text(0.98, 0.98, "Major Axis\nMinor Axis\nCentroid", 
                transform=ax_cell.transAxes, 
                color='white', fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax_cell.set_title(f"Cell {cell_num}", fontsize=14, color='white')
    
    minr, minc, maxr, maxc = region.bbox
    margin = max(maxr - minr, maxc - minc) * 0.1
    ax_cell.set_xlim(minc - margin, maxc + margin)
    ax_cell.set_ylim(maxr + margin, minr - margin)
    
    ax_cell.grid(False)
    ax_cell.set_xticks([])
    ax_cell.set_yticks([])
    
    ax_green = fig.add_subplot(gs[0, 1])
    
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')  
    
    info_text = (
        f"Cell Measurements:\n\n"
        f"Length: {major_length * pixel_size_um:.2f} µm\n"
        f"Width: {minor_length * pixel_size_um:.2f} µm\n"
        f"Growth Rate: {growth_rate:.3f} µm/min\n"
        f"Pixel Size: {pixel_size_um:.4f} µm\n"
        f"Time Interval: {time_interval} min"
    )
    
    ax_info.text(0.5, 0.5, info_text, fontsize=14, color='white',
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', transform=ax_info.transAxes)
    
    ax_red = fig.add_subplot(gs[1, 1])
    
    ax_blue = fig.add_subplot(gs[2, 1])
    
    A = np.array([major_x1, major_y1])
    B = np.array([major_x2, major_y2])
    L = np.linalg.norm(B - A)
    direction = (B - A) / L if L > 0 else np.array([1, 0])
    perp = np.array([-direction[1], direction[0]])
    
    cell_points = np.where(cell_mask)
    
    channel_axes = [ax_green, ax_red, ax_blue]
    channel_indices = [1, 0, 2]  # Green, Red, Blue
    channel_names = ["Green", "Red", "Blue"]
    
    colormaps = [
        LinearSegmentedColormap.from_list('green', ['#000000', '#00ff00']),
        LinearSegmentedColormap.from_list('red', ['#000000', '#ff0000']),
        LinearSegmentedColormap.from_list('blue', ['#000000', '#0000ff'])
    ]
    
    for ax, channel_idx, channel_name, colormap in zip(channel_axes, channel_indices, channel_names, colormaps):
        ax.set_facecolor('black')
        
        if image.ndim == 2:
            channel_data = image
        else:
            channel_data = image[:, :, channel_idx]
        
        num_x_bins = 100
        num_y_bins = 50
        spatial_grid = np.zeros((num_y_bins, num_x_bins))
        count_grid = np.zeros((num_y_bins, num_x_bins))
        
        for y, x in zip(*cell_points):
            P = np.array([x, y])
            vec = P - A
            x_pos = np.dot(vec, direction)  
            y_pos = np.dot(vec, perp)      
            
            x_idx = int(np.clip(x_pos / L * num_x_bins, 0, num_x_bins-1))
            y_idx = int(np.clip((y_pos + minor_length/2) / minor_length * num_y_bins, 0, num_y_bins-1))
            
            intensity = channel_data[y, x]
            spatial_grid[y_idx, x_idx] += intensity
            count_grid[y_idx, x_idx] += 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_intensity = np.divide(spatial_grid, count_grid)
            avg_intensity[count_grid == 0] = 0
        
        im = ax.imshow(avg_intensity, cmap=colormap, aspect='auto', 
                      extent=[0, major_length * pixel_size_um, 
                              -minor_length/2 * pixel_size_um, 
                              minor_length/2 * pixel_size_um],
                      origin='lower')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        ax.set_xlabel('Position along cell (µm)', fontsize=10, color='white')
        ax.set_ylabel('Mean Distance from axis (µm)', fontsize=10, color='white')
        ax.set_title(f' Mean {channel_name} Channel Intensity', fontsize=12, color='white')
        
        ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        
        ax.grid(True, linestyle=':', alpha=0.3, color='white')
        
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
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

def save_growth_rates(masks, pixel_size_um, output_path, time_interval=40):
    """
    Save growth rates of all cells to a text file.
    
    Args:
        masks (np.ndarray): Segmentation masks
        pixel_size_um (float): Pixel size in micrometers
        output_path (Path): Path for output text file
        time_interval (float): Time interval for growth rate calculation
    """
    regions = regionprops(masks)
    with open(output_path, 'w') as f:
        f.write("Cell_ID\tGrowth_Rate(µm/min)\tLength(µm)\tWidth(µm)\n")
        for i, region in enumerate(regions, 1):
            if region.major_axis_length > 0:  # Only consider valid cells
                growth_rate = calculate_growth_rate(
                    region.major_axis_length, 
                    pixel_size_um, 
                    time_interval
                )
                length_um = region.major_axis_length * pixel_size_um
                width_um = region.minor_axis_length * pixel_size_um
                f.write(f"{i}\t{growth_rate:.4f}\t{length_um:.2f}\t{width_um:.2f}\n")
    print(f"Saved growth rates to: {output_path}")


def analyze_image_pair_interactive(mask_path, image_path, pixel_size_um):
    """
    Interactive analysis function with cell selection and detailed visualization.
    
    Args:
        mask_path (Path): Path to segmentation masks
        image_path (Path): Path to corresponding image
        pixel_size_um (float): Pixel size in micrometers
    """
    masks, image = load_data(mask_path, image_path)
    regions = regionprops(masks)
    output_path = mask_path.parent / f"{mask_path.stem.replace('_masks', '_growth_rates.txt')}"
    save_growth_rates(masks, pixel_size_um, output_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    display_img = display_image_with_mask_borders(ax, image, masks)
    
    coord_to_cell = {}
    for i, region in enumerate(regions, 1):
        for coord in region.coords:
            coord_to_cell[(coord[0], coord[1])] = i  

    # Add cursor
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    
    highlighted_contour = None
    
    def on_click(event):
        nonlocal highlighted_contour
        
        if event.inaxes != ax:
            return
        
        x = event.xdata
        y = event.ydata
        
        if highlighted_contour is not None:
            try:
                highlighted_contour.remove()
            except ValueError:
                pass
            highlighted_contour = None
            fig.canvas.draw_idle()
        
        height, width = masks.shape[:2]
        
        if 0 <= y < height and 0 <= x < width:
            row = int(y)  
            col = int(x)  
            
            cell_num = masks[row, col]
            
            if cell_num > 0:
                print(f"Selected Cell: {cell_num}")
                
                mask = (masks == cell_num).astype(np.uint8)
                contours = find_contours(mask, 0.5)
                if contours:
                    contour = max(contours, key=len)
                    highlighted_contour = ax.plot(
                        contour[:, 1], contour[:, 0],
                        linewidth=2, color='cyan'
                    )[0]
                    fig.canvas.draw_idle()
                
                show_cell_details(image, masks, cell_num, pixel_size_um)
                return
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.title(f"Cell analyzer\n{image_path.stem} | Pixel size: {pixel_size_um} µm")
    plt.tight_layout()
    plt.show()

    print("Generating multi-channel intensity analysis...")
    all_cell_ids = list(range(1, masks.max() + 1))
    plot_all_cells_all_channels_normalized_spatial(masks, image, pixel_size_um, all_cell_ids, mask_path=mask_path)
  

def plot_all_cells_all_channels_normalized_spatial(masks, image, pixel_size_um, cell_ids, mask_path=None):
    """
    Plot MEAN intensity across ALL cells for ALL channels with Gaussian smoothing.
    
    Args:
        masks (np.ndarray): Segmentation masks
        image (np.ndarray): Multi-channel image
        pixel_size_um (float): Pixel size for scaling
        cell_ids (list): List of cell IDs to include
        mask_path (Path): Path to mask file for naming output
        
    Returns:
        dict: Analysis results for each channel
    """
    fig = plt.figure(figsize=(12, 15), facecolor='black', constrained_layout=True)
    
    
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1])  
    channel_indices = [1, 0, 2]
    channel_names = ['Green', 'Red', 'Blue']
    channel_colors = ['#00FF00', '#FF0000', '#0000FF']  # Green, Red, Blue
    
    channel_results = {}
    total_valid_cells = 0
    
    # Process each channel
    for channel_idx, channel_name, color in zip(channel_indices, channel_names, channel_colors):
        ax = fig.add_subplot(gs[channel_indices.index(channel_idx)])
        ax.set_facecolor('black')
        
        if image.ndim == 2:
            channel_data = image  
        else:
            channel_data = image[:, :, channel_idx]
        
        all_intensity_arrays = []
        
        valid_cells = 0
        
        for i, cell_id in enumerate(cell_ids):
            
            if cell_id not in np.unique(masks):
                continue
                
            valid_cells += 1
            
            cell_mask = (masks == cell_id).astype(np.uint8)
            region = regionprops(cell_mask)[0]
            
            centroid_y, centroid_x = region.centroid
            major_length = region.major_axis_length
            minor_length = region.minor_axis_length
            orientation = region.orientation + np.pi/2
            
            major_x1 = centroid_x + (major_length/2) * np.cos(orientation)
            major_y1 = centroid_y - (major_length/2) * np.sin(orientation)
            major_x2 = centroid_x - (major_length/2) * np.cos(orientation)
            major_y2 = centroid_y + (major_length/2) * np.sin(orientation)
            
            A = np.array([major_x1, major_y1])
            B = np.array([major_x2, major_y2])
            L = np.linalg.norm(B - A)
            direction = (B - A) / L if L > 0 else np.array([1, 0])
            perp = np.array([-direction[1], direction[0]])
            
            
            cell_points = np.where(cell_mask)
            
            num_x_bins = 100
            num_y_bins = 50
            
            spatial_grid_norm = np.zeros((num_y_bins, num_x_bins))
            count_grid_norm = np.zeros((num_y_bins, num_x_bins))
            
            for y, x in zip(*cell_points):
                P = np.array([x, y])
                vec = P - A
                
                x_pos_normalized = np.dot(vec, direction) / L if L > 0 else 0
                y_pos_normalized = np.dot(vec, perp) / minor_length if minor_length > 0 else 0
                
                x_idx_norm = int(np.clip(x_pos_normalized * num_x_bins, 0, num_x_bins-1))
                y_idx_norm = int(np.clip((y_pos_normalized + 0.5) * num_y_bins, 0, num_y_bins-1))
                
                intensity = channel_data[y, x]
                
                spatial_grid_norm[y_idx_norm, x_idx_norm] += intensity
                count_grid_norm[y_idx_norm, x_idx_norm] += 1
            
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_intensity_norm = np.divide(spatial_grid_norm, count_grid_norm)
                avg_intensity_norm[count_grid_norm == 0] = 0
            
            all_intensity_arrays.append(avg_intensity_norm)
        
        if valid_cells == 0:
            print(f"No valid cells found for {channel_name} channel!")
            continue
            
        total_valid_cells = valid_cells 
        
        all_intensities_stack = np.stack(all_intensity_arrays)
        mean_intensity = np.mean(all_intensities_stack, axis=0)
   
        sigma = 0.7
        smoothed_intensity = gaussian_filter(mean_intensity, sigma=sigma)
        
        rgb_color = tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))
        
        if channel_name == 'Blue':
            colors = [
                (0, 0, 0),          
                (0, 0, 0),          
                rgb_color,            
                rgb_color             
            ]
        else:
            colors = [
                (0, 0, 0),           
                (0, 0, 0),           
                (0, 0, 0),          
                rgb_color,            
                rgb_color           
            ]
            
        
        n_bins = 256
        cmap_name = f'{channel_name.lower()}_only'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        
        # Plot the SMOOTHED mean intensity 
        im = ax.imshow(smoothed_intensity, cmap=custom_cmap, aspect='auto',
                      extent=[0, 1, -0.5, 0.5], origin='lower')
        
        ax.set_xlabel('Normalized Position (0 to 1)', fontsize=11, color='white')
        ax.set_ylabel('Normalized Distance (-0.5 to 0.5)', fontsize=11, color='white')
        ax.axhline(0, color='white', linestyle='--', alpha=0.5, linewidth=1.2)
        ax.grid(True, linestyle=':', alpha=0.3, color='white', linewidth=0.8)
        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.12)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Intensity', color='white', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=9)
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=9)
        cax.set_facecolor('black')
        
        channel_results[channel_name] = {
            'total_cells_processed': valid_cells,
            'mean_intensity': mean_intensity,
            'smoothed_intensity': smoothed_intensity,
            'gaussian_sigma': sigma,
        }
    
    fig.suptitle(f'Multi-Channel Intensity Analysis - {total_valid_cells} Cells (Gaussian σ=0.7)\n', 
                fontsize=16, color='white', y=0.98)
    
    if mask_path is not None:
        save_dir = Path(r"C:\Users\zindi\PycharmProjects\P2\Evaluations\SAM")
        mask_filename = Path(mask_path).stem  # Gets 'WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM_masks'
        base_filename = mask_filename.replace('_masks', '')  
        save_filename = save_dir / f"{base_filename}_intensities.png"
        
        plt.savefig(save_filename, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        print(f"Saved intensity plot to: {save_filename}")
    
    plt.show()
    
    return channel_results


def main():
    """
    Main function with argument parsing and folder setup.
    Supports both interactive and pipeline modes.
    """
    parser = argparse.ArgumentParser(description='Interactive cell analysis')
    parser.add_argument('--pipeline', action='store_true',
                        help='Run in pipeline mode (processes only first image)')
    args = parser.parse_args()

    root_dir = get_project_root()
    sam_folder = Path(root_dir) / "Evaluations" / "SAM"
    brightness_folder = Path(root_dir) / "test_brightness"
    metadata_folder = Path(root_dir) / "test_data"
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