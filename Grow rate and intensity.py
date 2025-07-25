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
    
    if image.ndim == 3:
        if image.shape[0] in [1, 3, 4]:  
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
    
    # outlines  
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
        ax.set_ylabel('Distance from axis (µm)', fontsize=10, color='white')
        ax.set_title(f'{channel_name} Channel Intensity', fontsize=12, color='white')
        
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
    """Save growth rates of all cells to a text file"""
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
    """Interactive analysis function with cell selection"""
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

    plot_all_channels_intensity_profiles(
    masks, 
    image,
    pixel_size_um
)
  

def plot_all_channels_intensity_profiles(masks, image, pixel_size_um):
    """Plot intensity profiles for all channels colormaps"""
   
    regions = regionprops(masks)
    if len(regions) == 0:
        print("No cells found")
        return
    
    valid_regions = [r for r in regions if r.major_axis_length > 0]
    if len(valid_regions) == 0:
        print("No valid cells found")
        return
    
    sorted_regions = sorted(valid_regions, key=lambda r: r.major_axis_length)
    num_valid = len(sorted_regions)
    
    purple_yellow = LinearSegmentedColormap.from_list(
        'compact_purple_yellow', 
        [
            '#2E0854', 
            '#8A2BE2',  
            '#FFD700'   
        ],
        N=128  
    )
    
    channels = [
        {"idx": 1, "name": "Green"},
        {"idx": 0, "name": "Red"},
        {"idx": 2, "name": "Blue"}]
    
    fig = plt.figure(figsize=(16, 18), facecolor='black')
    main_gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)
    
    axes = []
    cbar_axes = []
    
    for i, channel in enumerate(channels):
        channel_gs = GridSpecFromSubplotSpec(1, 2, 
                                    subplot_spec=main_gs[i],
                                    width_ratios=[0.97, 0.03],
                                    wspace=0.02) 
        
        ax = fig.add_subplot(channel_gs[0])  
        ax.set_facecolor('black')
        axes.append(ax)
        
        cax = fig.add_subplot(channel_gs[1]) 
        cbar_axes.append(cax)
    
    max_length_um = max(r.major_axis_length * pixel_size_um for r in sorted_regions)
    
    for ax, cax, channel in zip(axes, cbar_axes, channels):
        if image.ndim == 2:
            channel_data = image
        else:
            channel_data = image[:, :, channel["idx"]]
        
        all_segments = []
        all_colors = []
        
        channel_min_intensity = float('inf')
        channel_max_intensity = 0
        
        #  intensity 
        for region in sorted_regions:
            label = region.label
            cell_mask = (masks == label)
            y_points, x_points = np.where(cell_mask)
            
            for y, x in zip(y_points, x_points):
                intensity = channel_data[y, x]
                if intensity > channel_max_intensity:
                    channel_max_intensity = intensity
                if intensity < channel_min_intensity:
                    channel_min_intensity = intensity
        
        if channel_min_intensity == float('inf'):
            channel_min_intensity = 0
        if channel_max_intensity == 0:
            channel_max_intensity = 1
            
        if np.any(channel_data):
            vmin = np.percentile(channel_data, 2)
            vmax = np.percentile(channel_data, 98)
        else:
            vmin, vmax = 0, 1
        norm = plt.Normalize(vmin, vmax)
            
        for i, region in enumerate(sorted_regions):
            label = region.label
            L = region.major_axis_length
            length_um = L * pixel_size_um
            
            cell_mask = (masks == label)
            y_points, x_points = np.where(cell_mask)
            
            # orientation
            cy, cx = region.centroid
            orientation = region.orientation
            dx = np.cos(orientation) * 0.5 * L
            dy = np.sin(orientation) * 0.5 * L
            
            # Direction 
            direction = np.array([-dx, dy])
            direction_norm = direction / np.linalg.norm(direction)
            
            num_bins = 50
            bin_means = np.zeros(num_bins)
            bin_counts = np.zeros(num_bins)
            positions = np.linspace(-length_um/2, length_um/2, num_bins)
            
            for y, x in zip(y_points, x_points):
                vec = np.array([x - cx, y - cy])
                pos = np.dot(vec, direction_norm) * pixel_size_um
                
                bin_idx = int(np.clip((pos + length_um/2) / length_um * num_bins, 0, num_bins-1))
                intensity = channel_data[y, x]
                
                bin_means[bin_idx] += intensity
                bin_counts[bin_idx] += 1
            
            # mean 
            valid = bin_counts > 0
            bin_means[valid] /= bin_counts[valid]
            
            
            for j in range(num_bins - 1):
                if valid[j] and valid[j+1]:
                    y_start = positions[j]
                    y_end = positions[j+1]
                    segment = [(i+1, y_start), (i+1, y_end)]
                    all_segments.append(segment)
                    
                    avg_intensity = (bin_means[j] + bin_means[j+1]) / 2
                    all_colors.append(avg_intensity)
        
        lc = LineCollection(
            all_segments,
            array=np.array(all_colors),
            cmap=purple_yellow,
            norm=norm,
            linewidth=1.5, 
            alpha=0.9
        )
        ax.add_collection(lc)
        
        ax.set_xlim(0, num_valid + 1)
        ax.set_ylim(-max_length_um * 0.55, max_length_um * 0.55)
        ax.set_ylabel('Position (µm)', fontsize=10, color='white')  
        ax.set_title(f'{channel["name"]} Channel', fontsize=12, color='white')  
        
        ax.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=0.8)
        
        ax.grid(True, linestyle=':', alpha=0.2, color='white')
        ax.tick_params(colors='white', labelsize=8)  
        
        if ax != axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Cell Index', fontsize=10, color='white')
    
        cbar = plt.colorbar(lc, cax=cax)
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.yaxis.set_tick_params(color='white', size=3) 
        plt.setp(cbar.ax.get_yticklabels(), color='white', fontsize=6)
        cax.set_facecolor('black')
    
    plt.subplots_adjust(hspace=0.15)  
    plt.show()


def main():
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