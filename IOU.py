import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def get_project_root():
    """
    Get the root directory where scripts are located.
    
    Returns:
        str: Absolute path to the directory containing this script
    """
    return os.path.dirname(os.path.abspath(__file__))

def calculate_metrics(pred, gt):
    """
    Calculate segmentation performance metrics between predicted and ground truth binary masks.
    
    Args:
        pred (np.ndarray): Predicted binary mask (0=background, 1=foreground)
        gt (np.ndarray): Ground truth binary mask (0=background, 1=foreground)
        
    Returns:
        dict: Dictionary containing:
            - iou: Intersection over Union (Jaccard index) for entire image
            - iou_foreground: IoU calculated only within ground truth regions
            - precision: TP / (TP + FP) - how many predictions are correct
            - accuracy: (TP + TN) / (TP + TN + FP + FN) - overall correctness
            - tp: True Positive count
            - fp: False Positive count  
            - fn: False Negative count
            - tn: True Negative count
    """
    # Calculate confusion matrix components
    tp = np.sum(np.logical_and(pred == 1, gt == 1))
    fp = np.sum(np.logical_and(pred == 1, gt == 0))
    fn = np.sum(np.logical_and(pred == 0, gt == 1))
    tn = np.sum(np.logical_and(pred == 0, gt == 0))

    # Standard IoU (Jaccard index)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # Precision 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Accuracy 
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Foreground IoU 
    gt_mask = gt == 1
    pred_gt_region = pred[gt_mask]
    gt_gt_region = gt[gt_mask]
    
    tp_gt = np.sum(pred_gt_region == 1)
    fn_gt = np.sum(pred_gt_region == 0)  
    
    iou_foreground = tp_gt / (tp_gt + fn_gt) if (tp_gt + fn_gt) > 0 else 0

    return {
        'iou': iou,
        'iou_foreground': iou_foreground,  
        'precision': precision,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def create_visualization(pred, gt, filename, metrics):
    """
    Generate a visual comparison of predicted and ground truth masks with color-coded overlays.
    
    Args:
        pred (np.ndarray): Predicted binary mask
        gt (np.ndarray): Ground truth binary mask  
        filename (str): Name of the image file being evaluated
        metrics (dict): Calculated quality metrics
        
    Returns:
        str: Path to the saved visualization image
        
    Visualization Colors:
        - Green: Ground truth only (false negatives)
        - Red: Prediction only (false positives) 
        - Purple: Overlap (true positives)
        - Black: Background (true negatives)
    """
    overlay = np.zeros((*pred.shape, 3))
    overlay[gt == 1] = [0, 1, 0]  # Green for ground truth
    overlay[pred == 1] = [1, 0, 0]  # Red for prediction
    overlay[np.logical_and(pred, gt)] = [0.5, 0, 0.5]  # Purple for overlap

    plt.figure(figsize=(12, 6))
    plt.imshow(overlay)
    plt.title(f"Segmentation Evaluation: {os.path.splitext(filename)[0]}", fontsize=14, fontweight='bold')
    plt.axis('off')

    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=4, label='Ground Truth (FN)'),
        plt.Line2D([0], [0], color='red', lw=4, label='Prediction (FP)'),
        plt.Line2D([0], [0], color='purple', lw=4, label='Overlap (TP)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    metrics_text = (
        f"Evaluation Metrics:\n"
        f"• Standard IoU: {metrics['iou']:.3f} (entire image)\n"
        f"• Foreground IoU: {metrics['iou_foreground']:.3f} (GT regions only)\n"  
        f"• Precision: {metrics['precision']:.3f}\n"
        f"• Accuracy: {metrics['accuracy']:.3f}\n"
        f"\nConfusion Matrix:\n"
        f"TP: {metrics['tp']} | FP: {metrics['fp']}\n"
        f"FN: {metrics['fn']} | TN: {metrics['tn']}"
    )

    plt.gcf().text(0.82, 0.65, metrics_text,  
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                 fontsize=9, fontfamily='monospace')

    output_path = get_project_root()
    results_dir = os.path.join(output_path, "Evaluations", "Results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, f"{os.path.splitext(filename)[0]}_eval.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    
    return output_file

def evaluate_and_visualize_masks(pred_path=None, gt_path=None, output_path=None, interactive=False):
    """
    Main evaluation function that compares predicted vs ground truth segmentation masks.
    Processes all mask files in the directories and generates comprehensive evaluation.
    
    Args:
        pred_path (str): Directory containing predicted masks from segmentation
                        (None for default: Evaluations/SAM)
        gt_path (str): Directory containing ground truth masks 
                      (None for default: Evaluations/Ground)
        output_path (str): Directory for evaluation results and visualizations
                         (None for default: Evaluations/Results)
        interactive (bool): Whether to ask for confirmation for each image
                          (True: manual confirmation, False: process all automatically)
                          
    Returns:
        list: List of evaluation results dictionaries for all processed images
        
    Output Files:
        - PNG visualization for each image with color-coded overlays
        - CSV file with all metrics across all images
        - Console summary of overall performance
    """
    root_dir = get_project_root()
    
    if pred_path is None:
        pred_path = os.path.join(root_dir, "Evaluations", "SAM")
    if gt_path is None:
        gt_path = os.path.join(root_dir, "Evaluations", "Ground")
    if output_path is None:
        output_path = os.path.join(root_dir, "Evaluations", "Results")
    
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 60)
    print("SEGMENTATION EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Predicted masks directory: {pred_path}")
    print(f"Ground truth directory: {gt_path}")
    print(f"Output directory: {output_path}")
    print(f"Interactive mode: {interactive}")
    print("-" * 60)

    pred_files = [f for f in os.listdir(pred_path) if f.endswith('_binary.tif')]
    
    if not pred_files:
        print("ERROR: No predicted mask files found!")
        print(f"Expected files ending with '_binary.tif' in: {pred_path}")
        return []

    print(f"Found {len(pred_files)} predicted mask files")
    
    results = []
    processed_count = 0
    skipped_count = 0

    for pred_file in pred_files:
        # Interactive mode: ask user for confirmation
        if interactive:
            user_input = input(f"\nProcess '{pred_file}'? (y/n/skip all): ").strip().lower()
            if user_input in ('n', 'no'):
                print(f"[SKIPPED] {pred_file}")
                skipped_count += 1
                continue
            elif user_input in ('s', 'skip all'):
                print("Skipping all remaining files...")
                break

        try:
            # Expected naming: predicted: 'image_binary.tif' -> ground truth: 'image_mask.tif'
            gt_file = pred_file.replace('_binary', '_mask')
            gt_path_full = os.path.join(gt_path, gt_file)

            if not os.path.exists(gt_path_full):
                print(f"[SKIPPED] No ground truth found for {pred_file}")
                print(f"         Expected: {gt_file}")
                skipped_count += 1
                continue

            # Load and binarize masks
            # USER NOTE: Assumes masks are binary (0/1 or 0/255)
            # Adjust threshold if your masks use different value ranges
            pred_mask = io.imread(os.path.join(pred_path, pred_file))
            gt_mask = io.imread(gt_path_full)
            
            # Convert to binary (handle both 0/1 and 0/255 formats)
            pred_binary = (pred_mask > 0).astype(np.uint8)
            gt_binary = (gt_mask > 0).astype(np.uint8)

            metrics = calculate_metrics(pred_binary, gt_binary)
            
            viz_path = create_visualization(pred_binary, gt_binary, pred_file, metrics)

            results.append({
                'image': pred_file,
                **metrics,
                'visualization': viz_path
            })

            processed_count += 1
            print(f"[PROCESSED] {pred_file}")
            print(f"  Standard IoU: {metrics['iou']:.3f} | "
                  f"Foreground IoU: {metrics['iou_foreground']:.3f} | "
                  f"Precision: {metrics['precision']:.3f}")

        except Exception as e:
            print(f"[ERROR] Processing {pred_file}: {str(e)}")
            skipped_count += 1

    # Save comprehensive results to CSV
    if results:
        csv_path = os.path.join(output_path, 'segmentation_metrics.csv')
        with open(csv_path, 'w') as f:
            f.write("Image,StandardIoU,ForegroundIoU,Precision,Accuracy,TP,FP,FN,TN,Visualization\n")  
            
            for r in results:
                f.write(f"{r['image']},{r['iou']:.4f},{r['iou_foreground']:.4f},"
                       f"{r['precision']:.4f},{r['accuracy']:.4f},"
                       f"{r['tp']},{r['fp']},{r['fn']},{r['tn']},{r['visualization']}\n")
        
        print(f"\nDetailed metrics saved to: {csv_path}")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")
    print(f"Total: {len(pred_files)} predicted mask files")
    
    if results:
        # Calculate overall statistics
        iou_scores = [r['iou'] for r in results]
        fg_iou_scores = [r['iou_foreground'] for r in results]
        precision_scores = [r['precision'] for r in results]
        accuracy_scores = [r['accuracy'] for r in results]
        
        print(f"\nOVERALL PERFORMANCE (mean ± standard deviation):")
        print(f"Standard IoU:    {np.mean(iou_scores):.3f} ± {np.std(iou_scores):.3f}")
        print(f"Foreground IoU:  {np.mean(fg_iou_scores):.3f} ± {np.std(fg_iou_scores):.3f}")
        print(f"Precision:       {np.mean(precision_scores):.3f} ± {np.std(precision_scores):.3f}")
        print(f"Accuracy:        {np.mean(accuracy_scores):.3f} ± {np.std(accuracy_scores):.3f}")
        
        print(f"\nPERFORMANCE RANGES:")
        print(f"Standard IoU:    [{np.min(iou_scores):.3f}, {np.max(iou_scores):.3f}]")
        print(f"Foreground IoU:  [{np.min(fg_iou_scores):.3f}, {np.max(fg_iou_scores):.3f}]")
        print(f"Precision:       [{np.min(precision_scores):.3f}, {np.max(precision_scores):.3f}]")
        
        mean_iou = np.mean(iou_scores)
        if mean_iou >= 0.9:
            iou_quality = "EXCELLENT"
        elif mean_iou >= 0.7:
            iou_quality = "GOOD"
        elif mean_iou >= 0.5:
            iou_quality = "FAIR"
        else:
            iou_quality = "POOR"
            
        print(f"\nQUALITY ASSESSMENT: {iou_quality}")
        print(f"All results and visualizations saved to: {output_path}")
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(iou_scores, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(iou_scores), color='red', linestyle='--', label=f'Mean: {np.mean(iou_scores):.3f}')
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        plt.title('Standard IoU Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.hist(fg_iou_scores, bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(fg_iou_scores), color='red', linestyle='--', label=f'Mean: {np.mean(fg_iou_scores):.3f}')
        plt.xlabel('Foreground IoU Score')
        plt.ylabel('Frequency')
        plt.title('Foreground IoU Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.scatter(iou_scores, precision_scores, alpha=0.6, color='purple')
        plt.xlabel('Standard IoU')
        plt.ylabel('Precision')
        plt.title('IoU vs Precision')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        summary_plot_path = os.path.join(output_path, 'performance_summary.png')
        plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Performance summary plot saved to: {summary_plot_path}")
    
    else:
        print("\nNo images were successfully processed!")
        print("Check that:")
        print("1. Predicted mask files exist in Evaluations/SAM/")
        print("2. Ground truth files exist in Evaluations/Ground/")
        print("3. File naming follows pattern: 'image_binary.tif' and 'image_mask.tif'")
    
    return results


if __name__ == "__main__":
    results = evaluate_and_visualize_masks(interactive=False)
    
    # Optional: Print detailed results for debugging
    if results and len(results) > 0:
        print(f"\nFirst result sample:")
        first_result = results[0]
        print(f"Image: {first_result['image']}")
        print(f"Standard IoU: {first_result['iou']:.3f}")
        print(f"Foreground IoU: {first_result['iou_foreground']:.3f}")
        print(f"Precision: {first_result['precision']:.3f}")
        print(f"Visualization: {first_result['visualization']}")