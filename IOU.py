import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def evaluate_and_visualize_masks(pred_path, gt_path, output_path, interactive=False):
    os.makedirs(output_path, exist_ok=True)

    def calculate_metrics(pred, gt):
        tp = np.sum(np.logical_and(pred == 1, gt == 1))
        fp = np.sum(np.logical_and(pred == 1, gt == 0))
        fn = np.sum(np.logical_and(pred == 0, gt == 1))
        tn = np.sum(np.logical_and(pred == 0, gt == 0))

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        gt_mask = gt == 1
        # Only consider pixels where ground truth exists
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
        overlay = np.zeros((*pred.shape, 3))
        overlay[gt == 1] = [0, 1, 0]  # Green 
        overlay[pred == 1] = [1, 0, 0]  # Red
        overlay[np.logical_and(pred, gt)] = [0.5, 0, 0.5]  # Purple

        plt.figure(figsize=(12, 6))
        plt.imshow(overlay)
        plt.title(f"Segmentation Evaluation: {os.path.splitext(filename)[0]}")
        plt.axis('off')

        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=4, label='Ground Truth'),
            plt.Line2D([0], [0], color='red', lw=4, label='Prediction'),
            plt.Line2D([0], [0], color='purple', lw=4, label='Overlap')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

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

        output_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_eval.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close()
        return output_file

    results = []
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('_binary.tif')]

    for pred_file in pred_files:
        if interactive:
            user_input = input(f"\nDo you want to process '{pred_file}'? (y/n): ").strip().lower()
            if user_input not in ('y', 'yes'):
                print(f"[SKIPPED] {pred_file}")
                continue

        try:
            gt_file = pred_file.replace('_binary', '_mask')
            gt_path_full = os.path.join(gt_path, gt_file)

            if not os.path.exists(gt_path_full):
                print(f"[SKIPPED] No ground truth found for {pred_file}")
                continue

            pred = (io.imread(os.path.join(pred_path, pred_file)) > 0).astype(np.uint8)
            gt = (io.imread(gt_path_full) > 0).astype(np.uint8)

            metrics = calculate_metrics(pred, gt)
            viz_path = create_visualization(pred, gt, pred_file, metrics)

            results.append({
                'image': pred_file,
                **metrics,
                'visualization': viz_path
            })

            print(f"[PROCESSED] {pred_file}")
            print(f"  Standard IoU: {metrics['iou']:.3f} | "
                  f"Foreground IoU: {metrics['iou_foreground']:.3f} | "
                  f"Precision: {metrics['precision']:.3f}")

        except Exception as e:
            print(f"[ERROR] Processing {pred_file}: {str(e)}")

    if results:
        csv_path = os.path.join(output_path, 'segmentation_metrics.csv')
        with open(csv_path, 'w') as f:
            f.write("Image,StandardIoU,ForegroundIoU,Precision,Accuracy,TP,FP,FN,TN,Visualization\n")  # Updated header
            for r in results:
                f.write(f"{r['image']},{r['iou']},{r['iou_foreground']},{r['precision']},{r['accuracy']},"
                       f"{r['tp']},{r['fp']},{r['fn']},{r['tn']},{r['visualization']}\n")

    print(f"\nEvaluation complete. Processed {len(results)} images.")
    print(f"Results saved to: {output_path}")
    return results

if __name__ == "__main__":
    results = evaluate_and_visualize_masks(
        pred_path=r"C:\Users\zindi\PycharmProjects\P2\Evaluations\SAM",
        gt_path=r"C:\Users\zindi\PycharmProjects\P2\Evaluations\Ground",
        output_path=r"C:\Users\zindi\PycharmProjects\P2\Evaluations\Results",
        interactive=False
    )