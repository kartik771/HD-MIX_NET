import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_metrics(metrics_path='./checkpoints/metrics_history.json'):
    with open(metrics_path, 'r') as f:
        return json.load(f)

def moving_average(arr, window=3):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='valid')

def plot_all_metrics():
    metrics = load_metrics()
    epochs = np.array(metrics['epochs'])

    fig = plt.figure(figsize=(20, 14))

    # 1. Training Loss
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(epochs, metrics['train_loss'], 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Training Loss\n(Lower = Better Learning)', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Dice Score
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(epochs, metrics['val_dice'], 'g-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_title('Validation Dice Score\n(Higher = Better Segmentation)', fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)

    # 3. IoU Score
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(epochs, metrics['val_iou'], 'r-o', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('IoU Score', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    ax3.set_title('Validation IoU Score\n(Higher = Better Overlap)', fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3)

    # 4. Hausdorff 95 Distance
    ax4 = plt.subplot(3, 4, 4)
    hd95_clean = [x for x in metrics['val_hd95'] if x is not None]
    epochs_hd95 = [epochs[i] for i in range(len(metrics['val_hd95'])) if metrics['val_hd95'][i] is not None]
    if hd95_clean:
        ax4.plot(epochs_hd95, hd95_clean, 'purple', marker='o', linewidth=2, markersize=4)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('HD95 Distance', color='purple')
        ax4.tick_params(axis='y', labelcolor='purple')
        ax4.set_title('Hausdorff 95 Distance\n(Lower = Better Boundaries)', fontweight='bold')
        ax4.grid(True, alpha=0.3)

    # 5. Loss vs Dice Correlation
    ax5 = plt.subplot(3, 4, 5)
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(epochs, metrics['train_loss'], 'b-o', linewidth=2, label='Train Loss', markersize=4)
    line2 = ax5_twin.plot(epochs, metrics['val_dice'], 'g-s', linewidth=2, label='Val Dice', markersize=4)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Train Loss', color='b')
    ax5_twin.set_ylabel('Val Dice', color='g')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5_twin.tick_params(axis='y', labelcolor='g')
    ax5.set_title('Loss vs Dice Correlation\n(Loss ↓ & Dice ↑ = Good Learning)', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Learning Rate Schedule
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(epochs, metrics['learning_rate'], 'orange', marker='^', linewidth=2, markersize=6)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Learning Rate', color='orange')
    ax6.tick_params(axis='y', labelcolor='orange')
    ax6.set_title('Learning Rate Schedule\n(How LR changes per epoch)', fontweight='bold')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)

    # 7. Optimal Threshold Evolution
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(epochs, metrics['val_threshold'], 'brown', marker='D', linewidth=2, markersize=5)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Threshold', color='brown')
    ax7.tick_params(axis='y', labelcolor='brown')
    ax7.set_title('Optimal Threshold Evolution\n(Best prediction threshold per epoch)', fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # 8. Smoothed Trends (Moving Average)
    ax8 = plt.subplot(3, 4, 8)
    if len(metrics['train_loss']) >= 3:
        loss_smooth = moving_average(metrics['train_loss'], window=3)
        dice_smooth = moving_average(metrics['val_dice'], window=3)
        epochs_smooth = moving_average(epochs, window=3)
        ax8_2 = ax8.twinx()
        ax8.plot(epochs_smooth, loss_smooth, 'b-', linewidth=2.5, label='Loss (MA)')
        ax8_2.plot(epochs_smooth, dice_smooth, 'g-', linewidth=2.5, label='Dice (MA)')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Loss (Moving Avg)', color='b')
        ax8_2.set_ylabel('Dice (Moving Avg)', color='g')
        ax8.tick_params(axis='y', labelcolor='b')
        ax8_2.tick_params(axis='y', labelcolor='g')
    ax8.set_title('Smoothed Trends\n(3-epoch moving average)', fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # 9. Per-Epoch Improvement Rate
    ax9 = plt.subplot(3, 4, 9)
    dice_improvement = np.diff(metrics['val_dice'])
    epochs_diff = epochs[1:]
    colors = ['green' if x > 0 else 'red' for x in dice_improvement]
    ax9.bar(epochs_diff, dice_improvement, color=colors, alpha=0.7, edgecolor='black')
    ax9.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Dice Improvement', color='darkgreen')
    ax9.set_title('Per-Epoch Improvement Rate\n(Green = Better, Red = Worse)', fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')

    # 10. All Metrics Normalized
    ax10 = plt.subplot(3, 4, 10)
    dice_norm = np.array(metrics['val_dice']) / (max(metrics['val_dice']) + 1e-6)
    iou_norm = np.array(metrics['val_iou']) / (max(metrics['val_iou']) + 1e-6)
    loss_norm = (np.max(metrics['train_loss']) - np.array(metrics['train_loss'])) / (np.max(metrics['train_loss']) - np.min(metrics['train_loss']) + 1e-6)

    ax10.plot(epochs, dice_norm, 'g-o', label='Dice (Norm)', linewidth=2, markersize=4)
    ax10.plot(epochs, iou_norm, 'r-s', label='IoU (Norm)', linewidth=2, markersize=4)
    ax10.plot(epochs, loss_norm, 'b-^', label='Loss Reduction (Norm)', linewidth=2, markersize=4)
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('Normalized Score')
    ax10.set_title('All Metrics Normalized\n(0-1 scale for comparison)', fontweight='bold')
    ax10.legend(loc='best', fontsize=9)
    ax10.set_ylim([0, 1.1])
    ax10.grid(True, alpha=0.3)

    # 11. Convergence Analysis (Distance to Best)
    ax11 = plt.subplot(3, 4, 11)
    best_dice = max(metrics['val_dice'])
    distance_to_best = best_dice - np.array(metrics['val_dice'])
    ax11.semilogy(epochs, distance_to_best + 1e-6, 'r-o', linewidth=2, markersize=4)
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('Distance to Best Dice')
    ax11.set_title('Convergence Analysis\n(Lower = Closer to optimum)', fontweight='bold')
    ax11.grid(True, alpha=0.3, which='both')

    # 12. Cumulative Improvement
    ax12 = plt.subplot(3, 4, 12)
    cumulative_improvement = np.cumsum(np.maximum(dice_improvement, 0))
    ax12.plot(epochs_diff, cumulative_improvement, 'purple', marker='o', linewidth=2.5, markersize=5)
    ax12.fill_between(epochs_diff, cumulative_improvement, alpha=0.3, color='purple')
    ax12.set_xlabel('Epoch')
    ax12.set_ylabel('Cumulative Improvement')
    ax12.set_title('Cumulative Improvement Over Time\n(Total positive changes)', fontweight='bold')
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./checkpoints/training_metrics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: ./checkpoints/training_metrics.png")
    plt.show()

def plot_individual_graphs():
    """Create individual high-quality graphs"""
    metrics = load_metrics()
    epochs = np.array(metrics['epochs'])

    plots_info = [
        ('Loss', metrics['train_loss'], 'blue', 'Training Loss Over Epochs\n(Lower loss = model learning better)'),
        ('Dice', metrics['val_dice'], 'green', 'Dice Score Over Epochs\n(Higher score = better segmentation accuracy)'),
        ('IoU', metrics['val_iou'], 'red', 'IoU Score Over Epochs\n(Measures overlap between predicted and actual masks)'),
    ]

    for name, values, color, title in plots_info:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values, color=color, marker='o', linewidth=2.5, markersize=8)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(name, fontsize=12, color=color)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./checkpoints/metric_{name.lower()}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: ./checkpoints/metric_{name.lower()}.png")
        plt.close()

def print_metrics_summary():
    """Print summary statistics"""
    metrics = load_metrics()
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)
    print(f"Total Epochs: {len(metrics['epochs'])}")
    print(f"\nLoss: {metrics['train_loss'][0]:.4f} → {metrics['train_loss'][-1]:.4f}")
    print(f"Dice: {metrics['val_dice'][0]:.4f} → {metrics['val_dice'][-1]:.4f} (Best: {max(metrics['val_dice']):.4f})")
    print(f"IoU:  {metrics['val_iou'][0]:.4f} → {metrics['val_iou'][-1]:.4f} (Best: {max(metrics['val_iou']):.4f})")

    hd95_clean = [x for x in metrics['val_hd95'] if x is not None]
    if hd95_clean:
        print(f"HD95: {hd95_clean[0]:.4f} → {hd95_clean[-1]:.4f} (Best: {min(hd95_clean):.4f})")

    print(f"Learning Rate: {metrics['learning_rate'][0]:.6f} → {metrics['learning_rate'][-1]:.6f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    print("Generating training visualizations...")
    try:
        plot_all_metrics()
        plot_individual_graphs()
        print_metrics_summary()
        print("\n✓ All visualizations complete!")
    except FileNotFoundError:
        print("Error: metrics_history.json not found. Run training first!")
    except Exception as e:
        print(f"Error: {e}")
