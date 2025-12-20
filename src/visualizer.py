import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

def draw_board(board_1d, save_path, step_num, prediction_conf=None, ground_truth_1d=None):
    """
    board_1d: shape (81,) containing values 1..10 (1=Empty, 2..10=Digits)
    ground_truth_1d: shape (81,) containing the CORRECT values
    prediction_conf: Can be a float (e.g. 0.95) or a string (e.g. "0.95 (Q:0.1)")
    """
    # Convert back to 0-9 for visualization (0=Empty, 1-9=Digits)
    board_vis = board_1d.copy() - 1 
    board = board_vis.reshape(9, 9)
    
    if ground_truth_1d is not None:
        gt_vis = ground_truth_1d.copy() - 1
        gt_board = gt_vis.reshape(9, 9)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Draw grid
    ax.imshow(np.zeros_like(board), cmap='binary', vmin=0, vmax=1)
    
    # Draw major grid lines
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i - 0.5, color='black', linewidth=lw)
        ax.axvline(i - 0.5, color='black', linewidth=lw)

    # Draw numbers
    for (i, j), z in np.ndenumerate(board):
        if z > 0: # If not empty
            text_color = 'black'
            
            # If we have ground truth, check correctness
            if ground_truth_1d is not None:
                correct_val = gt_board[i, j]
                if z == correct_val:
                    text_color = 'green'
                else:
                    text_color = 'red'
            
            ax.text(j, i, str(z), ha='center', va='center', fontsize=14, color=text_color, fontweight='bold')

    title = f"Step: {step_num}"
    if prediction_conf is not None:
        # Check if it's a number to apply formatting, otherwise print as is
        if isinstance(prediction_conf, (float, int)):
            title += f" | Conf: {prediction_conf:.2f}"
        else:
            title += f" | {prediction_conf}"
        
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_gif(step_folder, output_path):
    images = []
    # Sort by step number
    files = [f for f in os.listdir(step_folder) if f.endswith(".png")]
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for filename in files:
        images.append(imageio.v2.imread(os.path.join(step_folder, filename)))
    
    # Save GIF
    imageio.mimsave(output_path, images, fps=2)

def plot_convergence(train_losses, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()