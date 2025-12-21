import torchvision.transforms.functional as F
import random
from PIL import Image
import torch
import torchvision.transforms as transforms


def original_shear(img, shear_x, shear_y):
    """Apply shear transformation to PIL image with proper interpolation (no zero padding)"""
    # Convert PIL to tensor for more control over shear
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Convert to tensor
    img_tensor = to_tensor(img).unsqueeze(0)  # Add batch dimension

    # Apply shear with interpolation mode that handles boundaries better
    # Use reflection padding approach for better boundary handling
    sheared_tensor = apply_shear_with_reflection(img_tensor, shear_x, shear_y)

    # Convert back to PIL
    sheared_tensor = sheared_tensor.squeeze(0)  # Remove batch dimension
    return to_pil(sheared_tensor)


def apply_shear_with_reflection(img_tensor, shear_x, shear_y):
    """Apply shear with reflection padding to avoid zero padding"""
    import torch.nn.functional as F_torch

    # Get image dimensions - img_tensor has shape (batch, channels, height, width)
    B, C, H, W = img_tensor.shape

    # Pad the image with reflection to avoid boundary issues
    # Calculate padding needed based on shear amount
    import math
    max_dim = max(H, W)
    # Padding size based on maximum possible displacement from shear
    pad_size = int(max_dim * (abs(shear_x) + abs(shear_y)) * 0.5 + max_dim * 0.1)

    # Apply reflection padding
    padded_img = F_torch.pad(img_tensor,
                             (pad_size, pad_size, pad_size, pad_size),
                             mode='reflect')

    # Apply shear to padded image
    sheared_padded = F.affine(
        padded_img,
        angle=0,
        translate=[0, 0],
        scale=1,
        shear=[shear_x, shear_y],
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=None
    )

    # Crop back to original size from center
    B_pad, C_pad, H_pad, W_pad = sheared_padded.shape
    start_h = (H_pad - H) // 2
    start_w = (W_pad - W) // 2

    cropped = sheared_padded[:, :, start_h:start_h + H, start_w:start_w + W]

    return cropped


def alternative_shear_with_edge_fill(img, shear_x, shear_y):
    """Alternative method using cv2 for better boundary handling"""
    import cv2
    import numpy as np

    # Convert PIL to numpy array
    img_np = np.array(img)

    # Get image dimensions
    (h, w) = img_np.shape[:2]

    # Create shear transformation matrix
    # Shear matrix: [[1, shear_x], [shear_y, 1]]
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    # Apply shear transformation with border reflection
    sheared_np = cv2.warpAffine(
        img_np,
        shear_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101  # Use reflection instead of constant
    )

    # Convert back to PIL
    return Image.fromarray(sheared_np)


def time_aug_shear(img, tau=7, shear_ratio=0.5):
    """Generate temporal sequence by gradually shearing from 0 to random target shear values"""
    # Use CV2 version for better results (same as time_aug_shear_cv2)
    if tau == 1:
        # Traditional augmentation: single random shear
        target_shear_x = random.uniform(-shear_ratio, shear_ratio)
        target_shear_y = random.uniform(-shear_ratio, shear_ratio)
        return [alternative_shear_with_edge_fill(img, target_shear_x, target_shear_y)]

    # Generate random target shear values for the final frame
    target_shear_x = random.uniform(-shear_ratio, shear_ratio)
    target_shear_y = random.uniform(-shear_ratio, shear_ratio)

    # Create sequence of shear values that gradually change from 0 to target values
    shear_x_values = [target_shear_x * t / (tau - 1) for t in range(tau)]
    shear_y_values = [target_shear_y * t / (tau - 1) for t in range(tau)]

    # Apply shear to each frame using CV2 version
    return [alternative_shear_with_edge_fill(img, shear_x, shear_y)
            for shear_x, shear_y in zip(shear_x_values, shear_y_values)]


def time_aug_shear_cv2(img, tau=7, shear_ratio=0.5):
    """Alternative version using cv2 for better boundary handling"""
    if tau == 1:
        # Traditional augmentation: single random shear
        target_shear_x = random.uniform(-shear_ratio, shear_ratio)
        target_shear_y = random.uniform(-shear_ratio, shear_ratio)
        return [alternative_shear_with_edge_fill(img, target_shear_x, target_shear_y)]

    # Generate random target shear values for the final frame
    target_shear_x = random.uniform(-shear_ratio, shear_ratio)
    target_shear_y = random.uniform(-shear_ratio, shear_ratio)

    # Create sequence of shear values that gradually change from 0 to target values
    shear_x_values = [target_shear_x * t / (tau - 1) for t in range(tau)]
    shear_y_values = [target_shear_y * t / (tau - 1) for t in range(tau)]

    # Apply shear to each frame
    return [alternative_shear_with_edge_fill(img, shear_x, shear_y)
            for shear_x, shear_y in zip(shear_x_values, shear_y_values)]


def main():
    """Main function to visualize shear augmentation"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid PyCharm issues

    # Load the sample image
    img_path = "wing.jpg"
    try:
        img = Image.open(img_path)
        print(f"Loaded image: {img_path}, Size: {img.size}")
    except FileNotFoundError:
        print(f"Error: Could not find image at {img_path}")
        return

    # Traditional augmentation (tau=1)
    print("\nGenerating traditional augmentation (tau=1)...")
    traditional_aug = time_aug_shear_cv2(img, tau=1, shear_ratio=0.5)

    # Temporal sequence (tau=5)
    print("Generating temporal sequence (tau=5)...")
    temporal_seq = time_aug_shear_cv2(img, tau=5, shear_ratio=0.5)

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # First row: Original + Traditional augmentation
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(traditional_aug[0])
    axes[0, 1].set_title('Traditional Aug (tau=1)')
    axes[0, 1].axis('off')

    # Empty plots for first row
    for i in range(2, 5):
        axes[0, i].axis('off')

    # Second row: Temporal sequence
    for i, sheared_img in enumerate(temporal_seq):
        axes[1, i].imshow(sheared_img)
        axes[1, i].set_title(f'Frame {i + 1}/5')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('shear_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'shear_visualization.png'")

    # Show shear progression for the temporal sequence
    print("\nShear progression for temporal sequence:")
    target_shear_x = random.uniform(-0.8, 0.8)
    target_shear_y = random.uniform(-0.8, 0.8)
    shear_x_values = [target_shear_x * t / (5 - 1) for t in range(5)]
    shear_y_values = [target_shear_y * t / (5 - 1) for t in range(5)]

    for i, (shear_x, shear_y) in enumerate(zip(shear_x_values, shear_y_values)):
        print(f"Frame {i + 1}: shear_x={shear_x:.3f}, shear_y={shear_y:.3f}")


if __name__ == "__main__":
    main()