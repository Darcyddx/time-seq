import torchvision.transforms.functional as F
import random
from PIL import Image
import torch
import torchvision.transforms as transforms


def original_rotate(img, angle):
    """Apply rotation to PIL image with proper interpolation (no zero padding)"""
    # Convert PIL to tensor for more control over rotation
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Convert to tensor
    img_tensor = to_tensor(img).unsqueeze(0)  # Add batch dimension

    # Apply rotation with interpolation mode that handles boundaries better
    # Use 'reflection' or 'border' fill mode instead of constant (zero) padding
    rotated_tensor = F.rotate(
        img_tensor,
        angle,
        interpolation=transforms.InterpolationMode.BILINEAR,
        expand=False,
        fill=None  # This will use the edge pixels for filling
    )

    # Alternative approach: use affine transformation with reflection padding
    # This gives better results for boundary handling
    rotated_tensor = apply_rotation_with_reflection(img_tensor, angle)

    # Convert back to PIL
    rotated_tensor = rotated_tensor.squeeze(0)  # Remove batch dimension
    return to_pil(rotated_tensor)


def apply_rotation_with_reflection(img_tensor, angle):
    """Apply rotation with reflection padding to avoid zero padding"""
    import torch.nn.functional as F_torch

    # Get image dimensions - img_tensor has shape (batch, channels, height, width)
    B, C, H, W = img_tensor.shape

    # Pad the image with reflection to avoid boundary issues
    # Calculate padding needed (rough estimate based on rotation)
    import math
    max_dim = max(H, W)
    pad_size = int(max_dim * 0.5)  # Padding size

    # Apply reflection padding
    padded_img = F_torch.pad(img_tensor,
                             (pad_size, pad_size, pad_size, pad_size),
                             mode='reflect')

    # Apply rotation to padded image
    rotated_padded = F.rotate(
        padded_img,
        angle,
        interpolation=transforms.InterpolationMode.BILINEAR,
        expand=False
    )

    # Crop back to original size from center
    B_pad, C_pad, H_pad, W_pad = rotated_padded.shape
    start_h = (H_pad - H) // 2
    start_w = (W_pad - W) // 2

    cropped = rotated_padded[:, :, start_h:start_h + H, start_w:start_w + W]

    return cropped


def alternative_rotate_with_edge_fill(img, angle):
    """Alternative method using cv2 for better boundary handling"""
    import cv2
    import numpy as np

    # Convert PIL to numpy array
    img_np = np.array(img)

    # Get image dimensions
    (h, w) = img_np.shape[:2]

    # Calculate rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation with border reflection
    rotated_np = cv2.warpAffine(
        img_np,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101  # Use reflection instead of constant
    )

    # Convert back to PIL
    return Image.fromarray(rotated_np)


def time_aug_rotate(img, tau=7):
    """Generate temporal sequence by gradually rotating from 0 to a random target angle"""
    if tau == 1:
        # Traditional augmentation: single random rotation
        target_angle = random.uniform(-180, 180)
        return [original_rotate(img, target_angle)]

    # Generate random target angle for the final frame
    target_angle = random.uniform(-180, 180)

    # Create sequence of angles that gradually change from 0 to target_angle
    angles = [target_angle * t / (tau - 1) for t in range(tau)]

    # Apply rotation to each frame
    return [original_rotate(img, angle) for angle in angles]


def time_aug_rotate_cv2(img, tau=7):
    """Alternative version using cv2 for better boundary handling"""
    if tau == 1:
        # Traditional augmentation: single random rotation
        target_angle = random.uniform(-180, 180)
        return [alternative_rotate_with_edge_fill(img, target_angle)]

    # Generate random target angle for the final frame
    target_angle = random.uniform(-180, 180)

    # Create sequence of angles that gradually change from 0 to target_angle
    angles = [target_angle * t / (tau - 1) for t in range(tau)]

    # Apply rotation to each frame
    return [alternative_rotate_with_edge_fill(img, angle) for angle in angles]


def main():
    """Main function to visualize rotation augmentation"""
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
    traditional_aug = time_aug_rotate_cv2(img, tau=1)

    # Temporal sequence (tau=5)
    print("Generating temporal sequence (tau=5)...")
    temporal_seq = time_aug_rotate_cv2(img, tau=5)

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
    for i, rotated_img in enumerate(temporal_seq):
        axes[1, i].imshow(rotated_img)
        axes[1, i].set_title(f'Frame {i + 1}/5')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('rotation_visualization_fixed.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'rotation_visualization.png'")

    # Show angle progression for the temporal sequence
    print("\nAngle progression for temporal sequence:")
    target_angle = random.uniform(-180, 180)
    angles = [target_angle * t / (5 - 1) for t in range(5)]
    for i, angle in enumerate(angles):
        print(f"Frame {i + 1}: {angle:.1f}°")


if __name__ == "__main__":
    main()