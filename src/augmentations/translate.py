import torchvision.transforms.functional as F
import random
from PIL import Image
import torch
import torchvision.transforms as transforms


def original_translate(img, translate_x, translate_y):
    """Apply translation to PIL image with proper interpolation (no zero padding)"""
    # Convert PIL to tensor for more control over translation
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Convert to tensor
    img_tensor = to_tensor(img).unsqueeze(0)  # Add batch dimension

    # Apply translation with interpolation mode that handles boundaries better
    # Use affine transformation with reflection padding
    translated_tensor = apply_translation_with_reflection(img_tensor, translate_x, translate_y)

    # Convert back to PIL
    translated_tensor = translated_tensor.squeeze(0)  # Remove batch dimension
    return to_pil(translated_tensor)


def apply_translation_with_reflection(img_tensor, translate_x, translate_y):
    """Apply translation with reflection padding to avoid zero padding"""
    import torch.nn.functional as F_torch

    # Get image dimensions - img_tensor has shape (batch, channels, height, width)
    B, C, H, W = img_tensor.shape

    # Pad the image with reflection to avoid boundary issues
    # Calculate padding needed based on translation amount
    pad_x = int(abs(translate_x) * W) + 10  # Add some extra padding
    pad_y = int(abs(translate_y) * H) + 10

    # Apply reflection padding
    padded_img = F_torch.pad(img_tensor,
                             (pad_x, pad_x, pad_y, pad_y),
                             mode='reflect')

    # Calculate translation in pixels for the padded image
    B_pad, C_pad, H_pad, W_pad = padded_img.shape
    translate_x_pixels = translate_x * W
    translate_y_pixels = translate_y * H

    # Create affine transformation matrix for translation
    # PyTorch affine expects [2x3] matrix in format:
    # [[a, b, c], [d, e, f]] where transformation is [a*x + b*y + c, d*x + e*y + f]
    # For translation: [[1, 0, translate_x], [0, 1, translate_y]]
    theta = torch.tensor([[[1, 0, translate_x_pixels / (W_pad / 2)],
                           [0, 1, translate_y_pixels / (H_pad / 2)]]], dtype=torch.float32)

    # Create sampling grid
    grid = F_torch.affine_grid(theta, padded_img.size(), align_corners=False)

    # Apply translation
    translated_padded = F_torch.grid_sample(padded_img, grid,
                                            mode='bilinear',
                                            padding_mode='reflection',
                                            align_corners=False)

    # Crop back to original size from center
    start_h = (H_pad - H) // 2
    start_w = (W_pad - W) // 2

    cropped = translated_padded[:, :, start_h:start_h + H, start_w:start_w + W]

    return cropped


def alternative_translate_with_edge_fill(img, translate_x, translate_y):
    """Alternative method using cv2 for better boundary handling"""
    import cv2
    import numpy as np

    # Convert PIL to numpy array
    img_np = np.array(img)

    # Get image dimensions
    (h, w) = img_np.shape[:2]

    # Calculate translation in pixels
    translate_x_pixels = translate_x * w
    translate_y_pixels = translate_y * h

    # Create translation matrix
    # Translation matrix: [[1, 0, tx], [0, 1, ty]]
    translation_matrix = np.float32([[1, 0, translate_x_pixels],
                                     [0, 1, translate_y_pixels]])

    # Apply translation with border reflection
    translated_np = cv2.warpAffine(
        img_np,
        translation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101  # Use reflection instead of constant
    )

    # Convert back to PIL
    return Image.fromarray(translated_np)


def time_aug_translate(img, tau=7, translate_ratio=0.2):
    """Generate temporal sequence by gradually translating from 0 to random target translation"""
    if tau == 1:
        # Traditional augmentation: single random translation
        target_translate_x = random.uniform(-translate_ratio, translate_ratio)
        target_translate_y = random.uniform(-translate_ratio, translate_ratio)
        return [original_translate(img, target_translate_x, target_translate_y)]

    # Generate random target translation for the final frame
    target_translate_x = random.uniform(-translate_ratio, translate_ratio)
    target_translate_y = random.uniform(-translate_ratio, translate_ratio)

    # Create sequence of translations that gradually change from (0,0) to target translation
    translate_x_sequence = [target_translate_x * t / (tau - 1) for t in range(tau)]
    translate_y_sequence = [target_translate_y * t / (tau - 1) for t in range(tau)]

    # Apply translation to each frame
    return [original_translate(img, tx, ty) for tx, ty in zip(translate_x_sequence, translate_y_sequence)]


def time_aug_translate_cv2(img, tau=7, translate_ratio=0.2):
    """Alternative version using cv2 for better boundary handling"""
    if tau == 1:
        # Traditional augmentation: single random translation
        target_translate_x = random.uniform(-translate_ratio, translate_ratio)
        target_translate_y = random.uniform(-translate_ratio, translate_ratio)
        return [alternative_translate_with_edge_fill(img, target_translate_x, target_translate_y)]

    # Generate random target translation for the final frame
    target_translate_x = random.uniform(-translate_ratio, translate_ratio)
    target_translate_y = random.uniform(-translate_ratio, translate_ratio)

    # Create sequence of translations that gradually change from (0,0) to target translation
    translate_x_sequence = [target_translate_x * t / (tau - 1) for t in range(tau)]
    translate_y_sequence = [target_translate_y * t / (tau - 1) for t in range(tau)]

    # Apply translation to each frame
    return [alternative_translate_with_edge_fill(img, tx, ty) for tx, ty in
            zip(translate_x_sequence, translate_y_sequence)]


def main():
    """Main function to visualize translation augmentation"""
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
    traditional_aug = time_aug_translate_cv2(img, tau=1)

    # Temporal sequence (tau=5)
    print("Generating temporal sequence (tau=5)...")
    temporal_seq = time_aug_translate_cv2(img, tau=5)

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
    for i, translated_img in enumerate(temporal_seq):
        axes[1, i].imshow(translated_img)
        axes[1, i].set_title(f'Frame {i + 1}/5')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('translation_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'translation_visualization.png'")

    # Show translation progression for the temporal sequence
    print("\nTranslation progression for temporal sequence:")
    target_translate_x = random.uniform(-0.8, 0.8)
    target_translate_y = random.uniform(-0.8, 0.8)
    translate_x_sequence = [target_translate_x * t / (5 - 1) for t in range(5)]
    translate_y_sequence = [target_translate_y * t / (5 - 1) for t in range(5)]

    for i, (tx, ty) in enumerate(zip(translate_x_sequence, translate_y_sequence)):
        print(f"Frame {i + 1}: translate_x={tx:.2f}, translate_y={ty:.2f}")


if __name__ == "__main__":
    main()