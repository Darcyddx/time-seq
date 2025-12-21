import torchvision.transforms.functional as F
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import math


def original_zoom(img, scale_factor):
    """Apply zoom to PIL image with proper handling of boundaries"""
    # Convert PIL to tensor for more control over zoom
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Convert to tensor
    img_tensor = to_tensor(img).unsqueeze(0)  # Add batch dimension

    # Apply zoom with reflection padding to avoid zero padding
    zoomed_tensor = apply_zoom_with_reflection(img_tensor, scale_factor)

    # Convert back to PIL
    zoomed_tensor = zoomed_tensor.squeeze(0)  # Remove batch dimension
    return to_pil(zoomed_tensor)


def apply_zoom_with_reflection(img_tensor, scale_factor):
    """Apply zoom with reflection padding to avoid zero padding"""
    import torch.nn.functional as F_torch

    # Get image dimensions
    B, C, H, W = img_tensor.shape

    # Clamp scale factor to reasonable range to avoid extreme padding
    scale_factor = max(0.1, min(5.0, scale_factor))

    if scale_factor > 1.0:
        # Zoom in: crop center region and resize to original size
        crop_h = max(1, int(H / scale_factor))
        crop_w = max(1, int(W / scale_factor))

        # Calculate crop coordinates (center crop)
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2

        # Crop the center region
        cropped = img_tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

        # Resize back to original size using resize
        zoomed = F.resize(cropped.squeeze(0), (H, W)).unsqueeze(0)

    else:
        # Zoom out: resize image smaller and pad with reflection
        new_h = max(1, int(H * scale_factor))
        new_w = max(1, int(W * scale_factor))

        # Resize image smaller using resize
        resized = F.resize(img_tensor.squeeze(0), (new_h, new_w)).unsqueeze(0)

        # Calculate padding needed
        total_pad_h = H - new_h
        total_pad_w = W - new_w

        # Check if we need to pad at all
        if total_pad_h <= 0 or total_pad_w <= 0:
            # If no padding needed, just resize back to original size
            zoomed = F.resize(resized.squeeze(0), (H, W)).unsqueeze(0)
        else:
            # Calculate individual padding amounts
            pad_h_top = total_pad_h // 2
            pad_h_bottom = total_pad_h - pad_h_top
            pad_w_left = total_pad_w // 2
            pad_w_right = total_pad_w - pad_w_left

            # Ensure padding is not larger than the resized image dimensions
            # PyTorch pad format: (pad_left, pad_right, pad_top, pad_bottom)
            pad_w_left = min(pad_w_left, new_w - 1)
            pad_w_right = min(pad_w_right, new_w - 1)
            pad_h_top = min(pad_h_top, new_h - 1)
            pad_h_bottom = min(pad_h_bottom, new_h - 1)

            try:
                # Apply reflection padding
                zoomed = F_torch.pad(resized,
                                     (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom),
                                     mode='reflect')

                # If the result is not the right size, resize to match
                if zoomed.shape[2] != H or zoomed.shape[3] != W:
                    zoomed = F.resize(zoomed.squeeze(0), (H, W)).unsqueeze(0)

            except RuntimeError:
                # Fallback: just resize the small image back to original size
                zoomed = F.resize(resized.squeeze(0), (H, W)).unsqueeze(0)

    return zoomed


def alternative_zoom_with_edge_fill(img, scale_factor):
    """Alternative method using cv2 for better boundary handling"""
    import cv2
    import numpy as np

    # Convert PIL to numpy array
    img_np = np.array(img)

    # Get image dimensions
    (h, w) = img_np.shape[:2]

    # Clamp scale factor to reasonable range
    scale_factor = max(0.1, min(5.0, scale_factor))

    if scale_factor > 1.0:
        # Zoom in: crop center region and resize to original size
        crop_h = max(1, int(h / scale_factor))
        crop_w = max(1, int(w / scale_factor))

        # Calculate crop coordinates (center crop)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2

        # Crop the center region
        cropped = img_np[start_h:start_h + crop_h, start_w:start_w + crop_w]

        # Resize back to original size
        zoomed_np = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    else:
        # Zoom out: resize image smaller and pad with reflection
        new_h = max(1, int(h * scale_factor))
        new_w = max(1, int(w * scale_factor))

        # Resize image smaller
        resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate padding needed
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2

        try:
            # Apply reflection padding
            zoomed_np = cv2.copyMakeBorder(
                resized,
                pad_h, h - new_h - pad_h,  # top, bottom
                pad_w, w - new_w - pad_w,  # left, right
                cv2.BORDER_REFLECT_101
            )
        except cv2.error:
            # Fallback: just resize back to original size
            zoomed_np = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert back to PIL
    return Image.fromarray(zoomed_np)


def time_aug_zoom(img, tau=7, zoom_range=(0.8, 1.2)):
    """Generate temporal sequence by gradually zooming from 1.0 to a random target scale"""
    if tau == 1:
        # Traditional augmentation: single random zoom
        target_scale = random.uniform(zoom_range[0], zoom_range[1])
        return [original_zoom(img, target_scale)]

    # Generate random target scale for the final frame
    target_scale = random.uniform(zoom_range[0], zoom_range[1])

    # Create sequence of scales that gradually change from 1.0 to target_scale
    scales = [1.0 + (target_scale - 1.0) * t / (tau - 1) for t in range(tau)]

    # Apply zoom to each frame
    return [original_zoom(img, scale) for scale in scales]


def time_aug_zoom_cv2(img, tau=7, zoom_range=(0.8, 1.2)):
    """Alternative version using cv2 for better boundary handling"""
    if tau == 1:
        # Traditional augmentation: single random zoom
        target_scale = random.uniform(zoom_range[0], zoom_range[1])
        return [alternative_zoom_with_edge_fill(img, target_scale)]

    # Generate random target scale for the final frame
    target_scale = random.uniform(zoom_range[0], zoom_range[1])

    # Create sequence of scales that gradually change from 1.0 to target_scale
    scales = [1.0 + (target_scale - 1.0) * t / (tau - 1) for t in range(tau)]

    # Apply zoom to each frame
    return [alternative_zoom_with_edge_fill(img, scale) for scale in scales]


def main():
    """Main function to visualize zoom augmentation"""
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
    traditional_aug = time_aug_zoom_cv2(img, tau=1, zoom_range=(0.5, 2.0))

    # Temporal sequence (tau=5)
    print("Generating temporal sequence (tau=5)...")
    temporal_seq = time_aug_zoom_cv2(img, tau=5, zoom_range=(0.5, 2.0))

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
    for i, zoomed_img in enumerate(temporal_seq):
        axes[1, i].imshow(zoomed_img)
        axes[1, i].set_title(f'Frame {i + 1}/5')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('zoom_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'zoom_visualization.png'")

    # Show scale progression for the temporal sequence
    print("\nScale progression for temporal sequence:")
    target_scale = random.uniform(0.5, 2.0)
    scales = [1.0 + (target_scale - 1.0) * t / (5 - 1) for t in range(5)]
    for i, scale in enumerate(scales):
        print(f"Frame {i + 1}: {scale:.2f}x")
        if scale > 1.0:
            print(f"  -> Zoom in by {scale:.2f}x")
        elif scale < 1.0:
            print(f"  -> Zoom out by {1 / scale:.2f}x")
        else:
            print(f"  -> No zoom (original size)")


if __name__ == "__main__":
    main()