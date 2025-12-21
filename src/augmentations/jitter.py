import torchvision.transforms.functional as F
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np


def apply_color_jitter(img, brightness_factor=None, contrast_factor=None,
                       saturation_factor=None, hue_factor=None):
    """Apply color jitter with specific factors to PIL image"""
    if brightness_factor is not None:
        img = F.adjust_brightness(img, brightness_factor)
    if contrast_factor is not None:
        img = F.adjust_contrast(img, contrast_factor)
    if saturation_factor is not None:
        img = F.adjust_saturation(img, saturation_factor)
    if hue_factor is not None:
        img = F.adjust_hue(img, hue_factor)
    return img


def generate_jitter_params(brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1),
                           saturation_range=(0.95, 1.05), hue_range=(-0.05, 0.05)):
    """Generate random jitter parameters within specified ranges"""
    brightness_factor = random.uniform(*brightness_range) if brightness_range else None
    contrast_factor = random.uniform(*contrast_range) if contrast_range else None
    saturation_factor = random.uniform(*saturation_range) if saturation_range else None
    hue_factor = random.uniform(*hue_range) if hue_range else None

    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def interpolate_jitter_params(start_params, end_params, t):
    """Interpolate between start and end jitter parameters

    Args:
        start_params: tuple of (brightness, contrast, saturation, hue) for start frame
        end_params: tuple of (brightness, contrast, saturation, hue) for end frame
        t: interpolation factor between 0 and 1
    """
    interpolated = []
    for start_val, end_val in zip(start_params, end_params):
        if start_val is None or end_val is None:
            interpolated.append(None)
        else:
            interpolated.append(start_val + t * (end_val - start_val))
    return tuple(interpolated)


def time_aug_jitter(img, tau=7, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3),
                    saturation_range=(0.95, 1.05), hue_range=(-0.05, 0.05)):
    """Generate temporal sequence by gradually applying color jitter from original to target

    Args:
        img: PIL Image
        tau: number of frames in the sequence
        brightness_range: tuple of (min, max) brightness factors
        contrast_range: tuple of (min, max) contrast factors
        saturation_range: tuple of (min, max) saturation factors
        hue_range: tuple of (min, max) hue shift values
    """
    if tau == 1:
        # Traditional augmentation: single random jitter
        target_params = generate_jitter_params(brightness_range, contrast_range,
                                               saturation_range, hue_range)
        return [apply_color_jitter(img, *target_params)]

    # Generate random target parameters for the final frame
    target_params = generate_jitter_params(brightness_range, contrast_range,
                                           saturation_range, hue_range)

    # Start parameters (no change from original)
    start_params = (1.0, 1.0, 1.0, 0.0)  # neutral values

    # Create sequence of parameters that gradually change from start to target
    sequence = []
    for t in range(tau):
        # Interpolation factor from 0 to 1
        alpha = t / (tau - 1)

        # Interpolate parameters
        current_params = interpolate_jitter_params(start_params, target_params, alpha)

        # Apply jitter with current parameters
        jittered_img = apply_color_jitter(img, *current_params)
        sequence.append(jittered_img)

    return sequence

def main():
    """Main function to visualize color jitter augmentation"""
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
        # Create a dummy image for demonstration
        print("Creating dummy image for demonstration...")
        img = Image.new('RGB', (224, 224), color='red')
        # Add some pattern to make jitter effects more visible
        pixels = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                r = int(255 * (i / img.size[0]))
                g = int(255 * (j / img.size[1]))
                b = int(255 * ((i + j) / (img.size[0] + img.size[1])))
                pixels[i, j] = (r, g, b)

    # Traditional augmentation (tau=1)
    print("\nGenerating traditional augmentation (tau=1)...")
    traditional_aug = time_aug_jitter(img, tau=1)

    # Temporal sequence (tau=5)
    print("Generating temporal sequence (tau=5)...")
    temporal_seq = time_aug_jitter(img, tau=5)

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

    # Second row: Temporal sequence (subtle parameters)
    for i, jittered_img in enumerate(temporal_seq):
        axes[1, i].imshow(jittered_img)
        axes[1, i].set_title(f'Subtle Frame {i + 1}/5')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('jitter_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'jitter_visualization.png'")

    # Show parameter progression for the temporal sequence
    print("\nParameter progression for temporal sequence (subtle):")

    # Generate subtle ranges
    brightness_min = random.uniform(0.85, 0.95)
    brightness_max = random.uniform(1.05, 1.15)
    contrast_min = random.uniform(0.85, 0.95)
    contrast_max = random.uniform(1.05, 1.15)
    saturation_min = random.uniform(0.9, 0.98)
    saturation_max = random.uniform(1.02, 1.1)
    hue_min = random.uniform(-0.08, -0.02)
    hue_max = random.uniform(0.02, 0.08)

    target_params = generate_jitter_params(
        brightness_range=(brightness_min, brightness_max),
        contrast_range=(contrast_min, contrast_max),
        saturation_range=(saturation_min, saturation_max),
        hue_range=(hue_min, hue_max)
    )
    start_params = (1.0, 1.0, 1.0, 0.0)

    print(f"Target parameters: brightness={target_params[0]:.3f}, contrast={target_params[1]:.3f}, "
          f"saturation={target_params[2]:.3f}, hue={target_params[3]:.3f}")

    for t in range(5):
        alpha = t / (5 - 1)
        current_params = interpolate_jitter_params(start_params, target_params, alpha)
        print(f"Frame {t + 1}: brightness={current_params[0]:.3f}, contrast={current_params[1]:.3f}, "
              f"saturation={current_params[2]:.3f}, hue={current_params[3]:.3f}")


if __name__ == "__main__":
    main()