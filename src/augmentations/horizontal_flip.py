import torchvision.transforms.functional as F
import random
from PIL import Image


def interp_images(img1, img2, alpha):
    """Blend two PIL images by alpha (0..1)"""
    return Image.blend(img1, img2, alpha)


def original_horizontal_flip(img):
    """Apply horizontal flip to PIL image"""
    return F.hflip(img)


def time_aug_horizontal_flip(img, tau=7):
    """Generate temporal sequence by interpolating between original and flipped image"""
    if tau == 1:
        return [original_horizontal_flip(img)]

    img_flipped = original_horizontal_flip(img)
    return [interp_images(img, img_flipped, t / (tau - 1)) for t in range(tau)]


def main():
    """Main function to visualize horizontal flip augmentation"""
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
        # Add some pattern to make flip effects more visible
        pixels = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                r = int(255 * (i / img.size[0]))
                g = int(255 * (j / img.size[1]))
                b = int(255 * ((i + j) / (img.size[0] + img.size[1])))
                pixels[i, j] = (r, g, b)

    # Traditional augmentation (tau=1)
    print("\nGenerating traditional augmentation (tau=1)...")
    traditional_aug = time_aug_horizontal_flip(img, tau=1)

    # Temporal sequence (tau=5)
    print("Generating temporal sequence (tau=5)...")
    temporal_seq = time_aug_horizontal_flip(img, tau=5)

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
    for i, flipped_img in enumerate(temporal_seq):
        axes[1, i].imshow(flipped_img)
        axes[1, i].set_title(f'Frame {i + 1}/5')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('flip_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'flip_visualization.png'")

    # Show parameter progression for the temporal sequence
    print("\nParameter progression for temporal sequence:")
    for t in range(5):
        alpha = t / (5 - 1)
        print(f"Frame {t + 1}: blend factor (alpha) = {alpha:.3f}")


if __name__ == "__main__":
    main()