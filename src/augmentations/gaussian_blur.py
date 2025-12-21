import torchvision.transforms.functional as F
import random
from PIL import Image


def apply_gaussian_blur_with_sigma(img, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to PIL image with specific sigma value"""
    return F.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)


def original_gaussian_blur(img, kernel_size=5, sigma_range=(10.0, 20.0)):
    """Apply Gaussian blur to PIL image with random sigma from range"""
    target_sigma = random.uniform(sigma_range[0], sigma_range[1])
    return F.gaussian_blur(img, kernel_size=kernel_size, sigma=target_sigma)


def time_aug_gaussian_blur(img, tau=7, kernel_size=11, sigma_range=(20.0, 30.0)):
    """Generate temporal sequence by gradually blurring from original to target blur level"""
    if tau == 1:
        # Traditional augmentation: single random blur
        target_sigma = random.uniform(sigma_range[0], sigma_range[1])
        return [apply_gaussian_blur_with_sigma(img, kernel_size, target_sigma)]

    # Generate random target sigma for the final frame
    target_sigma = random.uniform(sigma_range[0], sigma_range[1])

    # Create sequence of sigma values that gradually change from 0 to target_sigma
    # Start from very small sigma (almost no blur) to target_sigma
    min_sigma = 0.1  # Very small blur to avoid issues with sigma=0
    sigma_values = [min_sigma + (target_sigma - min_sigma) * t / (tau - 1) for t in range(tau)]

    # Apply Gaussian blur to each frame using specific sigma values
    return [apply_gaussian_blur_with_sigma(img, kernel_size, sigma) for sigma in sigma_values]


def main():
    """Main function to visualize Gaussian blur augmentation"""
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
    traditional_aug = time_aug_gaussian_blur(img, tau=1)

    # Temporal sequence (tau=5)
    print("Generating temporal sequence (tau=5)...")
    temporal_seq = time_aug_gaussian_blur(img, tau=5)

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
    for i, blurred_img in enumerate(temporal_seq):
        axes[1, i].imshow(blurred_img)
        axes[1, i].set_title(f'Frame {i + 1}/5')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('gaussian_blur_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'gaussian_blur_visualization.png'")

    # Show sigma progression for the temporal sequence
    print("\nSigma progression for temporal sequence:")
    target_sigma = random.uniform(20.0, 30.0)
    min_sigma = 0.1
    sigma_values = [min_sigma + (target_sigma - min_sigma) * t / (5 - 1) for t in range(5)]
    for i, sigma in enumerate(sigma_values):
        print(f"Frame {i + 1}: σ = {sigma:.2f}")

    # # Demonstrate different variations
    # print("\nDemonstrating alternative version...")
    # alt_temporal_seq = time_aug_gaussian_blur_alternative(img, tau=5)
    #
    # print("Demonstrating kernel variation version...")
    # kernel_var_seq = time_aug_gaussian_blur_with_kernel_variation(img, tau=5)


if __name__ == "__main__":
    main()