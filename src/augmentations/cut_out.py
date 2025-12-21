import random
import numpy as np
from PIL import Image, ImageDraw


def apply_cutout_mask(img, mask_x, mask_y, mask_w, mask_h, fill_color=(0, 0, 0)):
    """Apply cutout mask to PIL image"""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    # Draw rectangle (cutout area)
    draw.rectangle([mask_x, mask_y, mask_x + mask_w, mask_y + mask_h], fill=fill_color)

    return img_copy


def generate_random_cutout_params(img_width, img_height, max_size_ratio=0.3):
    """Generate random cutout parameters for traditional augmentation"""
    # Random size (up to max_size_ratio of image dimensions)
    max_w = int(img_width * max_size_ratio)
    max_h = int(img_height * max_size_ratio)

    mask_w = random.randint(max_w // 4, max_w)
    mask_h = random.randint(max_h // 4, max_h)

    # Random position
    mask_x = random.randint(0, img_width - mask_w)
    mask_y = random.randint(0, img_height - mask_h)

    return mask_x, mask_y, mask_w, mask_h


def pattern1_growing_moving_mask(img, tau):
    """
    Pattern 1: Mask size gradually increases to full size, position moves from center to final position
    """
    img_width, img_height = img.size

    # Generate final mask parameters
    final_x, final_y, final_w, final_h = generate_random_cutout_params(img_width, img_height, max_size_ratio=0.3)

    # Initial position (center of image)
    center_x = img_width // 2
    center_y = img_height // 2

    frames = []

    for t in range(tau):
        progress = t / (tau - 1) if tau > 1 else 1.0

        # Interpolate size (from small to final size)
        current_w = int(final_w * 0.1 + (final_w - final_w * 0.1) * progress)
        current_h = int(final_h * 0.1 + (final_h - final_h * 0.1) * progress)

        # Interpolate position (from center to final position)
        current_x = int(center_x - current_w // 2 + (final_x - (center_x - current_w // 2)) * progress)
        current_y = int(center_y - current_h // 2 + (final_y - (center_y - current_h // 2)) * progress)

        # Ensure bounds
        current_x = max(0, min(current_x, img_width - current_w))
        current_y = max(0, min(current_y, img_height - current_h))

        frame = apply_cutout_mask(img, current_x, current_y, current_w, current_h)
        frames.append(frame)

    return frames


def pattern2_fixed_size_moving_mask(img, tau):
    """
    Pattern 2: Fixed mask size, position moves from center to final position
    """
    img_width, img_height = img.size

    # Generate final mask parameters
    final_x, final_y, final_w, final_h = generate_random_cutout_params(img_width, img_height, max_size_ratio=0.3)

    # Initial position (center of image)
    center_x = img_width // 2 - final_w // 2
    center_y = img_height // 2 - final_h // 2

    frames = []

    for t in range(tau):
        progress = t / (tau - 1) if tau > 1 else 1.0

        # Fixed size, moving position
        current_w = final_w
        current_h = final_h

        # Interpolate position (from center to final position)
        current_x = int(center_x + (final_x - center_x) * progress)
        current_y = int(center_y + (final_y - center_y) * progress)

        # Ensure bounds
        current_x = max(0, min(current_x, img_width - current_w))
        current_y = max(0, min(current_y, img_height - current_h))

        frame = apply_cutout_mask(img, current_x, current_y, current_w, current_h)
        frames.append(frame)

    return frames


def pattern3_progressive_coverage(img, tau, max_coverage=0.3):
    """
    Pattern 3: Progressive coverage - gradually increase covered area up to max_coverage
    Each frame builds upon the previous one by adding more rectangular masks
    """
    img_width, img_height = img.size
    total_area = img_width * img_height

    # Calculate coverage progression
    if tau == 1:
        coverage_levels = [max_coverage]
    else:
        # Calculate step size for progressive coverage
        min_coverage = max_coverage * 0.3  # Start with 30% of max coverage
        step_size = (max_coverage - min_coverage) / (tau - 1)
        coverage_levels = [min_coverage + i * step_size for i in range(tau)]

    # Generate all masks in reverse order (from max coverage to min coverage)
    all_masks = []  # List to store all rectangular masks

    # Start with the final frame (maximum coverage)
    final_target_area = int(total_area * max_coverage)
    current_area = 0
    attempts = 0
    max_attempts = 100

    while current_area < final_target_area and attempts < max_attempts:
        # Remaining area to cover
        remaining_area = final_target_area - current_area

        # Generate random rectangle size
        max_rect_area = min(remaining_area, total_area * 0.08)  # Max 8% per rectangle
        min_rect_area = min(max_rect_area, total_area * 0.005)  # Min 0.5% per rectangle

        if max_rect_area <= 0:
            break

        rect_area = random.randint(int(min_rect_area), int(max_rect_area))

        # Generate rectangle dimensions
        aspect_ratio = random.uniform(0.5, 2.0)  # More balanced aspect ratios
        rect_w = int(np.sqrt(rect_area * aspect_ratio))
        rect_h = int(rect_area / rect_w) if rect_w > 0 else 1

        # Ensure reasonable size constraints
        rect_w = max(5, min(rect_w, img_width // 2))
        rect_h = max(5, min(rect_h, img_height // 2))

        # Generate random position
        rect_x = random.randint(0, max(0, img_width - rect_w))
        rect_y = random.randint(0, max(0, img_height - rect_h))

        # Store mask parameters
        mask_params = (rect_x, rect_y, rect_w, rect_h)
        all_masks.append(mask_params)

        current_area += rect_w * rect_h
        attempts += 1

    # Now create frames by selecting subsets of masks
    frames = []

    for coverage in coverage_levels:
        target_area = int(total_area * coverage)

        # Select masks to achieve target coverage
        frame = img.copy()
        used_area = 0

        for mask_params in all_masks:
            rect_x, rect_y, rect_w, rect_h = mask_params
            mask_area = rect_w * rect_h

            # Check if adding this mask would exceed target area
            if used_area + mask_area <= target_area:
                frame = apply_cutout_mask(frame, rect_x, rect_y, rect_w, rect_h)
                used_area += mask_area
            else:
                break

        frames.append(frame)

    return frames


def time_aug_cutout(img, tau=7, max_coverage=0.3):
    """
    Generate temporal cutout sequence with one of three patterns randomly selected
    """
    # Randomly select one of three patterns
    pattern = random.randint(1, 3)

    if pattern == 1:
        return pattern1_growing_moving_mask(img, tau)
    elif pattern == 2:
        return pattern2_fixed_size_moving_mask(img, tau)
    else:  # pattern == 3
        return pattern3_progressive_coverage(img, tau, max_coverage)


def visualize_all_patterns(img, tau=5, max_coverage=0.3):
    """
    Generate all three patterns for comparison
    """
    patterns = {
        'Pattern 1 (Growing & Moving)': pattern1_growing_moving_mask(img, tau),
        'Pattern 2 (Fixed Size Moving)': pattern2_fixed_size_moving_mask(img, tau),
        'Pattern 3 (Progressive Coverage)': pattern3_progressive_coverage(img, tau, max_coverage)
    }

    return patterns


def main():
    """Main function to visualize cutout augmentation"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Load the sample image
    img_path = "wing.jpg"
    try:
        img = Image.open(img_path)
        print(f"Loaded image: {img_path}, Size: {img.size}")
    except FileNotFoundError:
        print(f"Error: Could not find image at {img_path}")
        return

    # Generate traditional augmentation (tau=1)
    print("\nGenerating traditional augmentation (tau=1)...")
    traditional_aug = time_aug_cutout(img, tau=1)

    # Generate all three patterns for comparison
    print("Generating all three patterns (tau=5)...")
    all_patterns = visualize_all_patterns(img, tau=5)

    # Create visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

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

    # Rows 2-4: Three patterns
    row_idx = 1
    for pattern_name, frames in all_patterns.items():
        for i, frame in enumerate(frames):
            axes[row_idx, i].imshow(frame)
            axes[row_idx, i].set_title(f'{pattern_name}\nFrame {i + 1}/5')
            axes[row_idx, i].axis('off')
        row_idx += 1

    plt.tight_layout()
    plt.savefig('cutout_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'cutout_visualization.png'")

    # Demonstrate random pattern selection
    print("\nDemonstrating random pattern selection...")
    for i in range(5):
        frames = time_aug_cutout(img, tau=5)
        print(f"Random sequence {i + 1}: Generated {len(frames)} frames")


if __name__ == "__main__":
    main()