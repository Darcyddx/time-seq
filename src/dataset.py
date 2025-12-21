import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import random
from collections import defaultdict
import os
import imageio
import numpy as np
from PIL import Image

from augmentations.cut_out import time_aug_cutout
from augmentations.gaussian_blur import time_aug_gaussian_blur
from augmentations.horizontal_flip import time_aug_horizontal_flip
from augmentations.jitter import time_aug_jitter
from augmentations.rotation import time_aug_rotate
from augmentations.shear import time_aug_shear
from augmentations.translate import time_aug_translate
from augmentations.zoom import time_aug_zoom

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class SoyageingDataset(Dataset):
    """Base class for soyaging datasets"""

    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'test'

        # Read annotation file
        anno_file_path = os.path.join(self.root, 'anno', self.mode + '.txt')
        with open(anno_file_path, 'r') as anno_txt_file:
            self.labels = []
            self.imgs_name = []
            for line in anno_txt_file:
                parts = line.strip().split(' ')
                self.imgs_name.append(parts[0])
                self.labels.append(int(parts[1]) - 1)  # Convert to 0-based indexing

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'images', self.imgs_name[index])
        img, target, imgname = imageio.imread(img_path), self.labels[index], self.imgs_name[index]

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)

        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs_name)

def _create_soybean_dataset_class(region_name):
    """Factory function to create soybean aging dataset classes"""
    class_name = f'soybean_aging_{region_name}'
    return type(class_name, (SoyageingDataset,), {})


# Generate the soybean aging dataset classes
soybean_aging_R1 = _create_soybean_dataset_class('R1')
soybean_aging_R3 = _create_soybean_dataset_class('R3')
soybean_aging_R4 = _create_soybean_dataset_class('R4')
soybean_aging_R5 = _create_soybean_dataset_class('R5')
soybean_aging_R6 = _create_soybean_dataset_class('R6')


def custom_collate_fn_with_support(batch):
    frames = []
    labels = []
    original_images = []
    indices = []
    all_support_frames = []
    all_support_labels = []

    for item in batch:
        frames.append(item[0])
        labels.append(item[1])
        original_images.append(item[2])
        indices.append(item[3])

        support_frames = item[4]
        support_labels = item[5]

        if support_frames is not None:
            all_support_frames.append(support_frames)
        if support_labels is not None:
            all_support_labels.append(support_labels)

    frames = default_collate(frames)
    labels = default_collate(labels)
    indices = default_collate(indices)

    support_frames_batch = None
    support_labels_batch = None

    if all_support_frames:
        support_frames_batch = default_collate(all_support_frames)
    if all_support_labels:
        support_labels_batch = default_collate(all_support_labels)

    return frames, labels, original_images, indices, support_frames_batch, support_labels_batch


def custom_collate_fn(batch):
    if len(batch[0]) == 6:
        return custom_collate_fn_with_support(batch)

    frames = []
    labels = []
    original_images = []
    indices = []

    for item in batch:
        frames.append(item[0])
        labels.append(item[1])
        original_images.append(item[2])
        indices.append(item[3])

    frames = default_collate(frames)
    labels = default_collate(labels)
    indices = default_collate(indices)

    return frames, labels, original_images, indices


class PseudoVideoDataset(Dataset):

    def __init__(self, dataset, num_frames, crop_size=224, use_imagenet_norm=False,
                 num_same_class_samples=None, support_mode=False):
        self.dataset = dataset
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.num_same_class_samples = num_same_class_samples
        self.support_mode = support_mode

        if use_imagenet_norm:
            self.norm_mean = IMAGENET_MEAN
            self.norm_std = IMAGENET_STD
        else:
            self.norm_mean = MEAN
            self.norm_std = STD

        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        frames = self._generate_smooth_video_sequence(image)
        frames = torch.stack(frames)

        support_frames = None
        support_labels = None

        if self.support_mode and self.num_same_class_samples is not None:
            support_frames, support_labels = self._generate_support_samples(label, idx)

        return frames, label, image, idx, support_frames, support_labels

    def _generate_support_samples(self, target_label, current_idx):
        same_class_indices = self.class_to_indices[target_label]

        # Always exclude the current sample
        available_indices = [idx for idx in same_class_indices if idx != current_idx]

        if len(available_indices) == 0:
            # If no other samples available, return empty tensors
            empty_frames = torch.empty(0, self.num_frames, 3, self.crop_size, self.crop_size)
            empty_labels = torch.empty(0, dtype=torch.long)
            return empty_frames, empty_labels

        # Sample indices with replacement if needed
        if len(available_indices) < self.num_same_class_samples:
            # Sample with replacement from available indices
            selected_indices = []
            for _ in range(self.num_same_class_samples):
                selected_idx = random.choice(available_indices)
                selected_indices.append(selected_idx)
        else:
            # Sample without replacement from available indices
            selected_indices = random.sample(available_indices, self.num_same_class_samples)

        # Generate frames for all selected support samples
        support_frames_list = []
        support_labels_list = []

        for support_idx in selected_indices:
            support_image, support_label = self.dataset[support_idx]
            support_frame_sequence = self._generate_smooth_video_sequence(support_image)
            support_frame_sequence = torch.stack(support_frame_sequence)

            support_frames_list.append(support_frame_sequence)
            support_labels_list.append(support_label)

        # Stack all support frames and labels
        support_frames = torch.stack(support_frames_list)  # (num_same_class_samples, num_frames, 3, H, W)
        support_labels = torch.tensor(support_labels_list, dtype=torch.long)  # (num_same_class_samples,)

        return support_frames, support_labels

    def get_traditional_frame(self, image):
        """Generate a single frame with traditional augmentation from an image."""
        return self._generate_smooth_video_sequence_single_frame(image)

    def get_same_class_samples(self, target_label, num_samples):
        """Get random samples from the same class as target_label."""
        same_class_indices = self.class_to_indices[target_label]

        if len(same_class_indices) == 0:
            return []

        # Sample random indices
        num_to_sample = min(num_samples, len(same_class_indices))
        sampled_indices = random.sample(same_class_indices, num_to_sample)

        # Return the actual samples
        samples = []
        for idx in sampled_indices:
            image, label = self.dataset[idx]
            frames = self._generate_smooth_video_sequence(image)
            frames = torch.stack(frames)
            samples.append((frames, label))

        return samples

    def _get_augmentation_functions(self, tau):
        """Get all augmentation functions with given tau value."""
        return [
            ('horizontal_flip', lambda img: time_aug_horizontal_flip(img, tau=tau)),
            ('zoom', lambda img: time_aug_zoom(img, tau=tau)),
            ('rotation', lambda img: time_aug_rotate(img, tau=tau)),
            ('color_jitter', lambda img: time_aug_jitter(img, tau=tau)),
            ('shear', lambda img: time_aug_shear(img, tau=tau)),
            ('translate', lambda img: time_aug_translate(img, tau=tau)),
            ('gaussian_blur', lambda img: time_aug_gaussian_blur(img, tau=tau)),
            ('cutout', lambda img: time_aug_cutout(img, tau=tau))
        ]

    def _apply_augmentations(self, image, tau):
        """Apply random augmentations to image and return processed frames."""
        aug_functions = self._get_augmentation_functions(tau)

        # Randomly decide which augmentations to apply (50% chance each)
        selected_augs = []
        for name, func in aug_functions:
            if random.random() < 0.5:
                selected_augs.append((name, func))

        # If no augmentations were selected, randomly choose one
        if not selected_augs:
            selected_aug = random.choice(aug_functions)
            selected_augs = [selected_aug]

        # Randomly shuffle the order of selected augmentations
        random.shuffle(selected_augs)

        # Apply the selected augmentations in random order
        current_image = image
        for name, aug_func in selected_augs[:-1]:
            augmented_sequence = aug_func(current_image)
            # Use the last frame from the sequence as input for the next augmentation
            current_image = augmented_sequence[-1]

        # Apply the final augmentation and get the complete sequence
        final_aug_name, final_aug_func = selected_augs[-1]
        return final_aug_func(current_image)

    def _convert_to_tensor_frames(self, pil_images):
        """Convert PIL images to normalized tensor frames."""
        frames = []
        for img in pil_images:
            # Convert to tensor and normalize
            frame = transforms.functional.to_tensor(img)
            frame = transforms.functional.normalize(frame, self.norm_mean, self.norm_std)
            frames.append(frame)
        return frames

    def _generate_smooth_video_sequence_single_frame(self, image):
        """Generate a single frame with traditional augmentation (num_frames=1)."""
        # Apply RandomResizedCrop with configurable size
        image = transforms.RandomResizedCrop(self.crop_size)(image)

        # Apply augmentations and get single frame
        augmented_images = self._apply_augmentations(image, tau=1)
        final_image = augmented_images[0]  # Take the single frame

        # Convert to tensor and normalize
        frame = transforms.functional.to_tensor(final_image)
        frame = transforms.functional.normalize(frame, self.norm_mean, self.norm_std)

        return frame

    def _generate_smooth_video_sequence(self, image):
        """Generate a sequence of frames using temporal augmentations."""
        image = transforms.RandomResizedCrop(self.crop_size)(image)

        # Apply augmentations and get sequence
        augmented_images = self._apply_augmentations(image, tau=self.num_frames)

        # Convert each PIL image to tensor and normalize
        return self._convert_to_tensor_frames(augmented_images)


def _create_dataloader(dataset, batch_size, num_workers, shuffle=True, collate_fn=None):
    """Helper function to create DataLoader with common settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2 if shuffle else None,
        collate_fn=collate_fn
    )


def _get_dataset_loaders(train_root, test_root, batch_size, num_frames, num_workers, test_resize=True, crop_size=224,
                         resize_size=256, use_imagenet_norm=False, num_same_class_samples=None, support_mode=False):
    """Generic function to create train and test loaders for any dataset."""
    # Set normalization parameters
    norm_mean = IMAGENET_MEAN if use_imagenet_norm else MEAN
    norm_std = IMAGENET_STD if use_imagenet_norm else STD

    # Traditional test transforms
    test_transforms = [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
    if test_resize:
        test_transforms.insert(0, transforms.Resize(resize_size))
        test_transforms.insert(1, transforms.CenterCrop(crop_size))

    transform_test = transforms.Compose(test_transforms)

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_root, transform=None)
    test_dataset = datasets.ImageFolder(root=test_root, transform=transform_test)

    # Training dataset: pseudo-video sequences with configurable crop size and support samples
    train_pseudo_dataset = PseudoVideoDataset(
        train_dataset,
        num_frames,
        crop_size=crop_size,
        use_imagenet_norm=use_imagenet_norm,
        num_same_class_samples=num_same_class_samples,
        support_mode=support_mode
    )

    train_loader = _create_dataloader(
        train_pseudo_dataset, batch_size, num_workers, shuffle=True, collate_fn=custom_collate_fn
    )

    test_loader = _create_dataloader(
        test_dataset, batch_size, num_workers, shuffle=False
    )

    return train_loader, test_loader


def _get_soyaging_dataset_loaders(train_root, test_root, batch_size, num_frames, num_workers, dataset_class,
                                  crop_size=448, resize_size=600, use_imagenet_norm=True,
                                  num_same_class_samples=None, support_mode=False):
    """Function to create train and test loaders for soyaging datasets using custom dataset classes."""
    # Set normalization parameters
    norm_mean = IMAGENET_MEAN if use_imagenet_norm else MEAN
    norm_std = IMAGENET_STD if use_imagenet_norm else STD

    # Traditional test transforms
    test_transforms = [
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]
    transform_test = transforms.Compose(test_transforms)

    # Load datasets using custom dataset classes
    train_dataset = dataset_class(root=train_root, is_train=True, transform=None)
    test_dataset = dataset_class(root=test_root, is_train=False, transform=transform_test)

    # Training dataset: pseudo-video sequences with configurable crop size and support samples
    train_pseudo_dataset = PseudoVideoDataset(
        train_dataset,
        num_frames,
        crop_size=crop_size,
        use_imagenet_norm=use_imagenet_norm,
        num_same_class_samples=num_same_class_samples,
        support_mode=support_mode
    )

    train_loader = _create_dataloader(
        train_pseudo_dataset, batch_size, num_workers, shuffle=True, collate_fn=custom_collate_fn
    )

    test_loader = _create_dataloader(
        test_dataset, batch_size, num_workers, shuffle=False
    )

    return train_loader, test_loader


# Updated specific loader functions to accept the new parameters
def get_soyageing_R1_loaders(batch_size, num_frames, num_workers=4, num_same_class_samples=None, support_mode=False):
    return _get_soyaging_dataset_loaders('./soy_aging/R1', './soy_aging/R1', batch_size, num_frames, num_workers,
                                         soybean_aging_R1, crop_size=448, resize_size=600, use_imagenet_norm=True,
                                         num_same_class_samples=num_same_class_samples, support_mode=support_mode)


def get_soyageing_R3_loaders(batch_size, num_frames, num_workers=4, num_same_class_samples=None, support_mode=False):
    return _get_soyaging_dataset_loaders('./soy_aging/R3', './soy_aging/R3', batch_size, num_frames, num_workers,
                                         soybean_aging_R3, crop_size=448, resize_size=600, use_imagenet_norm=True,
                                         num_same_class_samples=num_same_class_samples, support_mode=support_mode)


def get_soyageing_R4_loaders(batch_size, num_frames, num_workers=4, num_same_class_samples=None, support_mode=False):
    return _get_soyaging_dataset_loaders('./soy_aging/R4', './soy_aging/R4', batch_size, num_frames, num_workers,
                                         soybean_aging_R4, crop_size=448, resize_size=600, use_imagenet_norm=True,
                                         num_same_class_samples=num_same_class_samples, support_mode=support_mode)


def get_soyageing_R5_loaders(batch_size, num_frames, num_workers=4, num_same_class_samples=None, support_mode=False):
    return _get_soyaging_dataset_loaders('./soy_aging/R5', './soy_aging/R5', batch_size, num_frames, num_workers,
                                         soybean_aging_R5, crop_size=448, resize_size=600, use_imagenet_norm=True,
                                         num_same_class_samples=num_same_class_samples, support_mode=support_mode)


def get_soyageing_R6_loaders(batch_size, num_frames, num_workers=4, num_same_class_samples=None, support_mode=False):
    return _get_soyaging_dataset_loaders('./soy_aging/R6', './soy_aging/R6', batch_size, num_frames, num_workers,
                                         soybean_aging_R6, crop_size=448, resize_size=600, use_imagenet_norm=True,
                                         num_same_class_samples=num_same_class_samples, support_mode=support_mode)


def get_flowers_loaders(batch_size, num_frames, num_workers=4, num_same_class_samples=None, support_mode=False):
    return _get_dataset_loaders('./flower/train', './flower/test', batch_size, num_frames, num_workers,
                                num_same_class_samples=num_same_class_samples, support_mode=support_mode)


def get_stanford_dogs_loaders(batch_size, num_frames, num_workers=4, num_same_class_samples=None, support_mode=False):
    return _get_dataset_loaders('./cropped/train', './cropped/test', batch_size, num_frames, num_workers,
                                test_resize=False, num_same_class_samples=num_same_class_samples,
                                support_mode=support_mode, use_imagenet_norm=True)


def get_stanford_cars_loaders(batch_size, num_frames, num_workers=4, num_same_class_samples=None, support_mode=False):
    return _get_dataset_loaders('./car_data/car_data/train', './car_data/car_data/test', batch_size, num_frames,
                                num_workers, num_same_class_samples=num_same_class_samples, support_mode=support_mode)