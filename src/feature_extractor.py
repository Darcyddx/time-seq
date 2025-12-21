import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import torch
import torch.nn as nn
import open_clip


class FeatureExtractor(nn.Module):
    """Frozen CLIP visual encoder used as a feature extractor.
    - Tries to load local weights first (default: ./clip_vit_b_16_pretrained.pth).
    - If the file does not exist, falls back to OpenCLIP pretrained weights
      (requires internet on first run to download the weights).
    """

    def __init__(
        self,
        local_weights_path: str = "./clip_vit_b_16_pretrained.pth",
        fallback_pretrained: str = "openai",
        model_name: str = "ViT-B-16",
    ):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        local_path = Path(local_weights_path)

        if local_path.exists():
            # Create model without pretrained weights then load local state_dict
            clip_model = open_clip.create_model(model_name, pretrained=False)
            state_dict = torch.load(str(local_path), map_location=device)
            clip_model.load_state_dict(state_dict)
        else:
            # Fallback: use official pretrained weights from OpenCLIP
            clip_model = open_clip.create_model(model_name, pretrained=fallback_pretrained)

        clip_model.to(device)
        self.visual_encoder = clip_model.visual

        # Freeze parameters (feature extractor is fixed)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Train mode: pseudo video sequence (B, T, C, H, W)
        if x.dim() == 5:
            b, t = x.shape[:2]
            x = x.view(-1, *x.shape[2:])          # (B*T, C, H, W)
            feats = self.visual_encoder(x)        # (B*T, D)
            feats = feats.view(b, t, -1)          # (B, T, D)
            return feats

        # Test mode: single images (B, C, H, W)
        return self.visual_encoder(x)

    def get_feature_dim(self) -> int:
        return 512


def get_feature_extractor(dataset_name: str) -> FeatureExtractor:
    """Factory for feature extractor.

    This open-source version only supports Flowers102.
    """
    if dataset_name != "flowers":
        raise ValueError(
            f"Open-source build supports only dataset='flowers', got: {dataset_name}"
        )
    return FeatureExtractor()
