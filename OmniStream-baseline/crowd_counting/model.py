"""Video -> official OmniStream patch tokens -> TMTB head -> per-frame counts."""

from contextlib import nullcontext

import torch
from torch import nn

from model import OmnistreamMultiFrameTransformer
from model.configuration_omnistream import OmnistreamConfig
from .config import CrowdConfig
from .tmtb_head import CountingHead


class OmniStreamCrowd(nn.Module):
    """Clip-based, temporally causal feature extraction with a shared 2D head.

    Input: uint8 RGB in [0,255] or float RGB in [0,1], shape B,T,3,H,W.
    Set input_is_normalized=True ONLY for already ImageNet-normalized floats.
    Output density: B,T,1,H/4,W/4; counts: B,T (for the patch-16 checkpoint).
    No resize, center crop, frame pooling or additional positional encoding is used.
    """

    def __init__(self, backbone, config=None):
        super().__init__()
        self.config = config or CrowdConfig()
        self.backbone = backbone
        self.patch_size = int(backbone.config.patch_size)
        if self.patch_size % 4:
            raise ValueError("The TMTB head requires a patch size divisible by four.")
        self.output_stride = self.patch_size // 4
        self.head = CountingHead(
            in_channels=backbone.config.hidden_size,
            out_channels=1,
            inter_layer=list(self.config.head_channels),
        )
        # Match initialization used by TMTB's MAMBA4CC._init_weights, rather than
        # its unused CountingHead.init_weights method (which uses a different std).
        for module in self.head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        self.set_backbone_frozen(self.config.freeze_backbone)

    @classmethod
    def from_pretrained(cls, config=None):
        """Load a local official checkpoint strictly; never silently train from scratch."""
        config = config or CrowdConfig()
        directory = config.checkpoint_directory
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(
                f"OmniStream config.json is missing in {directory}. "
                "See README.md for the manual weight download command."
            )
        if not any(directory.glob("*.safetensors")):
            raise FileNotFoundError(f"No OmniStream .safetensors weights found in {directory}.")
        backbone, info = OmnistreamMultiFrameTransformer.from_pretrained(
            str(directory), local_files_only=True, output_loading_info=True
        )
        discrepancies = {name: info.get(name) for name in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
        ) if info.get(name)}
        if discrepancies:
            raise RuntimeError(f"Official checkpoint and source code do not match: {discrepancies}")
        return cls(backbone, config)

    @classmethod
    def from_backbone_config(cls, backbone_config, config=None):
        """Construct architecture for restoration; weights MUST be loaded afterward."""
        if isinstance(backbone_config, dict):
            backbone_config = OmnistreamConfig(**backbone_config)
        return cls(OmnistreamMultiFrameTransformer(backbone_config), config)

    def set_backbone_frozen(self, frozen=True):
        self.config.freeze_backbone = bool(frozen)
        self.backbone.requires_grad_(not frozen)
        self.backbone.train(self.training and not frozen)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Frozen extraction must also disable the upstream training-only RoPE jitter.
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def _apply(self, fn):
        result = super()._apply(fn)
        # Upstream positional caches are plain dictionaries, not registered buffers.
        # Invalidate them after device/dtype changes so cached tensors cannot stay on CPU.
        extractor = self.backbone.patch_embed
        if extractor.position_getter is not None:
            extractor.position_getter.position_cache.clear()
        if extractor.rope is not None:
            extractor.rope.frequency_cache.clear()
        return result

    def forward(self, pixel_values):
        if pixel_values.ndim != 5 or pixel_values.shape[2] != 3:
            raise ValueError("Expected RGB input with shape (B,T,3,H,W).")
        batch, frames, _, height, width = pixel_values.shape
        if min(batch, frames, height, width) < 1:
            raise ValueError("Input dimensions must be nonzero.")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"H and W must be multiples of patch_size={self.patch_size}; got {height}x{width}.")
        if frames > self.config.temporal_length:
            raise ValueError("Clip exceeds temporal_length; configure one fixed temporal scale for the experiment.")
        if self.config.input_is_normalized:
            if not pixel_values.is_floating_point():
                raise ValueError("Normalized inputs must be floating point.")
            inputs = pixel_values.float()
        else:
            if pixel_values.dtype == torch.uint8:
                inputs = pixel_values.float() / 255.0
            elif pixel_values.is_floating_point():
                inputs = pixel_values.float()
            else:
                raise ValueError("Use uint8 [0,255] or floating RGB [0,1].")
            if not torch.isfinite(inputs).all() or (inputs < 0).any() or (inputs > 1).any():
                raise ValueError("Unnormalized floating inputs must be finite RGB in [0,1].")
            inputs = (inputs - self.image_mean) / self.image_std
        if not torch.isfinite(inputs).all():
            raise ValueError("Input contains NaN or infinity.")
        # The official wrapper uses .view(), so contiguous input is required.
        context = torch.no_grad() if self.config.freeze_backbone else nullcontext()
        with context:
            outputs = self.backbone(
                pixel_values=inputs.contiguous(), return_dict=True,
                use_cache=False, total_length=self.config.temporal_length,
            )
            # Already patch-only and normalized in the official source.
            # hidden_states, unlike last_hidden_state, still contain special tokens.
            patches = outputs.last_hidden_state
            del outputs
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        expected = (batch, frames, grid_h * grid_w, self.backbone.config.hidden_size)
        if tuple(patches.shape) != expected:
            raise RuntimeError(f"Unexpected patch layout: {tuple(patches.shape)}, expected {expected}.")
        features = patches.reshape(batch * frames, grid_h, grid_w, -1).permute(0, 3, 1, 2).contiguous()
        density = self.head(features)
        density = density.reshape(batch, frames, 1, height // self.output_stride, width // self.output_stride)
        # Sum in float32 under autocast; values are count mass with no hidden scale factor.
        counts = density.float().sum(dim=(2, 3, 4))
        return {"density": density, "counts": counts}
