"""Configuration for the baseline; paths in JSON are relative to the project root."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CrowdConfig:
    pretrained_path: str = "checkpoints/OmniStream"
    freeze_backbone: bool = True
    head_channels: tuple = (64, 32, 16)
    input_is_normalized: bool = False
    temporal_length: int = 16
    density_loss_weight: float = 1.0
    count_loss_weight: float = 0.0
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4

    def __post_init__(self):
        self.head_channels = tuple(self.head_channels)
        if len(self.head_channels) != 3 or any(c <= 0 for c in self.head_channels):
            raise ValueError("head_channels must contain three positive channel widths.")
        if self.temporal_length < 1:
            raise ValueError("temporal_length must be positive.")
        if self.density_loss_weight <= 0 or self.count_loss_weight < 0:
            raise ValueError("Use a positive density weight and nonnegative count weight.")
        if min(self.learning_rate, self.backbone_learning_rate) <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rates must be positive; weight_decay must be nonnegative.")

    @property
    def checkpoint_directory(self):
        path = Path(self.pretrained_path).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))
