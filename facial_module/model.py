import torch
import torch.nn as nn
import logging
from timm import create_model

class EmotiEffLibFineTuned(nn.Module):
    """EfficientNet-B0 backbone + 7-class emotion classifier head."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=0
        )
        feature_dim = self.backbone.num_features  # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def load_model(model_path: str, device: torch.device = None) -> EmotiEffLibFineTuned:
    """Load fine-tuned model from checkpoint. Returns model in eval mode."""
    logger = logging.getLogger(__name__)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EmotiEffLibFineTuned(num_classes=7)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"Emotion model loaded from {model_path} on {device}")
    return model

