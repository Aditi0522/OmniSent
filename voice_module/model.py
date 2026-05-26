import torch
import torch.nn as nn
from transformers import WavLMModel

class VoiceEmotionClassifier(nn.Module):
    "WaveLM-Base + classification head for 7-class emotion Taxonomy"

    def __init__(self, num_classes = 7, freeze_feature_extractor = True):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        hidden_size = self.wavlm.config.hidden_size

        if freeze_feature_extractor:
            self.wavlm.feature_extractor._freeze_parameters()

        self.classifier = nn.Sequential(
                nn.Linear(hidden_size,256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256,num_classes),
        )

    def forward(self, input_values):
        "input_values: (batch, seq_lanes) raw waveform at 16 KHz"
        outputs = self.wavlm(input_values)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

    def freeze_transformer(self):
        for param in self.wavlm.parameters():
            param.required_grad = False

    def unfreeze_transformer(self):
        for param in self.wavlm.parameters():
            param.requires_grad = True

def load_voice_model(model_path, device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VoiceEmotionClassifier(num_classes=7)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model



